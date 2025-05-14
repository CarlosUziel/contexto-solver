import asyncio
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import nltk
import numpy as np
import typer
from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as rest_exceptions
from qdrant_client.http.models import Distance, VectorParams
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("taggers/averaged_perceptron_tagger")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from config.logger import app_logger as logger
from config.settings import settings

# --- Constants ---
DATA_DIR = Path(".data")
WORDS_FILE = DATA_DIR.parent / "words.txt"
GLOVE_URLS = {
    "glove.6B": ("https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"),
    "glove.twitter.27B": (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
    ),
    "glove.42B.300d": (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip"
    ),
    "glove.840B.300d": (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip"
    ),
}

DIMENSIONS = {
    "50d": 50,
    "100d": 100,
    "200d": 200,
    "300d": 300,
    "25d": 25,
}
app = typer.Typer()


def get_glove_url_and_filename(dataset_name: str) -> Tuple[str, str]:
    """Determines the download URL and expected filename for a given GloVe dataset.

    Args:
        dataset_name (str): The name of the GloVe dataset (e.g., "glove.6B.100d").

    Returns:
        Tuple[str, str]: A tuple containing the download URL and the target
            filename within the zip.

    Raises:
        ValueError: If the dataset name format is invalid or the prefix is unknown.
    """
    filename = f"{dataset_name}.txt"
    url_key = None

    if dataset_name.startswith("glove.6B."):
        url_key = "glove.6B"
    elif dataset_name.startswith("glove.twitter.27B."):
        url_key = "glove.twitter.27B"
    elif dataset_name == "glove.42B.300d":
        url_key = "glove.42B.300d"
    elif dataset_name == "glove.840B.300d":
        url_key = "glove.840B.300d"

    if url_key and url_key in GLOVE_URLS:
        return GLOVE_URLS[url_key], filename
    else:
        raise ValueError(
            f"Unknown GloVe dataset prefix or format for '{dataset_name}'. "
            f"Valid prefixes: glove.6B, glove.twitter.27B. "
            f"Valid full names: glove.42B.300d, glove.840B.300d."
        )


async def download_file_with_progress(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to a destination path with a progress bar."""
    async with httpx.AsyncClient(
        timeout=None, follow_redirects=True
    ) as client:  # Added follow_redirects=True
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with Progress(
                    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                    BarColumn(bar_width=None),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    DownloadColumn(),
                    "•",
                    TransferSpeedColumn(),
                    "•",
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task(
                        "download", total=total_size, filename=dest_path.name
                    )
                    with open(dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            logger.info(f"Successfully downloaded {dest_path.name}.")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Clean up partial download
            raise
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            raise


async def download_and_extract_glove(
    url: str, filename: str, extract_to: Path
) -> Optional[Path]:
    """Downloads a GloVe zip file and extracts the specified embeddings file.

    Ensures the target directory exists, handles existing files, downloads
    the zip archive if necessary, and extracts the required .txt file.

    Args:
        url (str): The URL to download the GloVe zip file from.
        filename (str): The specific .txt file to extract from the zip archive.
        extract_to (Path): The directory Path object where the file should be extracted.

    Returns:
        Optional[Path]: The Path object of the extracted embeddings file, or None
            if an error occurred during download or extraction.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / Path(url).name
    target_file_path = extract_to / filename

    if target_file_path.exists():
        logger.info(
            f"Embeddings file {target_file_path} already exists. Skipping download and extraction."
        )
        return target_file_path

    if not zip_path.exists():
        logger.info(f"Downloading {url} to {zip_path}...")
        try:
            await download_file_with_progress(url, zip_path)
        except Exception:
            return None  # Error handled in download_file_with_progress
    else:
        logger.info(f"Zip file {zip_path} already exists. Skipping download.")

    logger.info(f"Extracting {filename} from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if filename not in zip_ref.namelist():
                logger.error(
                    f"File '{filename}' not found in zip archive '{zip_path.name}'. "
                    f"Available files: {zip_ref.namelist()}"
                )
                return None
            zip_ref.extract(filename, path=extract_to)
        logger.info(f"Successfully extracted {filename} to {target_file_path}.")
        return target_file_path
    except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
        logger.error(f"Error extracting {filename} from {zip_path}: {e}")
        return None


def setup_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Sets up the Qdrant collection, creating it if it doesn't exist.
    Vectors are NOT normalized here; Cosine distance handles similarity appropriately.
    """
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(
            f"Collection '{collection_name}' already exists with {collection_info.points_count} points."
        )
    except (
        rest_exceptions.UnexpectedResponse,
        ValueError,
    ) as e:
        if isinstance(e, ValueError) and "not found" not in str(e).lower():
            if not (
                isinstance(e, rest_exceptions.UnexpectedResponse)
                and e.status_code == 404
            ):
                logger.error(
                    f"Unexpected error when checking collection '{collection_name}': {e}"
                )
                raise
        logger.info(
            f"Collection '{collection_name}' not found. Creating new collection..."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(
            f"Collection '{collection_name}' created successfully with vector size {vector_size}."
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred with collection '{collection_name}': {e}"
        )
        raise


def get_wordnet_pos(nltk_tag: str) -> Optional[str]:
    """Convert NLTK POS tag to a format WordNetLemmatizer can use."""
    if nltk_tag.startswith("J"):
        return wn.ADJ
    elif nltk_tag.startswith("V"):
        return wn.VERB
    elif nltk_tag.startswith("N"):
        return wn.NOUN
    elif nltk_tag.startswith("R"):
        return wn.ADV
    else:
        return None


def is_noun_word(lemma: str) -> bool:
    """Checks if a lemma can be a noun by checking all its WordNet synsets."""
    if not lemma:
        return False
    for synset in wn.synsets(lemma):
        if synset.pos() == wn.NOUN:
            return True
    return False


def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    embeddings_file: Path,
    use_only_nouns: bool,
) -> None:
    """
    Reads GloVe embeddings, lemmatizes words, averages embeddings for same lemmas,
    and upserts them to Qdrant. Optionally filters for nouns.
    Vectors are NOT normalized before upserting.
    Payload for each point will be {"word": "the_word_string"}.
    """
    lemmatizer = WordNetLemmatizer()
    lemma_embeddings_map: Dict[str, List[np.ndarray]] = {}
    processed_lines = 0

    logger.info(
        f"Processing embeddings from {embeddings_file}. Noun filtering: {'Enabled' if use_only_nouns else 'Disabled'}."
    )

    with embeddings_file.open("r", encoding="utf-8") as f:
        for line in f:
            processed_lines += 1
            if processed_lines % 50000 == 0:
                logger.info(f"Processed {processed_lines} lines from GloVe file...")

            parts = line.strip().split()
            word = parts[0]

            if not word.isalpha() or not word.isascii():
                continue

            try:
                vector = np.array([float(val) for val in parts[1:]], dtype=np.float32)
            except ValueError:
                logger.warning(
                    f"Skipping line due to non-float value for word '{word}': {line.strip()}"
                )
                continue

            lemma = lemmatizer.lemmatize(word.lower())

            if use_only_nouns:
                if not is_noun_word(lemma):
                    continue

            if lemma not in lemma_embeddings_map:
                lemma_embeddings_map[lemma] = []
            lemma_embeddings_map[lemma].append(vector)

    logger.info(
        f"Finished processing GloVe file. {len(lemma_embeddings_map)} unique lemmas found "
        f"{'after noun filtering' if use_only_nouns else ''}."
    )

    points_to_upsert = []
    unique_words_count = 0
    for lemma, vectors_list in lemma_embeddings_map.items():
        if not vectors_list:
            continue

        averaged_vector = np.mean(vectors_list, axis=0).tolist()
        point_id = str(uuid.uuid5(settings.qdrant_uuid_namespace, lemma))

        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector=averaged_vector,
                payload={"word": lemma},
            )
        )
        unique_words_count += 1

        if len(points_to_upsert) >= 256:
            client.upsert(collection_name=collection_name, points=points_to_upsert)
            logger.info(f"Upserted batch of {len(points_to_upsert)} points...")
            points_to_upsert = []

    if points_to_upsert:
        client.upsert(collection_name=collection_name, points=points_to_upsert)
        logger.info(f"Upserted final batch of {len(points_to_upsert)} points.")

    logger.info(
        f"Successfully upserted {unique_words_count} unique lemmas to '{collection_name}'."
    )


@app.command()
async def main():
    """
    Main function to download GloVe embeddings and set up the Qdrant collection.
    """
    logger.info("Starting Qdrant setup process...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        glove_url, glove_filename = get_glove_url_and_filename(settings.glove_dataset)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    try:
        dimension_str = settings.glove_dataset.split(".")[-1]
        vector_dimension = DIMENSIONS[dimension_str]
    except (IndexError, KeyError):
        logger.error(
            f"Could not determine vector dimension from GLOVE_DATASET: {settings.glove_dataset}"
        )
        return

    embeddings_file_path: Optional[Path] = None

    embeddings_file_path = await download_and_extract_glove(
        glove_url, glove_filename, DATA_DIR
    )

    if not embeddings_file_path or not embeddings_file_path.exists():
        logger.error("Failed to obtain GloVe embeddings file. Exiting.")
        return

    effective_collection_name = settings.effective_collection_name
    logger.info(f"Target Qdrant collection: '{effective_collection_name}'")

    try:
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_host_port,
            grpc_port=settings.qdrant_grpc_host_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            https=False,
        )
        qdrant_client.get_collections()
        logger.info(
            f"Successfully connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_grpc_host_port}."
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    try:
        setup_qdrant_collection(
            qdrant_client, effective_collection_name, vector_dimension
        )
        upsert_embeddings(
            qdrant_client,
            effective_collection_name,
            embeddings_file_path,
            settings.use_only_nouns,
        )
        logger.info("Qdrant setup and data upsert completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during Qdrant setup or upsert: {e}")
    finally:
        if "qdrant_client" in locals() and qdrant_client:
            qdrant_client.close()
            logger.info("Qdrant client closed.")


if __name__ == "__main__":
    asyncio.run(main())
