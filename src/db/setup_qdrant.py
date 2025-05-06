import itertools
import os
import uuid
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import grpc
import nltk
import numpy as np  # Added numpy import
import requests
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
except LookupError:
    nltk.download("wordnet")
from nltk.corpus import wordnet

from config.logger import app_logger as logger
from config.settings import Settings, settings

# --- Constants ---
DATA_DIR = Path(".data")
WORDS_FILE = DATA_DIR.parent / "words.txt"  # Path to words.txt at the project root
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
    # 1. Define the target filename based on the full dataset name
    filename = f"{dataset_name}.txt"
    url_key = None

    # 2. Determine the key to look up the URL in GLOVE_URLS based on dataset structure
    if dataset_name.startswith("glove.6B."):
        url_key = "glove.6B"
    elif dataset_name.startswith("glove.twitter.27B."):
        url_key = "glove.twitter.27B"
    elif dataset_name == "glove.42B.300d":
        url_key = "glove.42B.300d"
    elif dataset_name == "glove.840B.300d":
        url_key = "glove.840B.300d"

    # 3. Check if a valid URL key was found and retrieve the URL from the GLOVE_URLS map
    if url_key and url_key in GLOVE_URLS:
        url = GLOVE_URLS[url_key]
        return url, filename
    else:
        raise ValueError(f"Unknown or invalid GloVe dataset name: {dataset_name}")


def download_and_extract_glove(
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
    # 1. Ensure extraction directory exists and define paths
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / Path(url).name
    target_file_path = extract_to / filename

    # 2. Skip if target file already exists
    if target_file_path.exists():
        logger.info(
            f"Embeddings file {target_file_path} already exists. "
            "Skipping download and extraction."
        )
        return target_file_path

    # 3. Download the zip file if it doesn't exist locally
    if not zip_path.exists():
        logger.info(f"Downloading {url} to {zip_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            # 3a. Use rich progress bar for download
            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
            ) as progress:
                download_task = progress.add_task(
                    "download", total=total_size, filename=zip_path.name
                )
                with zip_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            progress.update(download_task, advance=len(chunk))

            logger.info("Download complete.")
        except requests.exceptions.RequestException as e:
            # 3b. Handle download errors and clean up incomplete file
            logger.error(f"Error downloading {url}: {e}")
            if zip_path.exists():
                try:
                    os.remove(zip_path)
                    logger.info(f"Removed incomplete download: {zip_path}")
                except OSError as remove_err:
                    logger.error(
                        f"Error removing incomplete download {zip_path}: {remove_err}"
                    )
            return None
    else:
        # 4. Log if zip file already exists (skip download)
        logger.info(f"Zip file {zip_path} already exists. Skipping download.")

    # 5. Extract the required file from the zip archive
    logger.info(f"Extracting {filename} from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 5a. Handle cases where filename in zip might differ
            if filename not in zip_ref.namelist():
                txt_files = [f for f in zip_ref.namelist() if f.endswith(".txt")]
                if not txt_files:
                    raise FileNotFoundError(
                        f"{filename} not found in {zip_path} and no other .txt files present."
                    )
                actual_filename_in_zip = txt_files[0]
                logger.warning(
                    f"{filename} not found directly. Extracting "
                    f"{actual_filename_in_zip} instead."
                )
                # 5b. Extract and rename the file
                zip_ref.extract(actual_filename_in_zip, path=extract_to)
                os.rename(extract_to / actual_filename_in_zip, target_file_path)
                logger.info(f"Renamed extracted file to {target_file_path}")

            # 5c. Extract the file directly if name matches
            else:
                zip_ref.extract(filename, path=extract_to)

        logger.info(f"Extraction complete: {target_file_path}")
        return target_file_path
    except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return None


def setup_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Checks if a Qdrant collection exists. If it does, it's deleted and recreated.
    Then, creates it if it doesn't exist.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name for the Qdrant collection.
        vector_size (int): The dimensionality of the vectors to be stored.

    Returns:
        None.

    Raises:
        Exception: If creating the collection fails.
    """
    try:
        # 1. Check if collection exists
        client.get_collection(collection_name=collection_name)
        logger.info(
            f"Collection '{collection_name}' already exists. Deleting and recreating."
        )
        # Delete existing collection
        client.delete_collection(collection_name=collection_name)
        # Recreate the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' recreated successfully.")

    except Exception as e:
        # 2. Check if the error is specifically 'Not Found'
        is_not_found_error = False
        if isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.NOT_FOUND:
            is_not_found_error = True
        elif isinstance(e, rest_exceptions.UnexpectedResponse) and e.status_code == 404:
            # Handle potential REST API 'Not Found' if gRPC fails or isn't used
            is_not_found_error = True

        if is_not_found_error:
            # Log cleanly if collection simply doesn't exist yet
            logger.info(
                f"Collection '{collection_name}' not found. Attempting to create."
            )
        else:
            # Log as warning for other unexpected errors during the check
            logger.warning(
                f"Warning during collection check for '{collection_name}': {e}. "
                f"Attempting to create anyway."
            )

        # 3. Attempt to create the collection
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as create_err:
            logger.error(
                f"Failed to create collection '{collection_name}': {create_err}"
            )
            raise


def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    embeddings_file: Path,
    allowed_words: Optional[set[str]] = None,
) -> None:
    """Loads word embeddings from a file and upserts them into a Qdrant collection.

    Reads the embeddings file in batches of lines, creates PointStruct objects,
    and upserts them into the specified Qdrant collection. Only valid English
    words are added.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection to upsert into.
        embeddings_file (Path): The Path object of the GloVe embeddings file
            (.txt format).
        allowed_words (Optional[set[str]]): A set of lowercase words to filter against.
                                            Only words in this set will be upserted.
                                            If None, no filtering by this set is done.

    Returns:
        None.
    """
    logger.info(
        f"Starting upsert process for {embeddings_file} into '{collection_name}'..."
    )
    # 1. Initialize variables (remove total_lines counting)
    batch_size = 4096
    total_upserted_count = 0
    skipped_lines = 0
    global_line_index = 0

    try:
        # 2. Open the file and set up the progress bar (without total)
        with (
            embeddings_file.open("r", encoding="utf-8") as f,
            Progress(
                TextColumn("[bold green]Upserting embeddings..."),
                BarColumn(bar_width=None),
                "•",
                TextColumn("{task.completed} lines processed"),
                "•",
                TimeElapsedColumn(),
            ) as progress,
        ):
            # Initialize task with total=None for indeterminate progress
            upsert_task = progress.add_task("upsert", total=None)

            # 3. Process the file in batches of lines
            while True:
                # 3a. Read the next batch_size lines using itertools.islice
                lines_batch_iter = itertools.islice(f, batch_size)
                points_to_upsert = []
                lines_processed_in_this_batch = 0

                # 3b. Process each line within the current batch
                for line in lines_batch_iter:
                    lines_processed_in_this_batch += 1
                    values = line.strip().split()
                    if len(values) < 2:
                        logger.warning(
                            f"Skipping malformed line {global_line_index + 1}: "
                            f"{line[:50]}..."
                        )
                        skipped_lines += 1
                        global_line_index += 1
                        continue

                    word_from_file = values[0].lower()  # Ensure word is lowercase
                    try:
                        # Filter by allowed_words set if provided
                        if allowed_words and word_from_file not in allowed_words:
                            logger.debug(
                                f"Skipping word '{word_from_file}' (line {global_line_index + 1}) "
                                f"as it is not in the provided allowed_words set."
                            )
                            skipped_lines += 1
                            global_line_index += 1
                            continue

                        # Check if the word is a valid English word
                        if not wordnet.synsets(
                            word_from_file
                        ):  # word_from_file is already lowercase
                            logger.debug(
                                f"Skipping non-English word {global_line_index + 1}: "
                                f"{word_from_file}"
                            )
                            skipped_lines += 1
                            global_line_index += 1
                            continue

                        vector_list = [float(val) for val in values[1:]]
                        # Normalize the vector
                        np_vector = np.array(vector_list, dtype=np.float32)
                        norm = np.linalg.norm(np_vector)
                        if norm > 0:
                            normalized_vector = (np_vector / norm).tolist()
                        else:
                            logger.warning(
                                f"Skipping word '{word_from_file}' (line {global_line_index + 1}) "
                                f"due to zero vector (cannot normalize)."
                            )
                            skipped_lines += 1
                            global_line_index += 1
                            continue

                        # UUID will be based on the lowercase word
                        point_id = str(
                            uuid.uuid5(settings.qdrant_uuid_namespace, word_from_file)
                        )
                        points_to_upsert.append(
                            models.PointStruct(
                                id=point_id,
                                vector=normalized_vector,  # Use normalized vector
                                payload={
                                    "word": word_from_file
                                },  # Store lowercase word
                            )
                        )
                    except ValueError:
                        logger.warning(
                            f"Skipping line {global_line_index + 1} due to non-float "
                            f"vector value: {line[:50]}..."
                        )
                        skipped_lines += 1

                    global_line_index += 1

                # 3c. If no lines were processed in this batch, we're done
                if lines_processed_in_this_batch == 0:
                    break

                # 3d. Upsert the points collected from this batch
                if points_to_upsert:
                    try:
                        client.upsert(
                            collection_name=collection_name,
                            points=points_to_upsert,
                            wait=True,
                        )
                        total_upserted_count += len(points_to_upsert)
                    except Exception as upsert_err:
                        logger.error(
                            f"Error during batch upsert near line {global_line_index}: "
                            f"{upsert_err}"
                        )

                # 3e. Update the progress bar (advance still works)
                progress.update(upsert_task, advance=lines_processed_in_this_batch)

        # 4. Log summary
        logger.info(
            f"Finished upserting {total_upserted_count} embeddings into "
            f"'{collection_name}'. Skipped {skipped_lines} lines."
        )

    except FileNotFoundError:
        # 5. Handle file not found error
        logger.error(f"Embeddings file not found at {embeddings_file}")
    except Exception as e:
        # 6. Handle other unexpected errors
        logger.error(f"An unexpected error occurred during upsert: {e}")


@app.command()
def main(
    dataset_name: str = typer.Option(
        settings.glove_dataset,
        "--dataset",
        "-d",
        help="GloVe dataset name",
        case_sensitive=False,
        show_choices=True,
    ),
) -> None:
    """Main execution function to download GloVe embeddings and upsert into Qdrant.

    Parses arguments, downloads/extracts embeddings, connects to Qdrant,
    sets up the collection, and upserts the embeddings.
    """
    # 1. Log the dataset being used (obtained from typer option)
    logger.info(f"Starting setup process for dataset: {dataset_name}")

    # 1a. Load allowed words from words.txt
    allowed_words_set: Optional[set[str]] = None
    if WORDS_FILE.exists():
        try:
            with WORDS_FILE.open("r", encoding="utf-8") as wf:
                allowed_words_set = {
                    line.strip().lower() for line in wf if line.strip()
                }
            logger.info(
                f"Successfully loaded {len(allowed_words_set)} words from {WORDS_FILE}"
            )
        except Exception as e:
            logger.error(
                f"Error reading {WORDS_FILE}: {e}. Proceeding without word filtering."
            )
            allowed_words_set = None
    else:
        logger.warning(f"{WORDS_FILE} not found. Proceeding without word filtering.")
        allowed_words_set = None

    # 2. Get dataset details (URL, filename, vector size)
    try:
        if (
            dataset_name
            not in Settings.model_fields["glove_dataset"].annotation.__args__
        ):
            raise ValueError(f"Invalid dataset choice: {dataset_name}")

        url, filename = get_glove_url_and_filename(dataset_name)
        dimension_key = dataset_name.split(".")[-1]
        vector_size = DIMENSIONS[dimension_key]
        logger.info(
            f"Dataset details: URL={url}, Filename={filename}, Vector Size={vector_size}"
        )
    except (ValueError, KeyError) as e:
        logger.error(f"Error determining dataset details for '{dataset_name}': {e}")
        raise typer.Exit(code=1)

    # 3. Download and extract embeddings file
    embeddings_file_path = download_and_extract_glove(url, filename, DATA_DIR)
    if not embeddings_file_path:
        logger.error("Failed to obtain embeddings file. Exiting.")
        raise typer.Exit(code=1)

    # 4. Connect to Qdrant
    logger.info(
        f"Connecting to Qdrant at {settings.qdrant_host}:"
        f"{settings.qdrant_grpc_host_port}..."
    )
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_host_port,
            grpc_port=settings.qdrant_grpc_host_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=True,
            https=False,
        )
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise typer.Exit(code=1)

    # 5. Setup Qdrant collection
    collection_name = dataset_name
    try:
        setup_qdrant_collection(client, collection_name, vector_size)
    except Exception as e:
        logger.error(
            f"Failed during collection setup for '{collection_name}'. Exiting. "
            f"Error: {e}"
        )
        raise typer.Exit(code=1)

    # 6. Upsert embeddings
    upsert_embeddings(client, collection_name, embeddings_file_path, allowed_words_set)

    # 7. Log completion
    logger.info("Qdrant setup script finished.")


if __name__ == "__main__":
    app()
