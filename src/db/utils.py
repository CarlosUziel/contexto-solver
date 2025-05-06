# Database utility functions

import random
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Record

from config.logger import app_logger as logger
from config.settings import settings


def get_qdrant_client() -> Optional[QdrantClient]:
    """Initializes and returns a Qdrant client instance.

    Uses Streamlit's cache_resource to avoid reconnecting on every script run.
    Reads connection details from the application settings.

    Returns:
        Optional[QdrantClient]: An initialized QdrantClient instance, or None if
                                connection fails.
    """
    logger.info("Attempting to connect to Qdrant...")
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_host_port,
            grpc_port=settings.qdrant_grpc_host_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            https=False,
        )
        client.get_collections()
        logger.info(
            f"Successfully connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_http_host_port}."
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return None


def get_collection_info(
    _client: QdrantClient, collection_name: str
) -> Optional[models.CollectionInfo]:
    """Retrieves information about a specific Qdrant collection.

    Uses Streamlit's cache_data for caching the result.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection to query.

    Returns:
        Optional[models.CollectionInfo]: Information about the collection, or None if
                                         the collection doesn't exist or an error occurs.
    """
    logger.info(f"Fetching collection info for '{collection_name}'...")
    try:
        collection_info = _client.get_collection(collection_name=collection_name)
        logger.info(f"Successfully fetched info for collection '{collection_name}'.")
        return collection_info
    except Exception as e:
        logger.error(f"Failed to get collection info for '{collection_name}': {e}")
        return None


def get_random_point(
    _client: QdrantClient, collection_name: str, total_points: int
) -> Optional[Record]:
    """Fetches a single random point (word and vector) from the collection.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection.
        total_points (int): The total number of points in the collection.
                           Note: This is used for logging but the scroll below will fetch all.

    Returns:
        Optional[Record]: A Qdrant Record object containing the random point's data,
                          or None if an error occurs or the collection is empty.
    """
    if total_points <= 0:
        logger.warning(
            f"Reported total_points is {total_points} for collection '{collection_name}'. Will attempt to scroll anyway."
        )

    logger.info(
        f"Attempting to select a random point from '{collection_name}' (reported total: {total_points})..."
    )
    try:
        logger.info(
            f"Fetching all point IDs from '{collection_name}' to select one randomly..."
        )
        all_ids: List[models.PointId] = []
        current_scroll_offset: Optional[models.PointId] = None

        while True:
            page_records, next_page_scroll_offset = _client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=current_scroll_offset,
                with_payload=False,
                with_vectors=False,
            )
            if not page_records:
                break

            all_ids.extend([record.id for record in page_records])

            if next_page_scroll_offset is None:
                break
            current_scroll_offset = next_page_scroll_offset

        logger.info(
            f"Fetched {len(all_ids)} actual point IDs from '{collection_name}'."
        )

        if not all_ids:
            logger.error(
                f"No IDs found in collection '{collection_name}'. Cannot select a random point."
            )
            return None

        random_id: models.PointId = random.choice(all_ids)
        logger.info(f"Randomly selected Point ID: {random_id}")

        # Retrieve the full data for the selected point ID
        retrieved_points = _client.retrieve(
            collection_name=collection_name,
            ids=[random_id],
            with_payload=True,
            with_vectors=True,
        )

        if retrieved_points:
            target_point = retrieved_points[0]
            actual_word = "N/A"  # Default if payload is malformed
            if target_point.payload and "word" in target_point.payload:
                actual_word = target_point.payload["word"]
            logger.info(
                f"Successfully retrieved random point: ID {target_point.id}, Word: '{actual_word}'"
            )
            return target_point
        else:
            logger.error(f"Failed to retrieve point data for selected ID {random_id}.")
            return None

    except Exception as e:
        logger.error(
            f"Error selecting random point from '{collection_name}': {e}", exc_info=True
        )
        return None


def get_all_similarities(
    _client: QdrantClient,
    collection_name: str,
    target_vector: List[float],
    total_points: int,
) -> Optional[Dict[str, Tuple[int, List[float]]]]:
    """Computes similarities between the target vector and all points in the collection.

    Performs a kNN search where k is the total number of points to get a ranked list.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection.
        target_vector (List[float]): The vector of the target word.
        total_points (int): The total number of points in the collection.

    Returns:
        Optional[Dict[str, Tuple[int, List[float]]]]:
            A dictionary mapping each word (lowercase) to a tuple containing its
            similarity rank (1 being the most similar) and its vector.
            Returns None if an error occurs or the collection is empty.
    """
    if total_points <= 0:
        logger.warning(
            f"Cannot compute similarities in collection '{collection_name}' with {total_points} points."
        )
        return None

    logger.info(
        f"Computing similarities for all {total_points} points in '{collection_name}'..."
    )
    try:
        limit = max(1, total_points)
        search_result = _client.search(
            collection_name=collection_name,
            query_vector=target_vector,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        ranked_similarities_with_vectors = {}
        for rank, hit in enumerate(search_result):
            # Word is now fetched from payload, ID is a UUID
            if hit.payload and "word" in hit.payload and hit.vector:
                actual_word = hit.payload["word"]
                # Store word in lowercase for case-insensitive matching
                ranked_similarities_with_vectors[actual_word.lower()] = (
                    rank + 1,
                    hit.vector,
                )
            else:
                logger.warning(
                    f"Point ID {hit.id} missing 'word' in payload or vector, skipping."
                )

        if not ranked_similarities_with_vectors:
            logger.warning(
                f"Similarity search returned results, but no words with vectors found in payload for collection '{collection_name}'."
            )
            return None

        logger.info(
            f"Successfully computed and ranked similarities with vectors for {len(ranked_similarities_with_vectors)} words."
        )
        return ranked_similarities_with_vectors

    except Exception as e:
        logger.error(f"Error computing similarities in '{collection_name}': {e}")
        return None
