from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest_models
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
            f"Successfully connected to Qdrant at {settings.qdrant_host}:"
            f"{settings.qdrant_http_host_port}."
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
    _client: QdrantClient,
    collection_name: str,
    total_points: int,
    excluded_words: Optional[Set[str]] = None,
) -> Optional[Record]:
    """Fetches a single random point (word and vector) from the collection using random sampling.
    Can optionally exclude a set of words.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection.
        total_points (int): The total number of points in the collection (used for logging).
        excluded_words (Optional[Set[str]], optional): A set of words to exclude. Defaults to None.

    Returns:
        Optional[Record]: A Qdrant Record object containing the random point's data,
                          or None if an error occurs or no suitable point is returned.
    """
    query_filter: Optional[rest_models.Filter] = None
    if excluded_words and len(excluded_words) > 0:
        logger.info(f"Applying exclusion filter for {len(excluded_words)} words.")
        query_filter = rest_models.Filter(
            must_not=[
                rest_models.FieldCondition(
                    key="word", match=rest_models.MatchAny(any=list(excluded_words))
                )
            ]
        )

    logger.info(
        f"Attempting to select a random point from '{collection_name}' using random "
        f"sampling (reported total: {total_points}), filter "
        f"{'applied' if query_filter else 'not applied'}."
    )
    try:
        query_responses = _client.query_points(
            collection_name=collection_name,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            query_filter=query_filter,  # Apply the filter here
            limit=1,  # We need only one random point
            with_payload=True,
            with_vectors=True,
        )

        if query_responses and query_responses.points:
            random_scored_point = query_responses.points[0]

            # Ensure payload and word exist, similar to previous checks
            if random_scored_point.payload and "word" in random_scored_point.payload:
                actual_word = random_scored_point.payload["word"]
                logger.info(
                    f"Successfully retrieved random point using sampling: ID "
                    f"{random_scored_point.id}, Word: '{actual_word}'"
                )
                return random_scored_point
            else:
                logger.error(
                    f"Random sampling returned a point (ID: {random_scored_point.id}) "
                    f"but it's missing payload or 'word'."
                )
                return None
        else:
            logger.error(
                f"Random sampling did not return any points from collection "
                f"'{collection_name}'."
            )
            return None

    except Exception as e:
        logger.error(
            f"Error selecting random point using sampling from '{collection_name}': {e}",
            exc_info=True,
        )
        return None


def create_qdrant_exclusion_filter(
    excluded_words: Optional[Set[str]],
) -> Optional[rest_models.Filter]:
    """
    Creates a Qdrant filter object to exclude a given set of words.

    Args:
        excluded_words (Optional[Set[str]]): A set of words to exclude.
                                             If None or empty, returns None.

    Returns:
        Optional[rest_models.Filter]: A Qdrant Filter object or None.
    """
    if not excluded_words:
        return None
    logger.info(f"Creating exclusion filter for {len(excluded_words)} words.")
    return rest_models.Filter(
        must_not=[
            rest_models.FieldCondition(
                key="word", match=rest_models.MatchAny(any=list(excluded_words))
            )
        ]
    )


def get_all_similarities(
    _client: QdrantClient,
    collection_name: str,
    target_vector: np.ndarray,
    total_points: int,
) -> Optional[Dict[str, Tuple[int, np.ndarray]]]:
    """Computes similarities between the target vector and all points in the collection.

    Performs a kNN search where k is the total number of points to get a ranked list.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection.
        target_vector (np.ndarray): The vector of the target word.
        total_points (int): The total number of points in the collection.

    Returns:
        Optional[Dict[str, Tuple[int, np.ndarray]]]:
            A dictionary mapping each word (lowercase) to a tuple containing its
            similarity rank (0 being the most similar) and its vector.
            Returns None if an error occurs.
    """
    logger.info(
        f"Computing similarities for all {total_points} points in '{collection_name}'..."
    )
    try:
        limit = max(1, total_points)
        search_result = _client.search(
            collection_name=collection_name,
            query_vector=target_vector.tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        ranked_similarities_with_vectors = {}
        for rank, hit in enumerate(search_result):
            # Assumes payload, payload["word"], and hit.vector exist
            actual_word = hit.payload["word"]
            ranked_similarities_with_vectors[actual_word.lower()] = (
                rank,  # Store 0-indexed rank
                np.array(hit.vector, dtype=np.float32),
            )

        if not ranked_similarities_with_vectors:
            # This case might still be possible if search_result is empty for some reason,
            # even if total_points > 0. For example, if the target_vector is malformed.
            logger.warning(
                f"Similarity search returned no results or no valid words in payload "
                f"for collection '{collection_name}'."
            )
            return None

        logger.info(
            f"Successfully computed and ranked similarities with vectors for "
            f"{len(ranked_similarities_with_vectors)} words."
        )
        return ranked_similarities_with_vectors

    except Exception as e:
        logger.error(f"Error computing similarities in '{collection_name}': {e}")
        return None


def get_vector_for_word(
    client: QdrantClient, collection_name: str, word: str
) -> Optional[np.ndarray]:
    """
    Retrieves the vector for a given word from the Qdrant collection.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the Qdrant collection.
        word (str): The word (lowercase) to search for.

    Returns:
        Optional[np.ndarray]: The vector as a NumPy array if found, else None.
    """
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="word", match=rest_models.MatchValue(value=word)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        if points and points[0] and points[0].vector:
            return np.array(points[0].vector, dtype=np.float32)
        logger.warning(
            f"get_vector_for_word: Word '{word}' not found or has no vector in "
            f"collection '{collection_name}'."
        )
    except Exception as e:
        logger.error(
            f"get_vector_for_word: Error fetching vector for '{word}': {e}",
            exc_info=True,
        )
    return None


def find_closest_word_to_point(
    client: QdrantClient,
    collection_name: str,
    point_vector: List[float],
    exclude_words: Set[str],
) -> Optional[str]:
    """Finds the closest word embedding in the Qdrant collection to a given point.

    This function searches the specified Qdrant collection for the word whose
    vector representation is most similar (closest) to the provided `point_vector`,
    while ensuring that any words listed in `exclude_words` are not considered.

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the Qdrant collection to search within.
        point_vector: A list of floats representing the vector of the point
            for which the closest word is to be found.
        exclude_words: A set of strings, where each string is a word to be
            excluded from the search results.

    Returns:
        Optional[str]: The closest word (as a string) found in the collection that
            is not in `exclude_words`. Returns None if no suitable word is found
            or if an error occurs during the search.
    """
    q_filter = create_qdrant_exclusion_filter(exclude_words)

    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=point_vector,
            query_filter=q_filter,
            limit=32,
        )

        for hit in hits:  # Iterate through hits for safety
            if hit.payload and "word" in hit.payload:
                found_word = hit.payload["word"].lower()  # Ensure lowercase
                if found_word not in exclude_words:  # Double-check exclusion
                    logger.debug(
                        f"find_closest_word_to_point: Closest word: '{found_word}' "
                        f"(Score: {hit.score})"
                    )
                    return found_word
        logger.warning(
            "find_closest_word_to_point: No suitable word found after checking Qdrant "
            "hits and exclusions."
        )

    except Exception as e:
        logger.error(
            f"find_closest_word_to_point: Error during Qdrant search: {e}",
            exc_info=True,
        )

    return None
