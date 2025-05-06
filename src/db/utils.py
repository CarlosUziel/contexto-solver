# Database utility functions

import random
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
    _client: QdrantClient, collection_name: str, total_points: int
) -> Optional[Record]:
    """Fetches a single random point (word and vector) from the collection.

    Args:
        _client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the collection.
        total_points (int): The total number of points in the collection.
                           Assured by caller to be > 0.

    Returns:
        Optional[Record]: A Qdrant Record object containing the random point's data,
                          or None if an error occurs (e.g., collection becomes empty unexpectedly).
    """
    logger.info(
        f"Attempting to select a random point from '{collection_name}' (reported total: "
        f"{total_points})..."
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
            actual_word = target_point.payload["word"]
            logger.info(
                f"Successfully retrieved random point: ID {target_point.id}, Word: "
                f"'{actual_word}'"
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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes the cosine similarity between two numpy vectors."""
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if vec1.shape != vec2.shape:
        raise ValueError("Input vectors must have the same shape.")

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        # Cosine similarity is undefined if one or both vectors are zero vectors.
        # Depending on the context, you might return 0, -1, or raise an error.
        # Returning 0 for now, assuming non-negative similarities are expected elsewhere.
        return 0.0

    similarity = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
    return float(similarity)


def get_all_words_with_vectors(
    client: QdrantClient, collection_name: str
) -> Dict[str, np.ndarray]:
    """
    Retrieves all words and their corresponding vectors from the Qdrant collection.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the Qdrant collection.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping each word (lowercase) to its vector.
                               Returns an empty dictionary if an error occurs or no words are found.
    """
    all_words_vectors: Dict[str, np.ndarray] = {}
    logger.info(
        f"Attempting to retrieve all words and vectors from collection '{collection_name}'."
    )
    try:
        next_page_offset = None
        processed_count = 0
        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=250,  # Adjust batch size as needed
                offset=next_page_offset,
                with_payload=True,  # Need payload for "word"
                with_vectors=True,
            )
            if not points:
                break  # No more points to fetch

            for point in points:
                if (
                    point.payload
                    and "word" in point.payload
                    and isinstance(point.payload["word"], str)
                    and point.vector is not None
                ):
                    word = point.payload["word"].lower()
                    all_words_vectors[word] = np.array(point.vector, dtype=np.float32)
                    processed_count += 1
                else:
                    logger.warning(
                        f"Skipping point with id {point.id} due to missing 'word' in payload or missing vector."
                    )

            logger.debug(
                f"Scrolled {len(points)} points. Total processed so far: {processed_count}"
            )

            if next_page_offset is None:
                break  # All points have been fetched

        logger.info(
            f"Successfully retrieved {len(all_words_vectors)} words and vectors from '{collection_name}'."
        )
        return all_words_vectors
    except Exception as e:
        logger.error(
            f"Error retrieving all words and vectors from '{collection_name}': {e}",
            exc_info=True,
        )
        return {}


def compute_qdrant_similarity_matrix(
    client: QdrantClient,
    collection_name: str,
    words_list: List[str],
    word_to_idx_map: Dict[str, int],
    all_word_vectors: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Computes the full N x N similarity matrix using Qdrant search for each word.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the Qdrant collection.
        words_list (List[str]): Ordered list of words. The matrix dimensions will be based on this list.
        word_to_idx_map (Dict[str, int]): Mapping from word string to its index in words_list.
        all_word_vectors (Dict[str, np.ndarray]): Dictionary mapping words to their vector representations.

    Returns:
        np.ndarray: An N x N NumPy array where N is the number of words in words_list.
                    matrix[i, j] is the similarity between words_list[i] and words_list[j].
                    Returns an empty array if words_list is empty.
    """
    N = len(words_list)
    if N == 0:
        logger.warning("compute_qdrant_similarity_matrix called with empty words_list.")
        return np.array([], dtype=np.float32)

    logger.info(
        f"Computing {N}x{N} similarity matrix for collection '{collection_name}' via Qdrant..."
    )
    similarity_matrix = np.zeros((N, N), dtype=np.float32)

    for i, query_word in enumerate(words_list):
        if query_word not in all_word_vectors:
            logger.warning(
                f"Word '{query_word}' (index {i}) not found in all_word_vectors. Skipping its row in similarity matrix."
            )
            continue  # Or fill with a default value like np.nan or -1, or raise error

        query_vector = all_word_vectors[query_word]

        try:
            # Search for the most similar N points to the query_vector
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=N,  # Get all other points for similarity
                with_payload=True,  # Need payload for "word"
                with_vectors=False,  # No need to return vectors themselves
            )

            for hit in hits:
                if hit.payload and "word" in hit.payload:
                    hit_word = hit.payload["word"].lower()
                    if hit_word in word_to_idx_map:
                        j = word_to_idx_map[hit_word]
                        similarity_matrix[i, j] = hit.score
                    else:
                        logger.debug(
                            f"Word '{hit_word}' from Qdrant search result not in provided word_to_idx_map. Skipping."
                        )
                else:
                    logger.debug(
                        f"Skipping hit for query_word '{query_word}' due to missing 'word' in payload: {hit.id}"
                    )

            # Ensure self-similarity is 1.0, Qdrant cosine similarity should be 1 for identical vectors.
            # If the collection uses a different metric, this might need adjustment.
            if (
                similarity_matrix[i, i] == 0 and query_word in word_to_idx_map
            ):  # if it wasn't set by a hit (e.g. if limit < N or word missing)
                similarity_matrix[i, i] = 1.0

        except Exception as e:
            logger.error(
                f"Error during Qdrant search for word '{query_word}' (index {i}): {e}",
                exc_info=True,
            )
            # Optionally, fill the row with a specific value like np.nan or continue
            # For now, it will remain zeros if an error occurs for a word.

    logger.info(f"Successfully computed {N}x{N} similarity matrix via Qdrant.")
    return similarity_matrix


def get_ranks_for_word(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    candidate_words: List[str],
    batch_size: int = 32,  # Batch size for Qdrant search_batch
) -> Dict[str, int]:
    """
    For a given query_vector, get its similarity rank compared to a list of candidate_words.
    Rank is 0-indexed (0 is most similar/identical).
    This function queries Qdrant using search_batch for efficiency.

    Args:
        client (QdrantClient): An initialized QdrantClient instance.
        collection_name (str): The name of the Qdrant collection.
        query_vector (np.ndarray): The vector for which to find similarities.
        candidate_words (List[str]): A list of words to rank against the query_vector.
        batch_size (int): Number of search requests to batch together for Qdrant.

    Returns:
        Dict[str, int]: A dictionary mapping each candidate word (that was found and ranked)
                        to its 0-indexed rank relative to the query_vector.
                        Lower rank means more similar. Words not found or errors might be excluded.
    """
    if not candidate_words:
        return {}

    logger.debug(
        f"get_ranks_for_word: Ranking {len(candidate_words)} candidates against query vector "
        f"in collection '{collection_name}'."
    )
    word_to_rank: Dict[str, int] = {}

    # Qdrant's search returns scored points. We need to convert these scores to ranks.
    # The most straightforward way is to search for the query_vector against ALL words
    # in the candidate_words list and then determine the rank from the order of results.

    # We will perform a single search request with the query_vector,
    # but we need to ensure Qdrant only considers points from our candidate_words list.
    # This can be done using a filter for the 'word' payload.

    if not candidate_words:
        return {}

    # Create a filter to only search within the candidate words
    # Qdrant expects a list of values for MatchAny
    scroll_filter = rest_models.Filter(
        must=[
            rest_models.FieldCondition(
                key="word",
                match=rest_models.MatchAny(any=candidate_words),
            )
        ]
    )

    try:
        # Search for the query_vector, filtered by candidate_words, limit to number of candidates
        # as we want all of them ranked.
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            query_filter=scroll_filter,
            limit=len(candidate_words),  # Get all candidate words ranked
            with_payload=True,  # Need payload for "word"
            with_vectors=False,  # No need for vectors
        )

        # The 'hits' are already sorted by similarity by Qdrant (highest score first)
        # So, the order in 'hits' gives us the rank.
        for rank, hit in enumerate(hits):
            if hit.payload and "word" in hit.payload:
                hit_word = hit.payload["word"].lower()
                # We only care about words that were in our original candidate_words list
                if hit_word in candidate_words:
                    word_to_rank[hit_word] = rank  # 0-indexed rank
            else:
                logger.debug(
                    f"get_ranks_for_word: Skipping hit with ID {hit.id} due to missing 'word' in payload."
                )

        logger.debug(
            f"get_ranks_for_word: Successfully ranked {len(word_to_rank)} out of {len(candidate_words)} candidates."
        )

    except Exception as e:
        logger.error(
            f"get_ranks_for_word: Error during Qdrant search for ranking: {e}",
            exc_info=True,
        )
        # Return whatever was processed so far, or an empty dict if total failure
        return word_to_rank

    return word_to_rank


def find_closest_word_to_point(
    client: QdrantClient,
    collection_name: str,
    point_vector: List[float],
    exclude_words: Set[str],
) -> Optional[str]:
    """
    Finds the closest word embedding in the Qdrant collection to a given point,
    excluding specified words.
    """
    q_filter = None
    if exclude_words:
        q_filter = rest_models.Filter(
            must_not=[
                rest_models.FieldCondition(
                    key="word", match=rest_models.MatchAny(any=list(exclude_words))
                )
            ]
        )

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
