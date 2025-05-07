from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models  # Added import

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import (
    find_closest_word_to_point,
    get_collection_info,
    get_random_point,
    get_vector_for_word,
)


class ContextoSolver:
    """
    Solves a ContextoGame by providing best guesses.
    Uses Qdrant's Discovery Search API by iteratively building context pairs.
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Initializes the ContextoSolver.

        Args:
            client (QdrantClient): An initialized QdrantClient instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.client = client
        self.collection_name = collection_name
        logger.info(
            f"Initializing ContextoSolver for collection '{collection_name}' with iterative discovery strategy."
        )
        # Store word, rank and embedding data of past guesses, sorted by rank (lower is better)
        self.past_guesses: List[Tuple[str, int, np.ndarray]] = []

        # Stores accumulated context pairs for Qdrant's discover API
        self.context_pairs_for_discovery: List[rest_models.ContextExamplePair] = []

        # Stores embeddings of guesses that were considered "positive" for centroid calculation
        self.positive_embeddings_for_centroid: List[np.ndarray] = []

        # Details of the point currently considered the best positive reference for context construction
        self.current_positive_point_details: Optional[Tuple[str, int, np.ndarray]] = (
            None
        )

        # Embedding used as the negative reference in the latest context pair construction
        self.current_negative_reference_embedding: Optional[np.ndarray] = None

    def _get_random_distant_embedding(
        self, excluded_words: List[str]
    ) -> Optional[np.ndarray]:
        """Attempts to get an embedding of a random word not in excluded_words."""
        for _ in range(5):  # Try a few times
            try:
                random_word_str = (
                    self._solve_no_past_guesses()
                )  # Leverages existing random word fetch
                if random_word_str not in excluded_words:
                    embedding = get_vector_for_word(
                        self.client, self.collection_name, random_word_str
                    )
                    if (
                        embedding is not None
                        and embedding.ndim == 1
                        and embedding.shape[0] > 0
                    ):
                        return embedding
            except (
                ValueError
            ):  # Raised by _solve_no_past_guesses if collection is empty/problematic
                logger.warning("Could not fetch a random word for distant embedding.")
                return None
            except Exception as e:
                logger.warning(f"Error fetching embedding for random distant word: {e}")
                return None
        logger.warning(
            "Failed to find a suitable random distant embedding after multiple attempts."
        )
        return None

    def add_guess(self, word: str, rank: int) -> bool:
        """
        Adds a new guess, updates context pairs, and manages positive/negative references.
        """
        if any(past_word == word for past_word, _, _ in self.past_guesses):
            logger.info(
                f"Word '{word}' has already been guessed. Skipping context update."
            )
            return False

        embedding = get_vector_for_word(self.client, self.collection_name, word)
        if embedding is None or not (
            isinstance(embedding, np.ndarray)
            and embedding.ndim == 1
            and embedding.shape[0] > 0
        ):
            logger.warning(
                f"Could not retrieve a valid embedding for word '{word}'. Guess not added, context not updated."
            )
            return False

        new_guess_details = (word, rank, embedding)

        pair_positive_embedding: Optional[np.ndarray] = None
        pair_negative_embedding: Optional[np.ndarray] = None

        if not self.past_guesses:  # This is the very first guess being added
            self.current_positive_point_details = new_guess_details
            self.positive_embeddings_for_centroid.append(embedding)

            distant_negative_embedding = self._get_random_distant_embedding(
                excluded_words=[word]
            )
            if distant_negative_embedding is None:
                logger.warning(
                    "Using negation of the first guess as fallback for initial negative context."
                )
                distant_negative_embedding = -embedding.copy()

            pair_positive_embedding = embedding
            pair_negative_embedding = distant_negative_embedding
            self.current_negative_reference_embedding = distant_negative_embedding
        else:
            # self.current_positive_point_details should be set from previous add_guess call
            if (
                self.current_positive_point_details is None
            ):  # Should not happen if logic is correct
                logger.error(
                    "Critical: current_positive_point_details is None after first guess. Resetting."
                )
                # Fallback: treat current new guess as the positive, and its negation as negative
                self.current_positive_point_details = new_guess_details
                self.positive_embeddings_for_centroid.append(embedding)
                pair_positive_embedding = embedding
                pair_negative_embedding = -embedding.copy()
                self.current_negative_reference_embedding = pair_negative_embedding
            else:
                prev_positive_word, prev_positive_rank, prev_positive_embedding = (
                    self.current_positive_point_details
                )

                if rank < prev_positive_rank:
                    pair_positive_embedding = embedding
                    pair_negative_embedding = prev_positive_embedding
                    self.current_positive_point_details = new_guess_details
                    # Check for duplicates using np.array_equal
                    is_duplicate = any(
                        np.array_equal(embedding, e)
                        for e in self.positive_embeddings_for_centroid
                    )
                    if not is_duplicate:
                        self.positive_embeddings_for_centroid.append(embedding)
                    self.current_negative_reference_embedding = prev_positive_embedding
                else:  # New guess is not better than current positive
                    pair_positive_embedding = prev_positive_embedding
                    pair_negative_embedding = embedding
                    # self.current_positive_point_details remains unchanged
                    self.current_negative_reference_embedding = embedding

        if pair_positive_embedding is not None and pair_negative_embedding is not None:
            self.context_pairs_for_discovery.append(
                rest_models.ContextExamplePair(
                    positive=pair_positive_embedding.tolist(),
                    negative=pair_negative_embedding.tolist(),
                )
            )
            logger.info(
                f"Added context pair. Total pairs: {len(self.context_pairs_for_discovery)}"
            )
        else:
            logger.error(
                "Failed to determine positive/negative for context pair. Pair not added."
            )

        self.past_guesses.append(new_guess_details)
        self.past_guesses.sort(key=lambda x: x[1])  # Keep sorted by rank

        logger.info(
            f"Added guess: Word '{word}', Rank {rank}. Total past guesses: {len(self.past_guesses)}."
        )
        return True

    def guess_word(self) -> Optional[str]:
        """
        Get the next best guess using Qdrant's Discovery Search with accumulated context.
        """
        num_past_guesses = len(self.past_guesses)

        if num_past_guesses == 0:
            try:
                logger.info("No past guesses. Making an initial random guess.")
                return self._solve_no_past_guesses()
            except ValueError as e:
                logger.error(f"Error making initial random guess: {e}")
                return None

        target_vector_list: Optional[List[float]] = None
        limit = 1

        if not self.context_pairs_for_discovery:
            logger.warning(
                "No context pairs available for discovery. Falling back to best guess step."
            )
            # This state implies add_guess might not have formed pairs, or it's just after the first guess
            # and guess_word is called before a context pair could be formed based on it.
            # For safety, use the old fallback if no context pairs.
            return self._fallback_to_best_guess_step_or_random()

        if num_past_guesses == 1:  # After 1st guess, pure context search
            logger.info("First discovery search: No target, using accumulated context.")
            limit = (
                10  # Sample more to find a good bisector, though we pick top for now
            )
        else:  # Subsequent searches, use centroid of positives as target
            if self.positive_embeddings_for_centroid:
                centroid_vector = np.mean(
                    np.array(self.positive_embeddings_for_centroid), axis=0
                )
                if centroid_vector is not None and centroid_vector.ndim == 1:
                    target_vector_list = centroid_vector.tolist()
                    logger.info(
                        f"Using centroid of {len(self.positive_embeddings_for_centroid)} positive embeddings as target."
                    )
                else:
                    logger.warning(
                        "Failed to compute valid centroid. Using best guess embedding as target fallback."
                    )
                    target_vector_list = self.past_guesses[0][
                        2
                    ].tolist()  # Fallback to best guess
            else:  # Should not happen if positive_embeddings_for_centroid is managed correctly
                logger.warning(
                    "No positive embeddings for centroid. Using best guess embedding as target."
                )
                target_vector_list = self.past_guesses[0][
                    2
                ].tolist()  # Fallback to best guess
            limit = 1  # More focused search

        excluded_words = {word for word, _, _ in self.past_guesses}
        query_filter = None
        if excluded_words:
            query_filter = rest_models.Filter(
                must_not=[
                    rest_models.FieldCondition(
                        key="word", match=rest_models.MatchAny(any=list(excluded_words))
                    )
                ]
            )

        try:
            logger.info(
                f"Performing discovery search: {len(self.context_pairs_for_discovery)} context pairs, target {'present' if target_vector_list else 'absent'}, excluding {len(excluded_words)} words."
            )

            search_results = self.client.discover(
                collection_name=self.collection_name,
                target=target_vector_list,  # Can be None
                context=self.context_pairs_for_discovery,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                search_params=rest_models.SearchParams(hnsw_ef=128),
            )

            if search_results:
                # For now, just pick the top result. Future: if limit > 1, pick median.
                for hit in search_results:
                    if hit.payload and "word" in hit.payload:
                        candidate_word = hit.payload["word"]
                        if (
                            candidate_word not in excluded_words
                        ):  # Double check, filter should handle
                            logger.info(
                                f"Discovery search suggested: '{candidate_word}' (score: {hit.score})"
                            )
                            return candidate_word
                logger.warning(
                    "Discovery search returned candidates, but none were new or suitable after processing."
                )
            else:
                logger.warning("Discovery search returned no results.")

        except Exception as e:
            logger.error(f"Qdrant discovery search error: {e}", exc_info=True)

        logger.warning("Discovery failed or no new word. Falling back.")
        return self._fallback_to_best_guess_step_or_random()

    def _fallback_to_best_guess_step_or_random(self) -> Optional[str]:
        """Fallback: random step from best guess, or fully random if that fails."""
        if (
            not self.past_guesses
        ):  # Should not be called if no past guesses, but as safety
            return self._fallback_random_guess()

        best_guess_word, _, best_guess_embedding = self.past_guesses[0]
        logger.warning(
            f"Falling back to random step from best guess: '{best_guess_word}'."
        )

        if not (
            isinstance(best_guess_embedding, np.ndarray)
            and best_guess_embedding.ndim == 1
            and best_guess_embedding.shape[0] > 0
        ):
            logger.error(
                "Fallback: Best guess embedding is invalid for random step. Trying full random."
            )
            return self._fallback_random_guess()

        embedding_dim = best_guess_embedding.shape[0]
        random_direction = np.random.randn(embedding_dim)
        norm_random_direction = np.linalg.norm(random_direction)

        if norm_random_direction > 1e-9:
            random_direction /= norm_random_direction
        else:
            logger.warning(
                "Fallback: Random direction norm is zero. Trying full random."
            )
            return self._fallback_random_guess()

        step_scale = settings.base_step_scale
        new_point_vector = best_guess_embedding + step_scale * random_direction

        excluded_words = {word for word, _, _ in self.past_guesses}
        return find_closest_word_to_point(
            self.client,
            self.collection_name,
            new_point_vector.tolist(),
            excluded_words,
        )

    def _solve_no_past_guesses(self) -> str:
        """
        Handles the case where there are no past guesses.
        Makes a random guess from the collection.

        Returns:
            str: A random word from the collection.
        Raises:
            ValueError: If collection is empty or random point cannot be fetched.
        """
        logger.info("No past guesses. Making a random guess.")
        collection_info = get_collection_info(self.client, self.collection_name)
        if not collection_info or not collection_info.points_count:
            logger.error(
                f"Cannot get collection info or collection '{self.collection_name}' is empty."
            )
            raise ValueError(
                f"Collection '{self.collection_name}' is empty or does not exist."
            )

        random_point_record = get_random_point(
            self.client, self.collection_name, collection_info.points_count
        )
        if random_point_record and random_point_record.payload:
            word = random_point_record.payload.get("word")
            if word:
                return word
        logger.error("Failed to get a random point or word from payload.")
        raise ValueError(
            f"Failed to get a random point/word from collection '{self.collection_name}'."
        )

    def _fallback_random_guess(self) -> Optional[str]:
        """Ultimate fallback: try to get any random word not yet guessed."""
        logger.info("Executing ultimate fallback: trying a random guess.")
        try:
            # Try a few times to get a new random word. This is not guaranteed.
            # The game layer should ultimately prevent re-guessing the same word.
            for _ in range(5):
                random_word = self._solve_no_past_guesses()
                if random_word not in {word for word, _, _ in self.past_guesses}:
                    return random_word
            logger.warning(
                "Fallback: Failed to find a new random word after multiple attempts. Returning first random found."
            )
            return self._solve_no_past_guesses()  # Return a random one anyway
        except ValueError as e:
            logger.error(
                f"Error in _fallback_random_guess -> _solve_no_past_guesses: {e}"
            )
            return None
