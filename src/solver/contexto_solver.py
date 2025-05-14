from typing import List, Optional, Set, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.models import CollectionInfo
from wordfreq import word_frequency

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import (
    create_qdrant_exclusion_filter,
    get_collection_info,
    get_multiple_random_points,
    get_random_point,
    get_vector_for_word,
)


class SolverUnableToGuessError(Exception):
    """Custom exception raised when the solver cannot determine a next guess."""

    pass


class ContextoSolver:
    """Solves a Contexto game by providing best guesses.

    Uses Qdrant's Discovery Search API by iteratively building context pairs
    and prioritizes suggestions based on word frequency to enhance relevance.

    Attributes:
        client: An initialized QdrantClient instance.
        collection_name: The name of the Qdrant collection being used.
        collection_info: Metadata about the Qdrant collection, such as point count.
    """

    TOP_N_POSITIVE_EMBEDDINGS: int = 3
    RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK: int = 256
    DISCOVERY_SEARCH_LIMIT: int = 4

    def __init__(self, client: QdrantClient, collection_name: str):
        """Initializes the ContextoSolver.

        Args:
            client: An initialized QdrantClient instance.
            collection_name: The name of the Qdrant collection.
                If an empty string is provided, it defaults to the value specified
                in `settings.effective_collection_name`.

        Raises:
            ValueError: If the Qdrant collection is empty or does not exist.
        """
        self.client: QdrantClient = client
        self.collection_name: str = (
            collection_name or settings.effective_collection_name
        )
        logger.info(
            f"Initializing ContextoSolver for collection '{self.collection_name}'"
            " with iterative discovery strategy."
        )

        collection_info: Optional[CollectionInfo] = get_collection_info(
            self.client, self.collection_name
        )
        if (
            not collection_info
            or not collection_info.points_count
            or collection_info.points_count == 0
        ):
            logger.error(
                f"Cannot initialize ContextoSolver: Collection "
                f"'{self.collection_name}' is empty or does not exist."
            )
            raise ValueError(
                f"Collection '{self.collection_name}' is empty or does not exist."
            )
        self.collection_info: CollectionInfo = collection_info

        self.__guessed_words_set: Set[str] = set()
        self.__not_allowed_words_set: Set[str] = set()
        self.__context_pairs_for_discovery: List[rest_models.ContextExamplePair] = []
        self.__ranked_positive_embeddings: List[Tuple[int, np.ndarray]] = []
        self.__current_positive_point_details: Optional[Tuple[int, np.ndarray]] = None

    def add_guess(self, word: str, rank: int) -> bool:
        """Adds a new guess, updates context, and manages positive/negative references.

        Args:
            word: The word being guessed.
            rank: The rank of the word (lower is better, 1 is the target).

        Returns:
            bool: True if the guess was successfully added and processed, False otherwise.
        """
        normalized_word: str = word.strip().lower()
        if not normalized_word:
            logger.warning("Attempted to add an empty string as a guess. Skipping.")
            return False

        if normalized_word in self.__guessed_words_set:
            logger.info(
                f"Word '{normalized_word}' has already been guessed. Skipping context update."
            )
            return False

        embedding: Optional[np.ndarray] = get_vector_for_word(
            self.client, self.collection_name, normalized_word
        )
        if embedding is None:
            logger.warning(
                f"Could not retrieve a valid embedding for word '{normalized_word}'. "
                "Guess not added, context not updated."
            )
            return False

        pair_positive_embedding: Optional[np.ndarray] = None
        pair_negative_embedding: Optional[np.ndarray] = None

        if not self.__guessed_words_set:
            pair_positive_embedding, pair_negative_embedding = (
                self._process_first_guess(normalized_word, rank, embedding)
            )
        else:
            pair_positive_embedding, pair_negative_embedding = (
                self._process_subsequent_guess(normalized_word, rank, embedding)
            )

        if pair_positive_embedding is not None and pair_negative_embedding is not None:
            self.__context_pairs_for_discovery.append(
                rest_models.ContextExamplePair(
                    positive=pair_positive_embedding.tolist(),
                    negative=pair_negative_embedding.tolist(),
                )
            )
            logger.info(
                "Added context pair. Total pairs: "
                f"{len(self.__context_pairs_for_discovery)}"
            )
        else:
            logger.error(
                "Failed to determine valid positive/negative embeddings for context pair. "
                "Pair not added. This may indicate an issue in guess processing logic."
            )

        self.__guessed_words_set.add(normalized_word)

        logger.info(
            f"Added guess: Word '{normalized_word}', Rank {rank}. "
            f"Total past guesses: {len(self.__guessed_words_set)}."
        )
        return True

    def mark_word_as_not_allowed(self, word: str) -> bool:
        """Marks a word as not allowed, preventing future suggestions.

        Adds the word to 'not allowed' and 'guessed' sets.

        Args:
            word: The word not allowed by the game.

        Returns:
            bool: True if newly marked as not allowed, False if already known or empty.
        """
        normalized_word: str = word.strip().lower()
        if not normalized_word:
            logger.warning(
                "Attempted to mark an empty string as not allowed. Skipping."
            )
            return False

        self.__guessed_words_set.add(normalized_word)

        if normalized_word in self.__not_allowed_words_set:
            logger.info(
                f"Word '{normalized_word}' was already marked as not allowed. No change in status."
            )
            return False

        logger.info(
            f"Marking word '{normalized_word}' as not allowed by the game. "
            "It will be excluded from future suggestions."
        )
        self.__not_allowed_words_set.add(normalized_word)
        return True

    def _process_first_guess(
        self,
        word: str,
        rank: int,
        embedding: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Processes the first guess, sets initial context pair.

        The first guess becomes the positive reference. Its negation is the negative.

        Args:
            word: The guessed word (normalized).
            rank: The rank of the guessed word.
            embedding: The embedding vector of the guessed word.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Positive and negative embeddings for the context pair.
        """
        self._update_positive_references(word, rank, embedding)
        return embedding, -embedding.copy()

    def _update_positive_references(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> None:
        """Updates best positive reference and top-ranked positive embeddings.

        Maintains up to `TOP_N_POSITIVE_EMBEDDINGS` best positive embeddings
        for centroid calculation. `__current_positive_point_details` stores the
        single best positive guess.

        Args:
            word: The word of the new positive reference.
            rank: The rank of the new positive reference.
            embedding: The embedding of the new positive reference.
        """
        self.__current_positive_point_details = (rank, embedding)

        self.__ranked_positive_embeddings.append((rank, embedding))

        self.__ranked_positive_embeddings.sort(key=lambda x: x[0])

        self.__ranked_positive_embeddings = self.__ranked_positive_embeddings[
            : self.TOP_N_POSITIVE_EMBEDDINGS
        ]

        logger.info(
            f"Updated current positive reference to: Word '{word}', Rank {rank}. "
            f"Total top-ranked positive embeddings for centroid: {len(self.__ranked_positive_embeddings)}."
        )

    def _process_subsequent_guess(
        self,
        word: str,
        rank: int,
        embedding: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Processes subsequent guess, determines context pair, updates positive references.

        If new guess is better than current best positive, new guess becomes positive
        for the pair, and previous best becomes negative. Otherwise, previous best
        remains positive, current guess becomes negative.

        Args:
            word: The guessed word (normalized).
            rank: The rank of the guessed word.
            embedding: The embedding vector of the guessed word.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Positive and negative embeddings for the context pair.
        """
        if self.__current_positive_point_details is None:
            logger.error(
                "Cannot process subsequent guess: `__current_positive_point_details` is None. "
                "This indicates a logical error in the solver's state management."
            )
            self._update_positive_references(word, rank, embedding)
            return embedding, -embedding.copy()

        prev_positive_rank, prev_positive_embedding = (
            self.__current_positive_point_details
        )

        pair_positive_embedding: np.ndarray
        pair_negative_embedding: np.ndarray

        if rank < prev_positive_rank:
            pair_positive_embedding = embedding
            pair_negative_embedding = prev_positive_embedding
            self._update_positive_references(word, rank, embedding)
        else:
            pair_positive_embedding = prev_positive_embedding
            pair_negative_embedding = embedding

        return pair_positive_embedding, pair_negative_embedding

    def guess_word(self) -> str:
        """Suggests the next best word using a multi-stage strategy.

        Strategies:
        1. Initial random word (frequency-prioritized) if no context.
        2. Discovery search with centroid of best positive guesses as target.
           Selects most frequent word from results.
        3. Fallback to new random word if discovery fails.

        Returns:
            str: The suggested word to guess next.

        Raises:
            SolverUnableToGuessError: If no suitable word can be determined.
        """
        if not self.__context_pairs_for_discovery:
            logger.info(
                "No context pairs yet. Fetching an initial random word from the collection."
            )
            initial_word: Optional[str] = self._fetch_initial_random_word(
                excluded_words=self.__guessed_words_set
            )
            if initial_word:
                return initial_word
            else:
                logger.error(
                    "Failed to fetch an initial random word. The collection might be too small or heavily filtered."
                )
                raise SolverUnableToGuessError(
                    "Unable to fetch an initial random word. Ensure the collection is populated and not overly filtered."
                )

        target_vector_list: Optional[List[float]] = None
        if self.__ranked_positive_embeddings:
            embeddings_for_centroid: List[np.ndarray] = [
                emb_data[1] for emb_data in self.__ranked_positive_embeddings
            ]
            if embeddings_for_centroid:
                logger.info(
                    f"Calculating centroid from {len(embeddings_for_centroid)} "
                    f"top-ranked positive embeddings (up to 3 lowest ranks)."
                )
                centroid_vector: np.ndarray = np.mean(
                    np.array(embeddings_for_centroid), axis=0
                )
                target_vector_list = centroid_vector.tolist()
                logger.info(
                    "Using centroid of top-ranked positive embeddings as target for discovery."
                )
            else:
                logger.warning(
                    "__ranked_positive_embeddings non-empty, but no embeddings extracted for centroid. "
                    "Target for discovery will be absent."
                )
        else:
            logger.warning(
                "No ranked positive embeddings found for centroid calculation (e.g., after first guess). "
                "Target for discovery will be absent. This might be normal if only one guess was bad."
            )

        candidate_word: Optional[str] = (
            self._execute_discovery_search_with_freq_selection(
                target_vector_list, self.__guessed_words_set
            )
        )

        if candidate_word:
            return candidate_word

        logger.info(
            "Discovery search did not yield a new suggestion. Falling back to random selection."
        )
        fallback_word: Optional[str] = self._fetch_initial_random_word(
            excluded_words=self.__guessed_words_set
        )
        if fallback_word:
            return fallback_word

        logger.error(
            "All guessing strategies exhausted. Unable to determine a next guess."
        )
        raise SolverUnableToGuessError(
            "All strategies failed to find a suitable next word. The collection might be too small, "
            "all words might have been guessed, or discovery parameters are too restrictive."
        )

    def _fetch_initial_random_word(
        self, excluded_words: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Fetches a random word, prioritizing by frequency.

        Samples `RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK` words, returns the most
        frequent one not in `excluded_words`. Falls back to a single random point.

        Args:
            excluded_words: Words to exclude. Defaults to an empty set.

        Returns:
            Optional[str]: A random word, or None if not found.
        """
        logger.info(
            "Attempting to fetch an initial random word, prioritizing by frequency."
        )
        current_excluded_words: Set[str] = excluded_words or set()

        if not self.collection_info or self.collection_info.points_count == 0:
            logger.error(
                "Cannot fetch random word: Collection info is missing or collection is empty."
            )
            return None

        num_samples_for_frequency_check = self.RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK
        random_records: List[rest_models.Record] = get_multiple_random_points(
            _client=self.client,
            collection_name=self.collection_name,
            excluded_words=current_excluded_words,
            count=num_samples_for_frequency_check,
        )

        most_frequent_word: Optional[str] = None
        highest_frequency: float = -1.0

        if random_records:
            for record in random_records:
                if record.payload and "word" in record.payload:
                    word_from_payload: Optional[str] = record.payload["word"]
                    if (
                        word_from_payload
                        and isinstance(word_from_payload, str)
                        and word_from_payload.strip()
                    ):
                        frequency: float = word_frequency(word_from_payload, "en")
                        if frequency > highest_frequency:
                            highest_frequency = frequency
                            most_frequent_word = word_from_payload
                    else:
                        logger.debug(
                            f"Skipping record with invalid or empty word in payload: {record.id}"
                        )
            if most_frequent_word:
                logger.info(
                    f"Selected most frequent word from {num_samples_for_frequency_check} samples: "
                    f"'{most_frequent_word}' (freq: {highest_frequency})."
                )
                return most_frequent_word
            else:
                logger.warning(
                    f"Could not determine a most frequent word from the {num_samples_for_frequency_check} samples. "
                    "This might happen if all sampled words have zero frequency, are invalid, or were excluded."
                )
        else:
            logger.warning(
                f"Failed to fetch a batch of {num_samples_for_frequency_check} random points for frequency check. "
                "Proceeding to fallback (single random point)."
            )

        logger.info("Falling back to fetching a single random point.")
        random_record_single: Optional[rest_models.Record] = get_random_point(
            _client=self.client,
            collection_name=self.collection_name,
            total_points=self.collection_info.points_count,
            excluded_words=current_excluded_words,
        )

        if (
            random_record_single
            and random_record_single.payload
            and "word" in random_record_single.payload
        ):
            word_single: Optional[str] = random_record_single.payload["word"]
            if word_single and isinstance(word_single, str) and word_single.strip():
                logger.info(
                    f"Fetched single random initial word (fallback): '{word_single}'"
                )
                return word_single
            else:
                logger.warning(
                    "Fallback single random point contained an invalid or empty word."
                )
        else:
            logger.warning(
                "Failed to fetch a valid single random word (fallback) or point was malformed."
            )

        return None

    def _execute_discovery_search_with_freq_selection(
        self,
        target_vector: Optional[List[float]],
        excluded_words: Set[str],
    ) -> Optional[str]:
        """Executes Qdrant Discovery search, selects result by highest word frequency.

        Uses up to `DISCOVERY_SEARCH_LIMIT` results.

        Args:
            target_vector: Target vector for discovery (e.g., centroid of positive
                           guesses). If None, relies solely on context pairs.
            excluded_words: Words to exclude from search results.

        Returns:
            Optional[str]: Highest-frequency word from discovery, or first valid
                           non-excluded word as fallback, or None if no suitable word.
        """
        if not self.__context_pairs_for_discovery:
            logger.warning("No context pairs available for discovery search. Skipping.")
            return None

        logger.info(
            f"Executing discovery search with {len(self.__context_pairs_for_discovery)} context pairs. "
            f"Target vector {'provided' if target_vector is not None else 'absent'}."
        )

        query_filter: Optional[rest_models.Filter] = create_qdrant_exclusion_filter(
            excluded_words
        )

        discovery_limit = self.DISCOVERY_SEARCH_LIMIT
        try:
            search_results: List[rest_models.ScoredPoint] = self.client.discover(
                collection_name=self.collection_name,
                target=target_vector,
                context=self.__context_pairs_for_discovery,
                limit=discovery_limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            if not search_results:
                logger.info("Discovery search returned no results.")
                return None

            most_frequent_word: Optional[str] = None
            highest_frequency: float = -1.0
            first_valid_word_fallback: Optional[str] = None

            for hit in search_results:
                if hit.payload and "word" in hit.payload:
                    word_from_payload: Optional[str] = hit.payload["word"]
                    if (
                        word_from_payload
                        and isinstance(word_from_payload, str)
                        and word_from_payload.strip()
                    ):
                        if not first_valid_word_fallback:  # Store the first valid one
                            first_valid_word_fallback = word_from_payload

                        frequency: float = word_frequency(word_from_payload, "en")
                        logger.debug(
                            f"Discovery candidate: '{word_from_payload}', Freq: {frequency:.4f}, Score: {hit.score:.4f}"
                        )
                        if frequency > highest_frequency:
                            highest_frequency = frequency
                            most_frequent_word = word_from_payload
                    else:
                        logger.debug(
                            f"Skipping discovery hit with invalid or empty word in payload: {hit.id}"
                        )

            if most_frequent_word:
                logger.info(
                    f"Selected most frequent word from discovery results (top {discovery_limit}): "
                    f"'{most_frequent_word}' (freq: {highest_frequency:.4f})."
                )
                return most_frequent_word

            if first_valid_word_fallback:
                logger.warning(
                    "Could not determine a most frequent word from discovery results. "
                    "Falling back to the first valid non-excluded word found."
                )
                logger.info(
                    f"Falling back to first valid word from discovery results: '{first_valid_word_fallback}'"
                )
                return first_valid_word_fallback

            logger.error(
                "No valid word found in discovery results to return, though results may have been present."
            )
            return None

        except Exception as e:
            logger.error(f"Error during discovery search execution: {e}", exc_info=True)
            return None
