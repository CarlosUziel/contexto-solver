from typing import List, Optional, Set, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import (
    find_closest_word_to_point,
    get_collection_info,
    get_random_point,
    get_vector_for_word,
)


class SolverUnableToGuessError(Exception):
    """Custom exception raised when the solver cannot determine a next guess."""

    pass


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

        collection_info = get_collection_info(self.client, self.collection_name)
        if (
            not collection_info
            or not collection_info.points_count
            or collection_info.points_count == 0
        ):
            logger.error(
                f"Cannot initialize ContextoSolver: Collection '{self.collection_name}' is empty or does not exist."
            )
            raise ValueError(
                f"Collection '{self.collection_name}' is empty or does not exist."
            )
        self.collection_info = collection_info

        self.__past_guesses: List[Tuple[str, int, np.ndarray]] = []

        self.__context_pairs_for_discovery: List[rest_models.ContextExamplePair] = []

        self.__positive_embeddings_for_centroid: List[np.ndarray] = []

        self.__current_positive_point_details: Optional[Tuple[str, int, np.ndarray]] = (
            None
        )

        self.__current_negative_reference_embedding: Optional[np.ndarray] = None

    def add_guess(self, word: str, rank: int) -> bool:
        """
        Adds a new guess, updates context pairs, and manages positive/negative references.

        Args:
            word: The word being guessed
            rank: The rank of the word (lower is better)

        Returns:
            bool: True if the guess was successfully added, False otherwise
        """
        if any(past_word == word for past_word, _, _ in self.__past_guesses):
            logger.info(
                f"Word '{word}' has already been guessed. Skipping context update."
            )
            return False

        embedding = get_vector_for_word(self.client, self.collection_name, word)
        if embedding is None:  # Check if embedding is None directly
            logger.warning(
                f"Could not retrieve a valid embedding for word '{word}'. Guess not added, context not updated."
            )
            return False

        pair_positive_embedding: Optional[np.ndarray] = None
        pair_negative_embedding: Optional[np.ndarray] = None

        if not self.__past_guesses:
            pair_positive_embedding, pair_negative_embedding = (
                self._process_first_guess(word, rank, embedding)
            )
        else:
            pair_positive_embedding, pair_negative_embedding = (
                self._process_subsequent_guess(word, rank, embedding)
            )

        if pair_positive_embedding is not None and pair_negative_embedding is not None:
            self.__context_pairs_for_discovery.append(
                rest_models.ContextExamplePair(
                    positive=pair_positive_embedding.tolist(),
                    negative=pair_negative_embedding.tolist(),
                )
            )
            logger.info(
                f"Added context pair. Total pairs: {len(self.__context_pairs_for_discovery)}"
            )
        else:
            logger.error(
                "Failed to determine valid positive/negative embeddings for context pair. Pair not added."
            )

        self.__past_guesses.append((word, rank, embedding))
        self.__past_guesses.sort(key=lambda x: x[1])

        logger.info(
            f"Added guess: Word '{word}', Rank {rank}. Total past guesses: {len(self.__past_guesses)}."
        )
        return True

    def _fetch_random_word_from_collection(self) -> str:
        """
        Fetches a random word from the Qdrant collection.
        Assumes __init__ has validated the collection's initial state.
        Handles potential issues if collection state changes post-init.

        Returns:
            str: A random word from the collection.
        Raises:
            ValueError: If a random point cannot be fetched or a valid word
                        cannot be extracted from its payload, or if the collection
                        becomes inaccessible/empty post-init.
            AttributeError: If collection_info becomes None unexpectedly (e.g.,
                            collection deleted post-init), when trying to access points_count.
        """
        logger.debug(
            f"Attempting to fetch a random word from collection '{self.collection_name}'."
        )

        random_point_record = get_random_point(
            self.client, self.collection_name, self.collection_info.points_count
        )

        if random_point_record and random_point_record.payload:
            word = random_point_record.payload.get("word")
            if word and isinstance(word, str) and word.strip():
                logger.debug(f"Successfully fetched random word: '{word}'.")
                return word

        logger.error(
            f"Failed to get a random point or extract a valid word from its payload from collection '{self.collection_name}'."
        )
        raise ValueError(
            f"Failed to get a random point/word from collection '{self.collection_name}'."
        )

    def _process_first_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the first guess, updating solver state and determining initial context pair embeddings.

        Side effects:
            - Sets `self.__current_positive_point_details`.
            - Appends to `self.__positive_embeddings_for_centroid`.
            - Sets `self.__current_negative_reference_embedding`.

        Args:
            word: The guessed word
            rank: The rank of the guessed word
            embedding: The embedding vector of the guessed word

        Returns:
            Tuple of (positive_embedding, negative_embedding) for context pair formation
        """
        new_guess_details = (word, rank, embedding)

        self.__current_positive_point_details = new_guess_details
        self.__positive_embeddings_for_centroid.append(embedding)

        distant_negative_embedding = self._get_random_distant_embedding(
            excluded_words=[word]
        )

        if distant_negative_embedding is None:
            logger.warning(
                "Using negation of the first guess as fallback for initial negative context."
            )
            distant_negative_embedding = -embedding.copy()

        self.__current_negative_reference_embedding = distant_negative_embedding

        return embedding, distant_negative_embedding

    def _get_random_distant_embedding(
        self, excluded_words: List[str]
    ) -> Optional[np.ndarray]:
        """
        Attempts to get an embedding of a random word not in excluded_words.

        Args:
            excluded_words: List of words to exclude from the random selection

        Returns:
            Optional embedding vector of a random word, or None if no suitable word was found
        """
        for i in range(settings.max_distant_embedding_attempts):
            try:
                random_word_str = self._fetch_random_word_from_collection()
                if random_word_str not in excluded_words:
                    return get_vector_for_word(
                        self.client, self.collection_name, random_word_str
                    )
            except (ValueError, AttributeError) as e_fetch:
                logger.warning(
                    f"Attempt {i + 1} to get random word for distant embedding failed: {e_fetch}"
                )
            except Exception as e:
                logger.warning(
                    f"Attempt {i + 1} to get random word for distant embedding failed with unexpected error: {e}"
                )

        logger.warning(
            "Failed to find a suitable random distant embedding after multiple attempts."
        )
        return None

    def _process_subsequent_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a guess after the first one, updating context reference points and solver state.

        Side effects:
            - May update `self.__current_positive_point_details`.
            - May append to `self.__positive_embeddings_for_centroid`.
            - Sets `self.__current_negative_reference_embedding`.
            - Handles a critical fallback if `self.__current_positive_point_details` is unexpectedly None.

        Args:
            word: The guessed word
            rank: The rank of the guessed word
            embedding: The embedding vector of the guessed word

        Returns:
            Tuple of (positive_embedding, negative_embedding) for context pair formation
        """
        new_guess_details = (word, rank, embedding)

        if self.__current_positive_point_details is None:
            logger.error(
                "Critical: __current_positive_point_details is None after first guess. Resetting."
            )
            self.__current_positive_point_details = new_guess_details
            self.__positive_embeddings_for_centroid.append(embedding)

            pair_positive_embedding = embedding
            pair_negative_embedding = -embedding.copy()
            self.__current_negative_reference_embedding = pair_negative_embedding
            return pair_positive_embedding, pair_negative_embedding

        _, prev_positive_rank, prev_positive_embedding = (
            self.__current_positive_point_details
        )

        if rank < prev_positive_rank:
            pair_positive_embedding = embedding
            pair_negative_embedding = prev_positive_embedding

            self.__current_positive_point_details = new_guess_details
            is_duplicate = any(
                np.array_equal(embedding, e)
                for e in self.__positive_embeddings_for_centroid
            )
            if not is_duplicate:
                self.__positive_embeddings_for_centroid.append(embedding)
            self.__current_negative_reference_embedding = prev_positive_embedding
        else:
            pair_positive_embedding = prev_positive_embedding
            pair_negative_embedding = embedding
            self.__current_negative_reference_embedding = embedding

        return pair_positive_embedding, pair_negative_embedding

    def guess_word(self) -> str:
        """
        Get the next best guess using Qdrant's Discovery Search with accumulated context.
        Follows a sequence of strategies: initial random, discovery search,
        step from best guess, and finally a general random guess.

        Returns:
            str: The suggested word to guess next.
        Raises:
            SolverUnableToGuessError: If no suitable word can be determined after all strategies.
        """
        if not self.__past_guesses:
            logger.info("No past guesses. Making an initial random guess.")
            try:
                return self._fetch_random_word_from_collection()
            except (ValueError, AttributeError) as e:
                logger.error(f"Error making initial random guess: {e}")
                raise SolverUnableToGuessError(
                    "Failed to make an initial random guess."
                ) from e

        if self.__context_pairs_for_discovery:
            logger.info("Attempting discovery search...")
            num_past_guesses = len(self.__past_guesses)
            target_vector: Optional[List[float]] = None
            limit = 10 if num_past_guesses == 1 else 1

            if num_past_guesses > 1:
                target_vector = self._determine_target_vector()

            excluded_words = {word for word, _, _ in self.__past_guesses}
            query_filter = self._create_exclusion_filter(excluded_words)

            candidate_word = self._execute_discovery_search(
                target_vector, limit, query_filter, excluded_words
            )
            if candidate_word:
                logger.info(
                    f"Discovery search successful. Suggesting word: '{candidate_word}'."
                )
                return candidate_word
            logger.warning("Discovery search yielded no new word.")
        else:
            logger.warning("No context pairs available for discovery search. Skipping.")

        logger.info("Attempting fallback: step from best guess.")
        try:
            return self._try_step_from_best_guess()
        except SolverUnableToGuessError as e_step:
            logger.warning(f"Fallback (step from best guess) failed: {e_step}")

        logger.info("Attempting ultimate fallback: random guess.")
        try:
            return self._fallback_random_guess()
        except SolverUnableToGuessError as e_random:
            logger.error("All guessing strategies exhausted.")
            raise SolverUnableToGuessError(
                "All guessing strategies failed."
            ) from e_random

    def _determine_target_vector(self) -> List[float]:
        """
        Determine the target vector for discovery search.
        Uses the centroid of positive embeddings. Assumes __positive_embeddings_for_centroid is non-empty.
        """
        logger.info(
            f"Calculating centroid from {len(self.__positive_embeddings_for_centroid)} positive embeddings."
        )

        centroid_vector = np.mean(
            np.array(self.__positive_embeddings_for_centroid), axis=0
        )
        logger.info("Using centroid of positive embeddings as target for discovery.")
        return centroid_vector.tolist()

    def _create_exclusion_filter(
        self, excluded_words: Set[str]
    ) -> Optional[rest_models.Filter]:
        """
        Create a Qdrant filter to exclude previously guessed words.

        Args:
            excluded_words: Set of words to exclude from search results

        Returns:
            Qdrant Filter object or None if no words to exclude
        """
        if not excluded_words:
            return None

        return rest_models.Filter(
            must_not=[
                rest_models.FieldCondition(
                    key="word", match=rest_models.MatchAny(any=list(excluded_words))
                )
            ]
        )

    def _execute_discovery_search(
        self,
        target_vector: Optional[List[float]],
        limit: int,
        query_filter: Optional[rest_models.Filter],
        excluded_words: Set[str],
    ) -> Optional[str]:
        """
        Execute the Qdrant discovery search and process results.

        Args:
            target_vector: Optional target vector for the search
            limit: Maximum number of results to return
            query_filter: Optional filter to exclude words (based on excluded_words)
            excluded_words: Set of words to exclude from search results, used for logging and final check.

        Returns:
            The best candidate word or None if no suitable word was found
        """
        logger.info(
            f"Performing discovery search: {len(self.__context_pairs_for_discovery)} context pairs, "
            f"target {'present' if target_vector else 'absent'}, "
            f"excluding {len(excluded_words)} words, hnsw_ef: {settings.qdrant_hnsw_ef}."
        )

        try:
            search_results = self.client.discover(
                collection_name=self.collection_name,
                target=target_vector,
                context=self.__context_pairs_for_discovery,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                search_params=rest_models.SearchParams(hnsw_ef=settings.qdrant_hnsw_ef),
            )

            if search_results:
                for hit in search_results:
                    if hit.payload and "word" in hit.payload:
                        candidate_word = hit.payload["word"]
                        if candidate_word not in excluded_words:
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

        return None

    def _try_step_from_best_guess(self) -> str:
        """
        Attempts to find a new guess by taking a random step from the current best guess.

        Returns:
            str: A suggested word.
        Raises:
            SolverUnableToGuessError: If preconditions are not met or no word is found.
        """
        if not self.__past_guesses:
            raise SolverUnableToGuessError(
                "StepFromBestGuess: Cannot attempt without past guesses."
            )

        best_guess_word, _, best_guess_embedding = self.__past_guesses[0]
        logger.info(
            f"StepFromBestGuess: Attempting from best guess '{best_guess_word}'."
        )

        # Assuming best_guess_embedding is valid if it exists
        assert best_guess_embedding is not None  # For type hinting

        embedding_dim = best_guess_embedding.shape[0]
        random_direction = np.random.randn(embedding_dim)
        norm_random_direction = np.linalg.norm(random_direction)

        if norm_random_direction <= 1e-9:
            raise SolverUnableToGuessError(
                "StepFromBestGuess: Random direction vector norm is too small."
            )
        random_direction /= norm_random_direction

        step_scale = settings.base_step_scale
        new_point_vector = best_guess_embedding + step_scale * random_direction

        excluded_words = {word for word, _, _ in self.__past_guesses}
        closest_word = find_closest_word_to_point(
            self.client,
            self.collection_name,
            new_point_vector.tolist(),
            excluded_words,
        )

        if closest_word:
            logger.info(f"StepFromBestGuess: Found closest word '{closest_word}'.")
            return closest_word
        else:
            raise SolverUnableToGuessError(
                "StepFromBestGuess: No closest word found for the new point."
            )

    def _fallback_random_guess(self) -> str:
        """
        Ultimate fallback: try to get any random word not yet guessed.
        If a new word cannot be found after attempts, returns any valid random word.

        Returns:
            str: A random word.
        Raises:
            SolverUnableToGuessError: If a random word cannot be fetched at all.
        """
        logger.info("Executing ultimate fallback: trying a random guess.")
        try:
            excluded_words = {word for word, _, _ in self.__past_guesses}
            for i in range(5):
                random_word = self._fetch_random_word_from_collection()
                if random_word not in excluded_words:
                    logger.info(
                        f"Fallback: Found new random word '{random_word}' on attempt {i + 1}."
                    )
                    return random_word

            logger.warning(
                "Fallback: Failed to find a new random word after multiple attempts. "
                "Returning last fetched random word (might be an old guess or a new one)."
            )
            return self._fetch_random_word_from_collection()
        except (ValueError, AttributeError) as e:
            logger.error(
                f"Error in _fallback_random_guess -> _fetch_random_word_from_collection: {e}"
            )
            raise SolverUnableToGuessError(
                "Ultimate fallback failed: Unable to fetch any random word from the collection."
            ) from e
