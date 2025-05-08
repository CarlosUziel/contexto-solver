from typing import List, Optional, Set, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import (
    create_qdrant_exclusion_filter,
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
            f"Initializing ContextoSolver for collection '{collection_name}'"
            " with iterative discovery strategy."
        )

        collection_info = get_collection_info(self.client, self.collection_name)
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
        self.collection_info = collection_info

        self.__past_guesses: List[Tuple[str, int, np.ndarray]] = []
        self.__context_pairs_for_discovery: List[rest_models.ContextExamplePair] = []
        self.__positive_embeddings_for_centroid: List[np.ndarray] = []
        self.__current_positive_point_details: Optional[Tuple[str, int, np.ndarray]] = (
            None
        )

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
                f"Could not retrieve a valid embedding for word '{word}'. "
                "Guess not added, context not updated."
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
                "Failed to determine valid positive/negative embeddings for context pair. "
                "Pair not added."
            )

        self.__past_guesses.append((word, rank, embedding))
        self.__past_guesses.sort(key=lambda x: x[1])

        logger.info(
            f"Added guess: Word '{word}', Rank {rank}. Total past guesses: {len(self.__past_guesses)}."
        )
        return True

    def _fetch_random_word_from_collection(
        self, excluded_words: Optional[Set[str]] = None
    ) -> str:
        """
        Fetches a random word from the Qdrant collection.
        Can optionally exclude a set of words.
        Assumes __init__ has validated the collection's initial state.
        Handles potential issues if collection state changes post-init.

        Args:
            excluded_words (Optional[Set[str]], optional): A set of words to exclude. Defaults to None.

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
            f"Attempting to fetch a random word from collection '{self.collection_name}'"
            f"{(', excluding specified words.' if excluded_words else '.')}"
        )

        random_point_record = get_random_point(
            self.client,
            self.collection_name,
            self.collection_info.points_count,
            excluded_words=excluded_words,
        )

        if random_point_record and random_point_record.payload:
            word = random_point_record.payload.get("word")
            if word and isinstance(word, str) and word.strip():
                logger.debug(f"Successfully fetched random word: '{word}'.")
                return word

        logger.error(
            f"Failed to get a random point or extract a valid word from its payload "
            f"from collection '{self.collection_name}'."
        )
        raise ValueError(
            f"Failed to get a random point/word from collection "
            f"'{self.collection_name}'."
        )

    def _process_first_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the first guess, updating solver state and determining initial context pair embeddings.
        The negative context is always the negation of the first guess's embedding.

        Side effects:
            - Sets `self.__current_positive_point_details`.
            - Appends to `self.__positive_embeddings_for_centroid`.

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

        logger.info(
            "Using negation of the first guess's embedding as the initial "
            "negative context."
        )
        distant_negative_embedding = -embedding.copy()

        return embedding, distant_negative_embedding

    def _process_subsequent_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a guess after the first one, updating context reference points and solver state.

        Side effects:
            - May update `self.__current_positive_point_details`.
            - May append to `self.__positive_embeddings_for_centroid`.

        Args:
            word: The guessed word
            rank: The rank of the guessed word
            embedding: The embedding vector of the guessed word

        Returns:
            Tuple of (positive_embedding, negative_embedding) for context pair formation
        """
        new_guess_details = (word, rank, embedding)

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
        else:
            pair_positive_embedding = prev_positive_embedding
            pair_negative_embedding = embedding

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

        num_past_guesses = len(self.__past_guesses)
        target_vector: Optional[List[float]] = None
        limit = 8 if num_past_guesses == 1 else 1

        if num_past_guesses > 1:
            logger.info(
                f"Calculating centroid from "
                f"{len(self.__positive_embeddings_for_centroid)} positive embeddings."
            )
            target_vector = np.mean(
                np.array(self.__positive_embeddings_for_centroid), axis=0
            ).tolist()
            logger.info(
                "Using centroid of positive embeddings as target for discovery."
            )

        excluded_words = {word for word, _, _ in self.__past_guesses}
        query_filter = create_qdrant_exclusion_filter(excluded_words)

        return self._execute_discovery_search(
            target_vector, limit, query_filter, excluded_words
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

            for hit in search_results:
                if hit.payload and "word" in hit.payload:
                    candidate_word = hit.payload["word"]
                    if candidate_word not in excluded_words:
                        logger.info(
                            f"Discovery search suggested: '{candidate_word}' (score: {hit.score})"
                        )
                        return candidate_word
            logger.warning(
                "Discovery search returned candidates, but none were new or suitable "
                "after processing."
            )
            raise SolverUnableToGuessError("Discovery search returned no results.")

        except Exception as e:
            raise SolverUnableToGuessError(f"Discovery search failed: {e}") from e
