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

        self.__guessed_words_set: Set[str] = set()
        self.__not_allowed_words_set: Set[str] = set()  # New set for not allowed words
        self.__context_pairs_for_discovery: List[rest_models.ContextExamplePair] = []
        self.__ranked_positive_embeddings: List[
            Tuple[int, np.ndarray]
        ] = []  # Replaces __positive_embeddings_for_centroid
        self.__current_positive_point_details: Optional[Tuple[int, np.ndarray]] = (
            None  # Stores (rank, embedding)
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
        normalized_word = word.strip().lower()
        if not normalized_word:
            logger.warning("Attempted to add an empty string as a guess. Skipping.")
            return False

        # 1. Check if the word has already been guessed
        if normalized_word in self.__guessed_words_set:
            logger.info(
                f"Word '{normalized_word}' has already been guessed. Skipping context update."
            )
            return False

        # 2. Get the embedding for the word
        embedding = get_vector_for_word(
            self.client, self.collection_name, normalized_word
        )
        if embedding is None:
            logger.warning(
                f"Could not retrieve a valid embedding for word '{normalized_word}'. "
                "Guess not added, context not updated."
            )
            return False

        # 3. Initialize variables for context pair embeddings
        pair_positive_embedding: Optional[np.ndarray] = None
        pair_negative_embedding: Optional[np.ndarray] = None

        # 4. Process the guess based on whether it's the first guess or a subsequent one
        if (
            not self.__guessed_words_set
        ):  # Checks the set before adding the current normalized_word
            pair_positive_embedding, pair_negative_embedding = (
                self._process_first_guess(normalized_word, rank, embedding)
            )
        else:
            pair_positive_embedding, pair_negative_embedding = (
                self._process_subsequent_guess(normalized_word, rank, embedding)
            )

        # 5. Add the new context pair if valid embeddings were determined
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
                "Pair not added."
            )

        # 6. Add the guess to guessed words set
        self.__guessed_words_set.add(normalized_word)

        logger.info(
            f"Added guess: Word '{normalized_word}', Rank {rank}. "
            f"Total past guesses: {len(self.__guessed_words_set)}."
        )
        return True

    def mark_word_as_not_allowed(self, word: str) -> bool:
        """
        Marks a word as not allowed by the game, ensuring it won't be suggested again.

        This method adds the word to a specific set of 'not allowed' words and also
        to the general set of 'guessed' words, which is used for exclusion in
        suggestion algorithms.

        Args:
            word: The word that is not allowed by the game.

        Returns:
            bool: True if the word's status as 'not allowed' was newly recorded.
                  False if the word was already known to be 'not allowed' or if the input word is empty.
        """
        normalized_word = word.strip().lower()
        if not normalized_word:
            logger.warning(
                "Attempted to mark an empty string as not allowed. Skipping."
            )
            return False

        # Add to general exclusion list first. If it's already there, no harm.
        # This ensures that even if it was a valid guess before, it's now also treated
        # as something to be excluded if it wasn't already.
        self.__guessed_words_set.add(normalized_word)

        if normalized_word in self.__not_allowed_words_set:
            logger.info(
                f"Word '{normalized_word}' was already marked as not allowed. No change in status."
            )
            return False  # Already known to be specifically 'not allowed'

        logger.info(
            f"Marking word '{normalized_word}' as not allowed by the game. "
            "It will be excluded from future suggestions."
        )
        self.__not_allowed_words_set.add(normalized_word)
        return True

    def _process_first_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the first guess, updating solver state and determining initial context pair embeddings.
        The negative context is always the negation of the first guess's embedding.

        Side effects:
            - Calls _update_positive_references to set current positive point and update centroid data.

        Args:
            word: The guessed word
            rank: The rank of the guessed word
            embedding: The embedding vector of the guessed word

        Returns:
            Tuple of (positive_embedding, negative_embedding) for context pair formation
        """
        self._update_positive_references(word, rank, embedding)
        return embedding, -embedding.copy()

    def _update_positive_references(self, word: str, rank: int, embedding: np.ndarray):
        """
        Updates the solver's current positive reference point (rank and embedding)
        and updates the list of top-ranked positive embeddings for centroid calculation.
        The list will store up to the 3 lowest ranked positive embeddings.
        The word is used for logging.

        Args:
            word: The word of the new positive reference (for logging).
            rank: The rank of the new positive reference.
            embedding: The embedding of the new positive reference.
        """
        self.__current_positive_point_details = (rank, embedding)

        # Add new embedding with its rank
        self.__ranked_positive_embeddings.append((rank, embedding))

        # Sort by rank (lowest rank first)
        self.__ranked_positive_embeddings.sort(key=lambda x: x[0])

        # Keep only the top 3
        self.__ranked_positive_embeddings = self.__ranked_positive_embeddings[:3]

        logger.info(
            f"Updated current positive reference to: Word '{word}', Rank {rank}. "
            f"Total top-ranked positive embeddings for centroid: {len(self.__ranked_positive_embeddings)}."
        )

    def _process_subsequent_guess(
        self, word: str, rank: int, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a guess after the first one, updating context reference points and solver state.

        Side effects:
            - May call _update_positive_references if the current guess is better than the previous positive.

        Args:
            word: The guessed word
            rank: The rank of the guessed word
            embedding: The embedding vector of the guessed word

        Returns:
            Tuple of (positive_embedding, negative_embedding) for context pair formation
        """
        # 1. Get details of the previous best positive guess (rank and embedding)
        prev_positive_rank, prev_positive_embedding = (
            self.__current_positive_point_details
        )

        # 2. Determine positive and negative embeddings for the context pair based on rank
        if rank < prev_positive_rank:
            pair_positive_embedding = embedding
            pair_negative_embedding = prev_positive_embedding
            self._update_positive_references(word, rank, embedding)
        else:
            pair_positive_embedding = prev_positive_embedding
            pair_negative_embedding = embedding

        return pair_positive_embedding, pair_negative_embedding

    def guess_word(self) -> str:
        """
        Get the next best guess using Qdrant's Discovery Search with accumulated context.
        Follows a sequence of strategies: initial random, discovery search,
        and finally a general random guess if discovery fails.

        Returns:
            str: The suggested word to guess next.
        Raises:
            SolverUnableToGuessError: If no suitable word can be determined after all strategies.
        """
        # 1. If no past guesses, fetch and return an initial random word
        if not self.__guessed_words_set:  # Changed from __past_guesses
            logger.info("No past guesses. Making an initial random guess.")
            try:
                # No exclusions for the very first guess
                return self._fetch_random_word_from_collection()
            except (ValueError, AttributeError) as e:
                logger.error(f"Error making initial random guess: {e}")
                raise SolverUnableToGuessError(
                    "Failed to make an initial random guess."
                ) from e

        # Initialize target vector for discovery search.
        # It will be calculated from the top (up to 3) lowest ranked positive embeddings.
        target_vector: Optional[List[float]] = None

        if self.__ranked_positive_embeddings:
            embeddings_for_centroid = [
                data[1] for data in self.__ranked_positive_embeddings
            ]

            if embeddings_for_centroid:
                logger.info(
                    f"Calculating centroid from {len(embeddings_for_centroid)} "
                    f"top-ranked positive embeddings (up to 3 lowest ranks)."
                )
                target_vector = np.mean(
                    np.array(embeddings_for_centroid), axis=0
                ).tolist()
                logger.info(
                    "Using centroid of top-ranked positive embeddings as target for discovery."
                )
            else:
                # This state (empty embeddings_for_centroid despite non-empty __ranked_positive_embeddings)
                # would indicate an issue if __ranked_positive_embeddings contained non-embedding data by mistake.
                logger.warning(
                    "__ranked_positive_embeddings non-empty, but no embeddings extracted for centroid. "
                    "Target for discovery will be absent."
                )
        else:
            # This state should ideally not be reached if at least one guess has been processed,
            # as the initial guess populates __ranked_positive_embeddings.
            logger.warning(
                "No ranked positive embeddings found for centroid calculation. "
                "Target for discovery will be absent."
            )

        # 4. Execute discovery search
        candidate_word = self._execute_discovery_search(
            target_vector, self.__guessed_words_set
        )

        # 5. Return candidate word or raise an exception if discovery fails
        if candidate_word:
            return candidate_word
        else:
            logger.error(
                "Discovery search did not yield a result. This is a critical failure."
            )
            raise SolverUnableToGuessError(
                "Discovery search failed to find a candidate word. Solver cannot proceed."
            )

    def _fetch_random_word_from_collection(
        self, excluded_words: Optional[Set[str]] = None
    ) -> str:
        """
        Fetches a random word from the Qdrant collection.
        Can optionally exclude a set of words.
        Assumes __init__ has validated the collection's initial state.
        Handles potential issues if collection state changes post-init.

        Args:
            excluded_words (Optional[Set[str]], optional): A set of words to exclude.
                Defaults to None.

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

        # 1. Get a random point from the Qdrant collection
        random_point_record = get_random_point(
            self.client,
            self.collection_name,
            self.collection_info.points_count,
            excluded_words=excluded_words,
        )

        # 2. Validate the random point and extract the word from its payload
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

    def _execute_discovery_search(
        self,
        target_vector: Optional[List[float]],
        excluded_words: Set[str],
    ) -> Optional[str]:
        """
        Execute the Qdrant discovery search and process results.
        Always fetches and processes only the top 1 result.

        Args:
            target_vector: Optional target vector for the search
            excluded_words: Set of words to exclude from search results.

        Returns:
            The best candidate word or None if no suitable word was found
        """
        query_filter = create_qdrant_exclusion_filter(excluded_words)
        logger.info(
            f"Performing discovery search: {len(self.__context_pairs_for_discovery)} "
            f"context pairs, target {'present' if target_vector else 'absent'}, "
            f"excluding {len(excluded_words)} words, hnsw_ef: {settings.qdrant_hnsw_ef}."
        )

        try:
            # 1. Perform discovery search using the Qdrant client, limit is hardcoded to 1
            search_results = self.client.discover(
                collection_name=self.collection_name,
                target=target_vector,
                context=self.__context_pairs_for_discovery,
                limit=1,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                search_params=rest_models.SearchParams(hnsw_ef=settings.qdrant_hnsw_ef),
            )

            # 2. Iterate through search results to find a suitable candidate word
            logger.info(
                f"Discovery search suggested: '{search_results[0].payload['word']}' "
                f"(score: {search_results[0].score})"
            )
            return search_results[0].payload["word"]

        except Exception as e:
            raise SolverUnableToGuessError(f"Discovery search failed: {e}") from e
