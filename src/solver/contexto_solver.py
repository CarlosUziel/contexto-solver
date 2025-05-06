from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient

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
    Solves a ContextoGame by providing best guesses based on a greedy information-gain algorithm.
    This class is independent of the game's state management.
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Initializes the ContextoSolver by fetching word embedding data.

        Args:
            client (QdrantClient): An initialized QdrantClient instance.
            collection_name (str): The name of the Qdrant collection.

        Raises:
            ValueError: If no word vectors are found in the specified collection.
        """
        self.client = client
        self.collection_name = collection_name

        logger.info(
            f"Initializing ContextoSolver for collection '{collection_name}'..."
        )

        # Store word, rank and embedding data of past guesses
        self.past_guesses: List[Tuple[str, int, np.ndarray]] = []
        self.base_step_scale = settings.base_step_scale

    def add_guess(self, word: str, rank: int) -> bool:
        """
        Adds a new guess to the solver's history if it is not already present.

        Args:
            word (str): The guessed word.
            rank (int): The rank of the guess.

        Returns:
            bool: True if the guess was added, False otherwise (e.g., word not found,
                  already guessed, or error fetching embedding).
        """
        # Check if word already guessed
        if any(past_word == word for past_word, _, _ in self.past_guesses):
            logger.info(f"Word '{word}' has already been guessed. Skipping.")
            return False

        embedding = get_vector_for_word(self.client, self.collection_name, word)
        if embedding is None:
            logger.warning(
                f"Could not retrieve embedding for word '{word}'. Guess not added."
            )
            return False

        self.past_guesses.append((word, rank, embedding))
        # sort the past guesses by rank (lower rank is better)
        self.past_guesses.sort(key=lambda x: x[1])
        logger.info(f"Added guess: Word '{word}', Rank {rank}.")
        return True

    def guess_word(self) -> str:
        """Get the next best guess based on the current game state.

        Returns:
            Optional[str]: The suggested next word to guess, or None if no guess can be made.
        """
        num_past_guesses = len(self.past_guesses)

        if num_past_guesses == 0:
            return self._solve_no_past_guesses()
        elif num_past_guesses == 1:
            return self._solve_one_past_guess()
        else:
            return self._solve_multiple_past_guesses()

    def _solve_no_past_guesses(self) -> str:
        """
        Handles the case where there are no past guesses.
        Makes a random guess from the collection.

        Returns:
            str: A random word from the collection.
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
            return random_point_record.payload.get("word")
        logger.error("Failed to get a random point.")
        raise ValueError(
            f"Failed to get a random point from collection '{self.collection_name}'."
        )

    def _solve_one_past_guess(self) -> Optional[str]:
        """
        Handles the case where there is exactly one past guess.
        Chooses a random direction, moves a step, and finds the closest word.

        Returns:
            Optional[str]: The next best guess, or None if an error occurs.
        """
        logger.info("One past guess. Choosing a random direction.")
        _, _, past_embedding = self.past_guesses[0]

        if not isinstance(past_embedding, np.ndarray) or past_embedding.ndim != 1:
            logger.error(
                f"_solve_one_past_guess: past_embedding is not a valid 1D numpy array. Shape: {getattr(past_embedding, 'shape', 'N/A')}"
            )
            return None

        embedding_dim = past_embedding.shape[0]
        if embedding_dim == 0:
            logger.error("_solve_one_past_guess: embedding_dim is 0.")
            return None

        # Generate a random direction vector
        random_direction = np.random.randn(embedding_dim)
        norm_random_direction = np.linalg.norm(random_direction)

        if norm_random_direction > 1e-9:  # Check against a small epsilon
            random_direction /= norm_random_direction
        else:
            logger.warning(
                "_solve_one_past_guess: Generated random_direction norm is close to zero. Using a fallback random guess."
            )
            return (
                self._solve_no_past_guesses()
            )  # Try a completely random guess as a fallback

        # New point is the past embedding moved by step_size in the random_direction
        new_point_vector = past_embedding + self.base_step_scale * random_direction

        return find_closest_word_to_point(
            self.client,
            self.collection_name,
            new_point_vector.tolist(),
            {word for word, _, _ in self.past_guesses},
        )

    def _solve_multiple_past_guesses(self) -> Optional[str]:
        """
        Handles the case where there are multiple past guesses.
        Computes a direction vector based on past guesses relative to the best guess,
        moves a step, and finds the closest word.

        Returns:
            Optional[str]: The next best guess, or None if an error occurs.
        """
        logger.info("Multiple past guesses. Computing a weighted direction.")
        # Past guesses are sorted by rank, so the first one is the best guess so far.
        best_guess_word, _, best_guess_embedding = self.past_guesses[0]

        if (
            not isinstance(best_guess_embedding, np.ndarray)
            or best_guess_embedding.ndim != 1
        ):
            logger.error(
                f"_solve_multiple_past_guesses: best_guess_embedding for word '{best_guess_word}' is not a valid 1D numpy array. Shape: {getattr(best_guess_embedding, 'shape', 'N/A')}"
            )
            return None

        embedding_dim = best_guess_embedding.shape[0]
        if embedding_dim == 0:
            logger.error(
                f"_solve_multiple_past_guesses: embedding_dim for word '{best_guess_word}' is 0."
            )
            return None

        direction_vectors = []
        for word, _, embedding in self.past_guesses[1:]:
            if (
                isinstance(embedding, np.ndarray)
                and embedding.shape == best_guess_embedding.shape
            ):
                direction_vectors.append(best_guess_embedding - embedding)
            else:
                logger.warning(
                    f"_solve_multiple_past_guesses: Skipping invalid embedding for word '{word}'. Shape: {getattr(embedding, 'shape', 'N/A')}"
                )

        if not direction_vectors:
            logger.warning(
                "_solve_multiple_past_guesses: No valid direction vectors could be computed. Using a random direction from best guess."
            )
            # Fallback to a random step from the best guess if no other directions are valid
            random_direction = np.random.randn(embedding_dim)
            norm_random_direction = np.linalg.norm(random_direction)
            if norm_random_direction > 1e-9:  # Check against a small epsilon
                summed_direction = random_direction / norm_random_direction
            else:
                logger.error(
                    "_solve_multiple_past_guesses: Fallback random_direction norm is close to zero. Cannot proceed."
                )
                return self._solve_no_past_guesses()  # Try a completely random guess
        else:
            summed_direction = np.sum(direction_vectors, axis=0)
            norm_summed_direction = np.linalg.norm(summed_direction)
            if norm_summed_direction > 1e-9:  # Check against a small epsilon
                summed_direction /= norm_summed_direction  # Normalize
            else:  # All direction vectors cancelled out, or were zero vectors
                logger.warning(
                    "_solve_multiple_past_guesses: Summed direction vector norm is close to zero. Using a random direction."
                )
                random_direction = np.random.randn(embedding_dim)
                norm_random_direction = np.linalg.norm(random_direction)
                if norm_random_direction > 1e-9:  # Check against a small epsilon
                    summed_direction = random_direction / norm_random_direction
                else:
                    logger.error(
                        "_solve_multiple_past_guesses: Fallback random_direction norm for zero summed_direction is also close to zero. Cannot proceed."
                    )
                    return self._solve_no_past_guesses()

        # New point is the best guess's embedding moved by step_size in the summed_direction
        new_point_vector = (
            best_guess_embedding + self.base_step_scale * summed_direction
        )

        return find_closest_word_to_point(
            self.client,
            self.collection_name,
            new_point_vector.tolist(),
            {word for word, _, _ in self.past_guesses},
        )
