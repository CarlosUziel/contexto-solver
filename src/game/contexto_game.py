from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models  # Ensure this import is present

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import (
    get_all_similarities,
    get_collection_info,
    get_random_point,
)


class ContextoGame:
    """Manages the state and logic of a Contexto game instance."""

    def __init__(self, client: QdrantClient, collection_name: Optional[str] = None):
        """Initializes the Contexto game.

        Args:
            client: An initialized QdrantClient instance.
            collection_name: The name of the Qdrant collection to use.
                Defaults to the value in settings.effective_collection_name.

        Raises:
            ConnectionError: If connection to Qdrant fails.
            ValueError: If the specified collection is empty or cannot be found,
                        or if a target word cannot be selected or is invalid.
            RuntimeError: If similarity calculation fails.
        """
        self.collection_name: str = (
            collection_name or settings.effective_collection_name
        )
        logger.info(
            f"Initializing Contexto Game with collection: {self.collection_name}"
        )

        self.client: QdrantClient = client

        collection_info = get_collection_info(self.client, self.collection_name)
        if not collection_info or not collection_info.points_count:
            logger.error(f"Collection '{self.collection_name}' is empty or not found.")
            raise ValueError(
                f"Collection '{self.collection_name}' is empty or could not be found."
            )
        self.total_points: int = collection_info.points_count

        # --- Target Word Selection ---
        target_point: Optional[rest_models.Record] = get_random_point(
            self.client, self.collection_name, self.total_points
        )

        if (
            not target_point
            or not target_point.payload
            or "word" not in target_point.payload
            or not isinstance(
                target_point.payload["word"], str
            )  # Added type check for robustness
            or not target_point.payload[
                "word"
            ].strip()  # Ensure word is not empty or just whitespace
            or not target_point.vector
        ):
            logger.error(
                "Failed to retrieve a valid random target point, its payload, word, or vector."
            )
            raise ValueError("Could not select a valid target word for the game.")

        selected_target_word: str = target_point.payload["word"].lower()

        self.target_word: str = selected_target_word
        self.target_vector: np.ndarray = np.array(target_point.vector, dtype=np.float32)
        logger.info(
            f"Target word selected: '{self.target_word}' (ID: {target_point.id})."
        )
        # --- End of Target Word Selection ---

        # Compute all similarities
        self.word_to_rank_and_vector: Optional[Dict[str, Tuple[int, np.ndarray]]] = (
            get_all_similarities(
                self.client,
                self.collection_name,
                self.target_vector,
                self.total_points,
            )
        )
        if not self.word_to_rank_and_vector:
            logger.error("Failed to compute word similarities with vectors.")
            raise RuntimeError(
                "Could not compute word similarities with vectors for the game."
            )

        self.guesses: List[
            Tuple[str, int, np.ndarray]
        ] = []  # Store word, rank, and vector
        logger.info(
            f"Contexto Game initialized successfully for target word '{self.target_word}'. "
            f"{len(self.word_to_rank_and_vector)} word similarities ranked."
        )

    def get_word_rank_and_vector(self, word: str) -> Optional[Tuple[int, np.ndarray]]:
        """Gets the similarity rank and vector of a given word (case-insensitive).

        Args:
            word: The word to check.

        Returns:
            Optional[Tuple[int, np.ndarray]]: A tuple containing the rank (int) and vector (np.ndarray)
                                             of the word, or None if the word is not found.
        """
        if not self.word_to_rank_and_vector:
            return None
        return self.word_to_rank_and_vector.get(word.lower())

    def make_guess(self, guessed_word: str) -> Optional[int]:
        """
        Processes a player's guess and returns its rank if valid.

        Args:
            guessed_word: The word guessed by the player.

        Returns:
            The rank of the word (0-indexed) if the guess is valid and found,
            None otherwise.

        Raises:
            ValueError: If the guessed word is not in the game's vocabulary.
        """
        normalized_word = guessed_word.strip().lower()
        if not normalized_word:
            logger.warning("Attempted to guess an empty or whitespace-only word.")
            raise ValueError("Guess cannot be empty.")

        # Retrieve rank and vector using the existing method
        rank_vector_tuple = self.get_word_rank_and_vector(normalized_word)

        if rank_vector_tuple is None:
            logger.warning(f"Word '{normalized_word}' not found in game vocabulary.")
            raise ValueError(f"Word '{normalized_word}' not found in game vocabulary.")

        rank, word_vector = rank_vector_tuple

        # Add to guesses list if not already guessed
        if not any(g[0] == normalized_word for g in self.guesses):
            self.guesses.append((normalized_word, rank, word_vector))
            self.guesses.sort(key=lambda x: x[1])  # Keep sorted by rank
            logger.info(
                f"Guess '{normalized_word}' (Rank: {rank}) added. Total guesses: "
                f"{len(self.guesses)}. Target: '{self.target_word}'"
            )
        return rank

    def get_guesses(self) -> List[Tuple[str, int, np.ndarray]]:
        """Returns the list of guesses made so far, sorted by rank.

        Each guess is a tuple of (word, rank, vector).

        Returns:
            List[Tuple[str, int, np.ndarray]]: A list of tuples (word, rank, vector).
        """
        return self.guesses

    def is_game_won(self) -> bool:
        """Checks if the target word (case-insensitive) has been guessed.

        Returns:
            bool: True if the target word is in the guesses, False otherwise.
        """
        return any(guess[0] == self.target_word.lower() for guess in self.guesses)

    def get_target_word(self) -> str:
        """Returns the target word for the current game."""
        return self.target_word
