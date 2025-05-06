from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient

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
            client (QdrantClient): An initialized QdrantClient instance.
            collection_name (Optional[str]): The name of the Qdrant collection to use.
                Defaults to the value in settings.glove_dataset.

        Raises:
            ConnectionError: If connection to Qdrant fails.
            ValueError: If the specified collection is empty or cannot be found,
                        or if a target word cannot be selected or is invalid.
            RuntimeError: If similarity calculation fails.
        """
        self.collection_name = collection_name or settings.glove_dataset
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

        # Select target word
        target_point = get_random_point(
            self.client, self.collection_name, self.total_points
        )
        if (
            not target_point
            or not target_point.vector
            or not target_point.payload
            or "word" not in target_point.payload
        ):
            logger.error("Failed to retrieve a valid target point.")
            raise ValueError("Could not select a valid target word for the game.")

        self.target_word: str = target_point.payload["word"]
        self.target_vector: np.ndarray = np.array(target_point.vector, dtype=np.float32)
        logger.info(
            f"Target word selected: '{self.target_word}' (ID: {target_point.id})"
        )

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
            f"Contexto Game initialized successfully for target '{self.target_word}'. "
            f"{len(self.word_to_rank_and_vector)} word similarities ranked."
        )

    def get_word_rank_and_vector(self, word: str) -> Optional[Tuple[int, np.ndarray]]:
        """Gets the similarity rank and vector of a given word (case-insensitive).

        Args:
            word (str): The word to check.

        Returns:
            Optional[Tuple[int, np.ndarray]]: A tuple (rank, vector), or None if
                                               the word is not found.
        """
        if not self.word_to_rank_and_vector:
            return None
        return self.word_to_rank_and_vector.get(word.lower())

    def make_guess(self, word: str) -> Optional[int]:
        """Processes a player's guess (case-insensitive).

        Checks if the word is valid, records it with its rank and vector.
        Guesses are stored in lowercase.

        Args:
            word (str): The word guessed by the player.

        Returns:
            Optional[int]: The rank of the guessed word if valid, otherwise None.
        """
        normalized_word = word.lower()
        rank_and_vector_data = self.get_word_rank_and_vector(normalized_word)

        if rank_and_vector_data is not None:
            rank, vector = rank_and_vector_data
            # Avoid adding duplicate guesses
            if not any(guess[0] == normalized_word for guess in self.guesses):
                self.guesses.append((normalized_word, rank, vector))
                # Sort guesses by rank after adding
                self.guesses.sort(key=lambda item: item[1])
                logger.info(f"Guess added: '{normalized_word}' (Rank: {rank})")
            else:
                logger.debug(f"Duplicate guess ignored: '{normalized_word}'")
            return rank
        else:
            logger.info(
                f"Invalid guess: '{normalized_word}' not found in similarity list."
            )
            return None

    def get_guesses(self) -> List[Tuple[str, int, np.ndarray]]:
        """Returns the list of guesses made so far, sorted by rank.

        Each guess is a tuple of (word, rank, vector), where word is lowercase.

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
        """Returns the target word for the current game.

        Returns:
            str: The target word.
        """
        return self.target_word
