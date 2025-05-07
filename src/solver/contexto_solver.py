from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models  # Required for models.Record

from config.logger import app_logger as logger
from db.utils import (
    find_closest_word_to_point,
    get_collection_info,
    get_random_point,
    get_vector_for_word,
)


class ContextoSolver:
    """
    Solves a ContextoGame by providing best guesses based on a Perceptron-style active learning algorithm.
    """

    TOP_K_FOR_CENTROID = (
        3  # Number of top-ranked good guesses to use for centroid calculation
    )
    FOCUSED_SEARCH_RANK_THRESHOLD = (
        30  # If a guess is better than this, use focused search
    )
    FOCUSED_SEARCH_STEP_SIZE = 0.15  # Step size for focused search perturbation

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
            f"Initializing ContextoSolver for collection '{collection_name}' with Perceptron strategy."
        )

        self.past_guesses: List[Tuple[str, int, np.ndarray]] = []

        self.t_hat: Optional[np.ndarray] = None
        self.embedding_dim: Optional[int] = None
        self.num_total_words: Optional[int] = None
        self.rank_threshold: Optional[int] = None
        self.last_proposed_word_for_perceptron: Optional[str] = None
        self.min_centroid_diff_norm = 1e-6  # Minimum norm for centroid difference

        try:
            self._initialize_perceptron_params()
        except ValueError as e:
            logger.error(f"Failed to initialize Perceptron params during __init__: {e}")
            # t_hat will remain None, guess_word will handle this

    def _initialize_perceptron_params(self):
        """Initializes parameters required for the Perceptron algorithm."""
        logger.info("Initializing Perceptron parameters...")
        collection_info = get_collection_info(self.client, self.collection_name)

        if (
            not collection_info
            or not collection_info.points_count
            or collection_info.points_count == 0
        ):
            raise ValueError(
                f"Cannot get collection info or collection '{self.collection_name}' is empty."
            )

        self.num_total_words = collection_info.points_count
        if self.num_total_words is None:  # Should not happen if points_count is valid
            raise ValueError("Failed to get total number of words.")
        self.rank_threshold = self.num_total_words // 2

        # Try to get embedding dimension from collection configuration if available
        if (
            collection_info.config
            and collection_info.config.params
            and collection_info.config.params.vectors
        ):
            # Assuming single vector config or taking the first one
            vectors_config = collection_info.config.params.vectors
            if isinstance(vectors_config, models.VectorParams):
                self.embedding_dim = vectors_config.size
            elif isinstance(vectors_config, dict):  # Named vectors
                first_vector_name = next(iter(vectors_config))
                self.embedding_dim = vectors_config[first_vector_name].size

        if self.embedding_dim is None:
            # Fallback: Get embedding dimension from a random point
            logger.info("Fetching a random point to determine embedding dimension.")
            # Ensure num_total_words is not None before passing to get_random_point
            if self.num_total_words is None:
                raise ValueError("num_total_words is None, cannot fetch random point.")
            random_point_record = get_random_point(
                self.client, self.collection_name, self.num_total_words
            )
            if random_point_record and random_point_record.vector:
                self.embedding_dim = len(random_point_record.vector)
            else:
                raise ValueError(
                    "Failed to get a random point or its vector to determine embedding dimension."
                )

        if self.embedding_dim is None or self.embedding_dim == 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")

        # Initialize t_hat randomly on the unit sphere
        self.t_hat = np.random.randn(self.embedding_dim)
        self.t_hat /= np.linalg.norm(self.t_hat)
        logger.info(
            f"Perceptron params initialized: dim={self.embedding_dim}, total_words={self.num_total_words}, threshold={self.rank_threshold}"
        )

    def add_guess(self, word: str, rank: int) -> bool:
        """
        Adds a new guess and updates the Perceptron estimate if applicable.

        Args:
            word (str): The guessed word.
            rank (int): The rank of the guess.

        Returns:
            bool: True if the guess was processed, False otherwise.
        """
        if any(past_word == word for past_word, _, _ in self.past_guesses):
            logger.info(f"Word '{word}' has already been guessed. Skipping.")
            return False

        embedding = get_vector_for_word(self.client, self.collection_name, word)
        if embedding is None:
            logger.warning(
                f"Could not retrieve embedding for word '{word}'. Guess not fully processed."
            )
            # Still add to past_guesses with a dummy embedding if needed, or return False
            # For Perceptron, if it was the proposed word, we can't update without embedding.
            if word == self.last_proposed_word_for_perceptron:
                self.last_proposed_word_for_perceptron = None  # Cannot update
            return False

        self.past_guesses.append((word, rank, embedding))
        self.past_guesses.sort(key=lambda x: x[1])
        logger.info(f"Added guess: Word '{word}', Rank {rank}.")

        # Perceptron update logic - only if the guess was proposed by the Perceptron strategy
        if (
            self.t_hat is not None
            and self.embedding_dim is not None
            and self.rank_threshold is not None
            and self.last_proposed_word_for_perceptron == word  # Crucial condition
        ):
            logger.info(
                f"Updating t_hat based on Perceptron-proposed guess for '{word}' (rank {rank})"
            )
            y = 1 if rank <= self.rank_threshold else -1

            # Ensure embedding is a numpy array for the update
            embedding_np = (
                np.array(embedding)
                if not isinstance(embedding, np.ndarray)
                else embedding
            )

            if embedding_np.shape[0] == self.embedding_dim:
                self.t_hat += y * embedding_np
                norm_t_hat = np.linalg.norm(self.t_hat)
                if norm_t_hat > 1e-9:  # Avoid division by zero
                    self.t_hat /= norm_t_hat
                else:
                    logger.warning(
                        "t_hat norm is close to zero after update. Re-initializing t_hat randomly."
                    )
                    self.t_hat = np.random.randn(self.embedding_dim)  # Re-initialize
                    if np.linalg.norm(self.t_hat) > 1e-9:
                        self.t_hat /= np.linalg.norm(self.t_hat)
                    else:  # Extremely unlikely, but handle
                        logger.error(
                            "Failed to re-initialize t_hat with non-zero norm."
                        )
                        self.t_hat = None  # Mark as uninitialized
            else:
                logger.warning(
                    f"Skipping t_hat update. Embedding shape mismatch for '{word}'. Expected dim {self.embedding_dim}, got {embedding_np.shape[0]}"
                )
            self.last_proposed_word_for_perceptron = None  # Reset after processing
        return True

    def _get_basic_random_guess(self) -> Optional[str]:
        """A basic fallback to get any random word, trying to avoid repeats."""
        logger.info("Attempting a basic random guess as a fallback.")
        if not self.num_total_words:  # Should be set by _initialize_perceptron_params
            collection_info = get_collection_info(self.client, self.collection_name)
            if not collection_info or not collection_info.points_count:
                logger.error(
                    "Fallback: Cannot get collection info or collection is empty."
                )
                return None
            current_total_words = collection_info.points_count
        else:
            current_total_words = self.num_total_words

        if not current_total_words:
            return None

        guessed_words_set = {word for word, _, _ in self.past_guesses}
        for _ in range(10):  # Try a few times
            random_point_record = get_random_point(
                self.client, self.collection_name, current_total_words
            )
            if random_point_record and random_point_record.payload:
                word = random_point_record.payload.get("word")
                if word and word not in guessed_words_set:
                    return word

        # If still not found or all attempts failed, get any random (might be a repeat)
        random_point_record = get_random_point(
            self.client, self.collection_name, current_total_words
        )
        if random_point_record and random_point_record.payload:
            return random_point_record.payload.get("word")
        return None

    def guess_word(self) -> Optional[str]:
        """Get the next best guess, incorporating a focused search for top guesses."""
        if self.embedding_dim is None:  # Basic check, full init check later
            try:
                self._initialize_perceptron_params()
                if self.embedding_dim is None:  # Still not set after attempt
                    raise ValueError("Initialization did not set embedding_dim.")
            except ValueError as e:
                logger.error(
                    f"Failed to initialize Perceptron params for guess_word: {e}. Falling back."
                )
                return self._get_basic_random_guess()

        guessed_words_set = {word for word, _, _ in self.past_guesses}
        v_query_direction: Optional[np.ndarray] = None
        self.last_proposed_word_for_perceptron = None  # Reset before trying to set it

        # --- Strategy 0: Focused search if a very good guess exists ---
        best_guess_for_focused_search = None
        if (
            self.past_guesses and self.past_guesses[0][2] is not None
        ):  # Embedding must exist
            # self.past_guesses is sorted by rank, [0] is the best
            if self.past_guesses[0][1] <= self.FOCUSED_SEARCH_RANK_THRESHOLD:
                best_guess_for_focused_search = self.past_guesses[0]

        if (
            best_guess_for_focused_search
            and self.t_hat is not None
            and self.embedding_dim is not None
        ):
            best_word, best_rank, best_embedding_val = best_guess_for_focused_search

            best_embedding_np: np.ndarray
            if isinstance(best_embedding_val, np.ndarray):
                best_embedding_np = best_embedding_val
            else:  # Assuming it's a list
                best_embedding_np = np.array(best_embedding_val)

            if best_embedding_np.shape[0] == self.embedding_dim:
                logger.info(
                    f"Attempting focused search around best guess: '{best_word}' (rank {best_rank}) with step {self.FOCUSED_SEARCH_STEP_SIZE}."
                )
                random_direction = np.random.randn(self.embedding_dim)
                # Project random_direction to be orthogonal to t_hat
                # Ensure t_hat is a 1D array for dot product
                t_hat_1d = (
                    self.t_hat.squeeze()
                )  # Squeeze to handle (dim,1) or (1,dim) if necessary
                if t_hat_1d.ndim != 1:
                    logger.error(
                        f"Focused search: t_hat has unexpected dimensions {self.t_hat.shape} after squeeze. Falling back."
                    )
                else:
                    orthogonal_component = (
                        random_direction - np.dot(random_direction, t_hat_1d) * t_hat_1d
                    )
                    norm_orth_comp = np.linalg.norm(orthogonal_component)

                    if norm_orth_comp > 1e-9:
                        exploration_vector = orthogonal_component / norm_orth_comp
                        v_query_direction_candidate = (
                            best_embedding_np
                            + self.FOCUSED_SEARCH_STEP_SIZE * exploration_vector
                        )

                        if (
                            np.linalg.norm(
                                v_query_direction_candidate - best_embedding_np
                            )
                            > 1e-7
                        ):  # Ensure it's a new point
                            v_query_direction = v_query_direction_candidate
                            logger.info("Using focused search strategy vector.")
                        else:
                            logger.info(
                                "Focused search vector too close to best guess, will fallback."
                            )
                    else:
                        logger.warning(
                            "Focused search: Orthogonal component norm is too small. Falling back."
                        )
            else:
                logger.warning(
                    f"Focused search: Best guess '{best_word}' embedding dim {best_embedding_np.shape} mismatch with expected {self.embedding_dim}. Falling back."
                )

        # --- Fallback Strategies if Focused Search didn't apply ---
        if v_query_direction is None:
            if (
                self.t_hat is None or self.rank_threshold is None
            ):  # Check if Perceptron is ready for other strategies
                logger.warning(
                    "Perceptron params (t_hat/rank_threshold) not ready for centroid/random strategies. Attempting init."
                )
                try:
                    self._initialize_perceptron_params()
                    if self.t_hat is None or self.rank_threshold is None:
                        raise ValueError(
                            "Initialization did not set all required Perceptron parameters for fallback."
                        )
                except ValueError as e:
                    logger.error(
                        f"Failed to initialize Perceptron params for fallback: {e}. Using basic random guess."
                    )
                    return self._get_basic_random_guess()

            # Strategy 1: Top-K good guesses centroid projection
            all_good_guess_tuples = [
                guess_tuple
                for guess_tuple in self.past_guesses
                if guess_tuple[2] is not None and guess_tuple[1] <= self.rank_threshold
            ]

            if len(all_good_guess_tuples) >= 1:
                logger.info(
                    f"Attempting Top-K (K={self.TOP_K_FOR_CENTROID}) informed guess based on best past guesses."
                )
                top_k_embeddings_to_average = [
                    emb
                    for _, _, emb in all_good_guess_tuples[: self.TOP_K_FOR_CENTROID]
                ]

                if top_k_embeddings_to_average:
                    centroid_top_k = np.mean(top_k_embeddings_to_average, axis=0)
                    t_hat_1d = self.t_hat.squeeze()
                    if t_hat_1d.ndim == 1 and centroid_top_k.ndim == 1:
                        dot_product_top_k = np.dot(t_hat_1d, centroid_top_k)
                        v_orthogonal_top_k = centroid_top_k - (
                            t_hat_1d * dot_product_top_k
                        )
                        norm_v_orthogonal_top_k = np.linalg.norm(v_orthogonal_top_k)

                        if norm_v_orthogonal_top_k > 1e-9:
                            v_query_direction = (
                                v_orthogonal_top_k / norm_v_orthogonal_top_k
                            )
                            logger.info(
                                "Using Top-K informed direction vector from best guesses."
                            )
                        else:
                            logger.warning(
                                "Top-K informed orthogonalized vector norm is close to zero. Falling back."
                            )
                    else:
                        logger.warning(
                            "t_hat or centroid_top_k has unexpected dimensions for dot product in Top-K strategy. Falling back."
                        )

            # Strategy 2: Fallback to general good/bad centroid difference projection
            if v_query_direction is None:
                good_guesses_embeddings = [emb for _, _, emb in all_good_guess_tuples]
                bad_guesses_embeddings = [
                    emb
                    for _, r, emb in self.past_guesses
                    if emb is not None and r > self.rank_threshold
                ]

                if len(good_guesses_embeddings) > 0 and len(bad_guesses_embeddings) > 0:
                    logger.info(
                        "Attempting informed guess based on general good/bad centroids."
                    )
                    centroid_good = np.mean(good_guesses_embeddings, axis=0)
                    centroid_bad = np.mean(bad_guesses_embeddings, axis=0)
                    v_candidate_informed = centroid_good - centroid_bad
                    norm_v_candidate_informed = np.linalg.norm(v_candidate_informed)

                    if norm_v_candidate_informed > self.min_centroid_diff_norm:
                        t_hat_1d = self.t_hat.squeeze()
                        if t_hat_1d.ndim == 1 and v_candidate_informed.ndim == 1:
                            dot_product_informed = np.dot(
                                t_hat_1d, v_candidate_informed
                            )
                            v_orthogonal_informed = v_candidate_informed - (
                                t_hat_1d * dot_product_informed
                            )
                            norm_v_orthogonal_informed = np.linalg.norm(
                                v_orthogonal_informed
                            )

                            if norm_v_orthogonal_informed > 1e-9:
                                v_query_direction = (
                                    v_orthogonal_informed / norm_v_orthogonal_informed
                                )
                                logger.info(
                                    "Using general good/bad centroid informed direction vector."
                                )
                            else:
                                logger.warning(
                                    "General informed orthogonalized vector norm is close to zero. Falling back."
                                )
                        else:
                            logger.warning(
                                "t_hat or v_candidate_informed has unexpected dimensions for dot product in general centroid strategy. Falling back."
                            )
                    else:
                        logger.info(
                            "General centroid difference norm too small. Falling back."
                        )
                else:
                    logger.info(
                        "Not enough good/bad guesses for general centroid strategy. Falling back if needed."
                    )

            # Strategy 3: Fallback to random projection
            if v_query_direction is None:
                logger.info("Using random projection strategy for guess.")
                v_candidate_raw = np.random.randn(self.embedding_dim)
                t_hat_1d = self.t_hat.squeeze()
                if (
                    t_hat_1d.ndim != 1
                ):  # Should be caught by init checks earlier, but defensive
                    logger.error(
                        f"Random projection: t_hat has unexpected dimensions {self.t_hat.shape}. Basic random guess."
                    )
                    return self._get_basic_random_guess()

                dot_product_random = np.dot(t_hat_1d, v_candidate_raw)
                v_orthogonal_random = v_candidate_raw - (t_hat_1d * dot_product_random)
                norm_v_orthogonal_random = np.linalg.norm(v_orthogonal_random)

                if norm_v_orthogonal_random < 1e-9:
                    logger.warning(
                        "Random orthogonalized vector norm is close to zero. Using a new random normalized vector for query."
                    )
                    v_orthogonal_random = np.random.randn(
                        self.embedding_dim
                    )  # Fresh random vector
                    norm_fallback = np.linalg.norm(v_orthogonal_random)
                    if norm_fallback < 1e-9:  # Extremely unlikely
                        logger.error(
                            "Fallback random vector for query is also near zero. Cannot proceed."
                        )
                        return self._get_basic_random_guess()
                    v_query_direction = v_orthogonal_random / norm_fallback
                else:
                    v_query_direction = v_orthogonal_random / norm_v_orthogonal_random

        # If after all strategies, v_query_direction is still None, something is wrong.
        if v_query_direction is None:
            logger.error(
                "All guessing strategies failed to produce a query direction. Falling back to basic random guess."
            )
            return self._get_basic_random_guess()

        # 4. Find the closest actual word in the Qdrant collection to the chosen v_query_direction
        logger.info(
            "Finding closest word to the chosen query direction vector for Perceptron strategy."
        )
        query_vector_list = v_query_direction.tolist()

        next_guess_word = find_closest_word_to_point(
            self.client,
            self.collection_name,
            query_vector_list,
            guessed_words_set,
        )

        if next_guess_word:
            self.last_proposed_word_for_perceptron = (
                next_guess_word  # Mark that this came from Perceptron logic
            )
            logger.info(f"Proposing guess: '{next_guess_word}'")
            return next_guess_word
        else:
            logger.warning(
                "find_closest_word_to_point returned None. Falling back to basic random guess."
            )
            return self._get_basic_random_guess()
