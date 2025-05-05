import random

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

from config.logger import app_logger as logger
from config.settings import settings


@st.cache_resource
def get_qdrant_client():
    logger.info("Connecting to Qdrant...")
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_host_port,
            grpc_port=settings.qdrant_grpc_host_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            https=False,
        )
        logger.info("Successfully connected to Qdrant.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        st.error(f"Failed to connect to Qdrant: {e}")
        return None


@st.cache_data(ttl=3600)
def get_collection_info(_client: QdrantClient, collection_name: str):
    logger.info(f"Getting info for collection: {collection_name}")
    try:
        info = _client.get_collection(collection_name=collection_name)
        logger.info(f"Collection info: {info}")
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        st.error(f"Failed to get collection info for '{collection_name}': {e}")
        return None


@st.cache_data(ttl=3600)
def get_target_word_and_similarities(
    _client: QdrantClient, collection_name: str, total_points: int
):
    logger.info("Selecting target word and calculating similarities...")
    try:
        # 1. Select a random point ID
        random_id = random.randint(0, total_points - 1)
        logger.info(f"Selected random ID: {random_id}")

        # 2. Retrieve the target point (vector and payload)
        target_points = _client.retrieve(
            collection_name=collection_name,
            ids=[random_id],
            with_payload=True,
            with_vectors=True,
        )
        if not target_points:
            st.error(f"Could not retrieve target point with ID: {random_id}")
            return None, None, None
        target_point = target_points[0]
        target_word = target_point.payload.get("word", "Unknown")
        target_vector = target_point.vector
        logger.info(f"Target word: {target_word}")

        # 3. Get all points sorted by similarity to the target vector
        # Use search with limit=total_points+1 just in case, offset=0
        search_result = _client.search(
            collection_name=collection_name,
            query_vector=target_vector,
            query_filter=None,  # No filter
            limit=total_points,  # Get all points
            with_payload=True,  # Need the 'word' payload
            with_vectors=False,  # Don't need vectors here
        )
        logger.info(f"Retrieved {len(search_result)} similar points.")

        # 4. Create a ranked list and a word-to-rank mapping
        ranked_list = []
        word_to_rank = {}
        for rank, hit in enumerate(search_result, 1):
            word = hit.payload.get("word", "Unknown")
            score = hit.score
            ranked_list.append(
                {"rank": rank, "word": word, "score": score, "id": hit.id}
            )
            word_to_rank[word.lower()] = {
                "rank": rank,
                "score": score,
                "id": hit.id,
            }  # Use lower case for matching

        logger.info("Finished calculating similarities and ranks.")
        return target_word, ranked_list, word_to_rank

    except Exception as e:
        logger.error(f"Error getting target word/similarities: {e}", exc_info=True)
        st.error(f"An error occurred while setting up the game: {e}")
        return None, None, None


st.set_page_config(page_title="Contexto Clone", layout="wide")
st.title("Contexto Clone")

client = get_qdrant_client()
collection_name = settings.glove_dataset  # Use dataset name from settings

if client:
    collection_info = get_collection_info(client, collection_name)

    if collection_info:
        total_points = collection_info.points_count

        # Initialize game state in session state if not present
        if "game_ready" not in st.session_state:
            st.session_state.game_ready = False
            st.session_state.target_word = None
            st.session_state.ranked_list = None
            st.session_state.word_to_rank = None
            st.session_state.guesses_df = pd.DataFrame(
                columns=["Guess #", "Word", "Rank", "Similarity"]
            )
            st.session_state.guess_count = 0

        # Button to start a new game
        if st.button("Start New Game") or not st.session_state.game_ready:
            with st.spinner("Setting up new game..."):
                target_word, ranked_list, word_to_rank = (
                    get_target_word_and_similarities(
                        client, collection_name, total_points
                    )
                )
                if target_word and ranked_list and word_to_rank:
                    st.session_state.target_word = target_word
                    st.session_state.ranked_list = ranked_list
                    st.session_state.word_to_rank = word_to_rank
                    st.session_state.guesses_df = pd.DataFrame(
                        columns=["Guess #", "Word", "Rank", "Similarity"]
                    )
                    st.session_state.guess_count = 0
                    st.session_state.game_ready = True
                    st.success(
                        f"New game started! Target word has been chosen (ID: {st.session_state.word_to_rank[target_word.lower()]['id']}). Good luck!"
                    )
                    # Force rerun to update UI elements based on new state
                    st.rerun()
                else:
                    st.error("Failed to initialize the game.")
                    st.session_state.game_ready = False

        # --- Game Play Area ---
        if st.session_state.game_ready:
            st.write(f"Guess the secret word! There are {total_points} words in total.")

            # Input form
            with st.form(key="guess_form"):
                guess_word = (
                    st.text_input("Enter your guess:", key="guess_input")
                    .lower()
                    .strip()
                )
                submit_button = st.form_submit_button(label="Guess")

            if submit_button and guess_word:
                st.session_state.guess_count += 1
                word_info = st.session_state.word_to_rank.get(guess_word)

                if word_info:
                    rank = word_info["rank"]
                    similarity = word_info["score"]

                    # Add guess to DataFrame
                    new_guess = pd.DataFrame(
                        [
                            {
                                "Guess #": st.session_state.guess_count,
                                "Word": guess_word,
                                "Rank": rank,
                                "Similarity": f"{similarity:.4f}",  # Format similarity
                            }
                        ]
                    )
                    st.session_state.guesses_df = pd.concat(
                        [st.session_state.guesses_df, new_guess], ignore_index=True
                    )

                    # Check for win
                    if guess_word == st.session_state.target_word.lower():
                        st.balloons()
                        st.success(
                            f"Congratulations! You found the secret word '{st.session_state.target_word}' in {st.session_state.guess_count} guesses!"
                        )
                        # Optionally reveal the full list or offer a new game
                    else:
                        # Clear the input by rerunning - Streamlit widgets are stateless by default
                        # A bit of a hack, but common in Streamlit for clearing forms post-submit
                        st.rerun()

                else:
                    st.warning(
                        f"The word '{guess_word}' was not found in the dataset. Try again."
                    )
                    # Decrement guess count as it was invalid
                    st.session_state.guess_count -= 1

            # Display guesses
            if not st.session_state.guesses_df.empty:
                st.write("### Your Guesses:")
                # Sort by Rank for display
                display_df = st.session_state.guesses_df.sort_values(
                    by="Rank"
                ).reset_index(drop=True)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.write("Enter a word to start guessing.")

        else:
            st.info("Click 'Start New Game' to begin.")

    else:
        st.error(
            f"Could not retrieve information for collection '{collection_name}'. Cannot start the app."
        )
else:
    st.error(
        "Failed to connect to Qdrant. Please ensure Qdrant is running and accessible."
    )
