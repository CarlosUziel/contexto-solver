import pandas as pd
import streamlit as st

from config.logger import app_logger as logger
from config.settings import settings
from game.contexto_game import ContextoGame

st.set_page_config(page_title="Contexto Clone", layout="wide")
st.title("Contexto Clone")

if "game" not in st.session_state:
    st.session_state.game = None
if "game_initialized_successfully" not in st.session_state:
    st.session_state.game_initialized_successfully = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "show_win_message" not in st.session_state:
    st.session_state.show_win_message = False


def initialize_game():
    try:
        with st.spinner("Setting up new game..."):
            st.session_state.game = ContextoGame(collection_name=settings.glove_dataset)
            st.session_state.game_initialized_successfully = True
            st.session_state.error_message = None
            logger.info(
                f"New game started successfully. Target: {st.session_state.game.get_target_word()}"
            )
            st.success("New game started! Target word has been chosen. Good luck!")
    except (ConnectionError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to initialize game: {e}", exc_info=True)
        st.session_state.game_initialized_successfully = False
        st.session_state.error_message = (
            f"Failed to initialize the game: {e}. "
            "Please ensure Qdrant is running and the collection is populated."
        )
        st.error(st.session_state.error_message)
    finally:
        st.rerun()


col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Start New Game"):
        st.session_state.game = None
        st.session_state.game_initialized_successfully = False
        st.session_state.error_message = None
        initialize_game()

if (
    st.session_state.error_message
    and not st.session_state.game_initialized_successfully
):
    st.error(st.session_state.error_message)

if st.session_state.game_initialized_successfully and st.session_state.game:
    game: ContextoGame = st.session_state.game

    if st.session_state.show_win_message:
        st.balloons()
        st.success(
            f"Congratulations! You found the secret word '{game.get_target_word()}' "
            f"in {len(game.get_guesses())} guesses!"
        )
        st.session_state.show_win_message = False

    st.write(
        f"Guess the secret word! The word is from the '{game.collection_name}' dataset "
        f"which has {game.total_points} words in total."
    )

    if not game.is_game_won():  # Only show the guess form if the game is not yet won
        with st.form(key="guess_form"):
            guess_word_input = st.text_input(
                "Enter your guess:", key="guess_input", value=""
            )
            submit_button = st.form_submit_button(label="Guess")

        if submit_button and guess_word_input:
            normalized_guess = guess_word_input.strip()
            if normalized_guess:
                rank = game.make_guess(normalized_guess)
                if rank is not None:
                    if game.is_game_won():  # Check if the guess resulted in a win
                        st.session_state.show_win_message = True
                    st.rerun()  # Rerun to update the UI (e.g., hide form if won, show new guess)
                else:
                    st.warning(
                        f"The word '{normalized_guess}' was not found in the dataset. Try again."
                    )
            else:
                st.warning("Please enter a word.")

    guesses = game.get_guesses()
    if guesses:
        st.write("### Your Guesses:")
        guesses_df_data = [
            {"Guess #": i + 1, "Word": word, "Rank": rank}
            for i, (word, rank, _) in enumerate(guesses)
        ]
        guesses_df = pd.DataFrame(guesses_df_data)
        st.dataframe(guesses_df, use_container_width=True, hide_index=True)
    elif not st.session_state.game.is_game_won():
        st.write("Enter a word to start guessing.")

elif not st.session_state.game and not st.session_state.error_message:
    st.info("Click 'Start New Game' to begin.")
