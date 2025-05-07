import time  # Added for timing

import numpy as np  # Added for median and std calculations
import pandas as pd
import streamlit as st

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import get_qdrant_client
from game.contexto_game import ContextoGame
from solver.contexto_solver import ContextoSolver

st.set_page_config(page_title="Contexto Clone", layout="wide")
st.title("Contexto Clone")

# Initialize session state variables
if "game" not in st.session_state:
    st.session_state.game = None
if "solver" not in st.session_state:
    st.session_state.solver = None
if "game_initialized_successfully" not in st.session_state:
    st.session_state.game_initialized_successfully = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "show_win_message" not in st.session_state:
    st.session_state.show_win_message = False
if "solver_message" not in st.session_state:
    st.session_state.solver_message = None
if "run_full_autosolve" not in st.session_state:
    st.session_state.run_full_autosolve = False
if (
    "autosolve_start_time" not in st.session_state
):  # Added for single game autosolve timing
    st.session_state.autosolve_start_time = None

# New session state variables for benchmarking
if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "benchmark_progress" not in st.session_state:
    st.session_state.benchmark_progress = 0.0
if "benchmark_status_text_content" not in st.session_state:
    st.session_state.benchmark_status_text_content = ""


def initialize_game():
    try:
        client = get_qdrant_client()
        if not client:
            st.session_state.error_message = (
                "Failed to connect to Qdrant. Please ensure it's running."
            )
            st.session_state.game_initialized_successfully = False
            return

        st.session_state.game = ContextoGame(
            collection_name=settings.glove_dataset, client=client
        )
        st.session_state.solver = ContextoSolver(
            client=client, collection_name=settings.glove_dataset
        )

        st.session_state.game_initialized_successfully = True
        st.session_state.solver_message = None
        st.session_state.run_full_autosolve = False
        st.session_state.error_message = None
        st.session_state.show_win_message = False
        st.session_state.autosolve_start_time = None  # Reset autosolve timer
        logger.info(
            f"New game started. Target: {st.session_state.game.get_target_word()}"
        )
    except Exception as e:
        logger.error(f"Error initializing game: {e}")
        st.session_state.error_message = f"Error initializing game: {str(e)}"
        st.session_state.game_initialized_successfully = False


# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a mode:", ("Play Game", "Benchmark Solver"))

if app_mode == "Play Game":
    st.header("Play Contexto")

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("Start New Game", key="play_mode_start_new_game"):
            # Reset all relevant game states
            st.session_state.game = None
            st.session_state.solver = None
            st.session_state.game_initialized_successfully = False
            st.session_state.error_message = None
            st.session_state.solver_message = None
            st.session_state.show_win_message = False
            st.session_state.run_full_autosolve = False
            st.session_state.autosolve_start_time = None  # Reset autosolve timer
            with st.spinner("Setting up new game..."):
                initialize_game()  # This function also resets run_full_autosolve and show_win_message

            if st.session_state.game_initialized_successfully:
                st.success("New game started! Target word has been chosen. Good luck!")
            elif st.session_state.error_message:  # Show error if init failed
                st.error(st.session_state.error_message)
                st.session_state.error_message = None  # Clear after showing
            st.rerun()  # Explicitly rerun after all state changes and messages

        if st.session_state.game_initialized_successfully and st.session_state.game:
            # Autosolve Controls
            if (
                not st.session_state.game.is_game_won()
                and not st.session_state.show_win_message
            ):
                if not st.session_state.get("run_full_autosolve", False):
                    if st.button(
                        "🤖 Run Full Autosolve", key="run_autosolve_button_play_mode"
                    ):
                        st.session_state.run_full_autosolve = True
                        st.session_state.solver_message = "🤖 Full autosolve started."
                        st.session_state.autosolve_start_time = (
                            time.time()
                        )  # Start timer for autosolve
                        # Initialize solver's past_guesses with current game guesses
                        if st.session_state.game and st.session_state.solver:
                            st.session_state.solver.past_guesses = []  # Clear first
                            logger.info(
                                "Populating solver with existing game guesses for autosolve session."
                            )
                            for item in st.session_state.game.guesses:
                                word_from_game = item[0]
                                actual_rank_from_game = item[1]
                                st.session_state.solver.add_guess(
                                    word_from_game, actual_rank_from_game
                                )
                        st.rerun()
                else:
                    st.info("🤖 Autosolving game...")
                    if st.button(
                        "Stop Autosolve",
                        key="stop_autosolve_button_play_mode",
                        type="secondary",
                    ):
                        st.session_state.run_full_autosolve = False
                        st.session_state.solver_message = (
                            "🤖 Full autosolve stopped by user."
                        )
                        st.rerun()

            if st.session_state.solver_message:
                st.info(st.session_state.solver_message)

            # Guess Input Form
            if (
                not st.session_state.game.is_game_won()
                and not st.session_state.show_win_message
            ):
                with st.form(key="guess_form_play_mode"):
                    guess_word_input = st.text_input(
                        "Enter your guess:",
                        key="guess_input_play_mode",
                        disabled=st.session_state.get("run_full_autosolve", False),
                    )
                    submit_guess_button = st.form_submit_button(
                        label="Submit Guess",
                        disabled=st.session_state.get("run_full_autosolve", False),
                    )

                    if submit_guess_button and guess_word_input:
                        st.session_state.error_message = None
                        try:
                            st.session_state.game.make_guess(
                                guess_word_input.strip().lower()
                            )
                            st.session_state.solver_message = None
                        except ValueError as e:
                            st.session_state.error_message = str(e)
                        st.rerun()

            if (
                st.session_state.game.is_game_won()
                and not st.session_state.show_win_message
            ):
                st.balloons()
                duration_message = ""
                if st.session_state.autosolve_start_time:
                    solve_duration = time.time() - st.session_state.autosolve_start_time
                    duration_message = f" in {solve_duration:.2f} seconds"
                    st.session_state.autosolve_start_time = None  # Reset timer

                st.success(
                    f"🎉 Congratulations! You found the word: {st.session_state.game.get_target_word()} in {len(st.session_state.game.guesses)} guesses{duration_message}."
                )
                st.session_state.show_win_message = True
                st.session_state.run_full_autosolve = False

    with col2:
        if st.session_state.game_initialized_successfully and st.session_state.game:
            st.subheader("Game Status")

            if st.session_state.game.guesses:
                # Guesses are tuples: (word, 0-indexed_rank, vector)
                guesses_df = pd.DataFrame(
                    st.session_state.game.guesses,
                    columns=[
                        "Word",
                        "Rank",
                        "_Vector",
                    ],  # Use _Vector for the unused vector column
                )
                # Create a 1-indexed "Position" for display from the 0-indexed "Rank"
                guesses_df["Position"] = guesses_df["Rank"].apply(
                    lambda x: f"#{x + 1}"
                    if pd.notna(x) and isinstance(x, int)
                    else "N/A"
                )
                st.dataframe(
                    guesses_df[
                        ["Word", "Position"]
                    ],  # Display only Word and the formatted Position
                    height=600,
                    use_container_width=True,
                    column_config={
                        "Word": st.column_config.TextColumn("Guessed Word"),
                        "Position": st.column_config.TextColumn("Rank"),
                    },
                )
            else:
                st.write("No guesses made yet.")
        elif not st.session_state.game and not st.session_state.error_message:
            st.info("Click 'Start New Game' to begin.")

    # Autosolve Logic Block (should be outside col1/col2 but within "Play Game" mode)
    if (
        st.session_state.get("run_full_autosolve", False)
        and st.session_state.game
        and not st.session_state.game.is_game_won()
        and st.session_state.game_initialized_successfully
        and st.session_state.solver
    ):
        try:
            # Solver's past_guesses are populated once when run_full_autosolve is set.
            # No need to reset and re-populate here in the loop.

            suggested_word = st.session_state.solver.guess_word()

            if suggested_word:
                # game.make_guess returns the 0-indexed rank or None
                rank = st.session_state.game.make_guess(suggested_word)

                if rank is not None:
                    st.session_state.solver_message = (
                        f"🤖 Solver guessed: {suggested_word}"
                    )
                    # Add the suggested_word (which was processed by make_guess) and its rank to the solver's history
                    st.session_state.solver.add_guess(suggested_word, rank)
                else:
                    # This case implies the solver suggested a word not in the game's list
                    st.session_state.solver_message = f"🤖 Solver's guess '{suggested_word}' was not recognized by the game."
                    logger.warning(
                        f"Autosolve: Solver suggested invalid word '{suggested_word}' which make_guess returned None for."
                    )
                    # Optionally, stop autosolve if the solver makes an invalid suggestion
                    # st.session_state.run_full_autosolve = False
            else:
                st.session_state.solver_message = (
                    "🤖 Solver has no more suggestions or failed."
                )
                st.session_state.run_full_autosolve = False
        except Exception as e:
            logger.error(f"Error during autosolve step: {e}")
            st.session_state.error_message = f"Autosolve error: {str(e)}"
            st.session_state.run_full_autosolve = False
        st.rerun()

elif app_mode == "Benchmark Solver":
    st.header("Solver Benchmark")
    MAX_GUESSES_PER_GAME = 500

    num_games_input = st.number_input(
        "Number of games to benchmark:",
        min_value=1,
        value=10,
        step=1,
        key="num_benchmark_games",
        disabled=st.session_state.benchmark_running,
    )

    if st.button(
        "Start Benchmark",
        key="start_benchmark_button",
        disabled=st.session_state.benchmark_running,
    ):
        st.session_state.benchmark_running = True
        st.session_state.benchmark_results = None
        st.session_state.benchmark_progress = 0.0
        st.session_state.benchmark_status_text_content = "Initializing benchmark..."

        all_individual_guesses_counts = []
        all_individual_game_times = []  # Added for benchmark per-game timing
        total_guesses_for_all_solved_games = 0
        total_time_for_all_solved_games = 0.0  # Added for benchmark average timing
        successful_games_count = 0

        # Placeholders for dynamic updates
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()

        for i in range(num_games_input):
            current_game_num = i + 1
            st.session_state.benchmark_progress = i / num_games_input
            st.session_state.benchmark_status_text_content = (
                f"Running game {current_game_num}/{num_games_input}..."
            )

            # Update UI before intensive work
            status_text_placeholder.text(st.session_state.benchmark_status_text_content)
            progress_bar_placeholder.progress(st.session_state.benchmark_progress)

            # Initialize a new game for the benchmark
            initialize_game()  # This also resets autosolve_start_time, which is fine.

            if (
                not st.session_state.game_initialized_successfully
                or not st.session_state.game
                or not st.session_state.solver
            ):
                error_msg = (
                    st.session_state.error_message
                    or "Unknown error during game initialization."
                )
                logger.error(
                    f"Benchmark: Failed to initialize game {current_game_num}. Error: {error_msg}"
                )
                all_individual_guesses_counts.append(
                    -1
                )  # Indicate initialization failure
                all_individual_game_times.append(None)  # No time for failed init
                continue  # Skip to next game

            game_instance = st.session_state.game
            solver_instance = st.session_state.solver
            solver_instance.past_guesses = []

            current_game_guesses_made = 0
            game_start_time = time.time()  # Start timer for this benchmark game

            logger.info(
                f"Benchmark Game {current_game_num}: Target word is {game_instance.get_target_word()}"
            )

            while (
                not game_instance.is_game_won()
                and current_game_guesses_made < MAX_GUESSES_PER_GAME
            ):
                word_to_try = None
                try:
                    # solver_instance.past_guesses is managed by adding new guesses one by one.
                    # No need to clear and repopulate from game_instance.guesses here.
                    suggested_word = solver_instance.guess_word()
                    if suggested_word:
                        word_to_try = suggested_word
                except Exception as e:
                    logger.error(
                        f"Benchmark Game {current_game_num}: Error during solver_instance.guess_word(): {e}"
                    )
                    break

                if word_to_try:
                    try:
                        # game_instance.make_guess returns the 0-indexed rank or None
                        rank_bench = game_instance.make_guess(word_to_try)

                        if rank_bench is not None:
                            current_game_guesses_made = len(game_instance.guesses)
                            # Add the word_to_try (which was processed by make_guess) and its rank to the solver's history
                            solver_instance.add_guess(word_to_try, rank_bench)
                        else:
                            # This case implies the solver suggested a word not in the game's list
                            logger.warning(
                                f"Benchmark Game {current_game_num}: Solver's guess '{word_to_try}' was not recognized by the game (make_guess returned None). Target: {game_instance.get_target_word()}"
                            )
                            # Stop this game if solver makes a bad guess, as it can't proceed.
                            # all_individual_guesses_counts.append(-2) # Already handled below if not won
                            break

                    except ValueError as e:  # Should not be reached if make_guess only returns Optional[int]
                        logger.warning(
                            f"Benchmark Game {current_game_num}: Solver suggested invalid word '{word_to_try}': {e}. Target: {game_instance.get_target_word()}"
                        )
                        break
                    except Exception as e:
                        logger.error(
                            f"Benchmark Game {current_game_num}: Error during game_instance.make_guess('{word_to_try}'): {e}"
                        )
                        break
                else:  # No word_to_try from solver
                    logger.warning(
                        f"Benchmark Game {current_game_num}: Solver returned no suggestion. Breaking game loop."
                    )
                    break  # Break from the while loop for this game

            game_duration = (
                time.time() - game_start_time
            )  # Calculate duration for this game

            if game_instance.is_game_won():
                num_guesses_this_game = len(game_instance.guesses)
                total_guesses_for_all_solved_games += num_guesses_this_game
                all_individual_guesses_counts.append(num_guesses_this_game)
                successful_games_count += 1

                total_time_for_all_solved_games += game_duration  # Add to total time
                all_individual_game_times.append(game_duration)  # Store individual time
                logger.info(
                    f"Benchmark Game {current_game_num} WON in {num_guesses_this_game} guesses and {game_duration:.2f}s. Target: {game_instance.get_target_word()}"
                )
            else:  # Game not won (max guesses or solver failure)
                final_guess_count = len(game_instance.guesses)
                if final_guess_count >= MAX_GUESSES_PER_GAME:
                    all_individual_guesses_counts.append(MAX_GUESSES_PER_GAME)
                    all_individual_game_times.append(
                        game_duration
                    )  # Store time even if maxed out but not won by target
                else:  # Solver failed mid-game or other break
                    all_individual_guesses_counts.append(-2)  # Mark as solver failure
                    all_individual_game_times.append(
                        None
                    )  # No valid solve time for solver failure

                logger.warning(
                    f"Benchmark Game {current_game_num} NOT WON. Guesses: {final_guess_count}, Time: {game_duration:.2f}s. Target: {game_instance.get_target_word()}"
                )

        st.session_state.benchmark_progress = 1.0
        st.session_state.benchmark_status_text_content = "Benchmark finished!"
        status_text_placeholder.text(st.session_state.benchmark_status_text_content)
        progress_bar_placeholder.progress(st.session_state.benchmark_progress)
        st.session_state.benchmark_running = False

        average_solve_time = 0.0  # This will be replaced by median and std
        median_guesses = "N/A"
        std_guesses = "N/A"
        median_solve_time = "N/A"
        std_solve_time = "N/A"

        if successful_games_count > 0:
            successful_guesses_list = [
                g for g in all_individual_guesses_counts if g is not None and g > 0
            ]
            successful_times_list = [
                t
                for i, t in enumerate(all_individual_game_times)
                if t is not None
                and all_individual_guesses_counts[i] > 0
                and all_individual_guesses_counts[i] != MAX_GUESSES_PER_GAME
            ]

            if successful_guesses_list:
                median_guesses = np.median(successful_guesses_list)
                std_guesses = np.std(successful_guesses_list)

            if successful_times_list:
                median_solve_time = np.median(successful_times_list)
                std_solve_time = np.std(successful_times_list)

            # Keep average for logging, but UI will use median
            average_guesses_log = (
                total_guesses_for_all_solved_games / successful_games_count
            )
            average_solve_time_log = (
                total_time_for_all_solved_games / successful_games_count
            )

            st.session_state.benchmark_results = {
                "median_guesses": median_guesses,
                "std_guesses": std_guesses,
                "median_solve_time": median_solve_time,
                "std_solve_time": std_solve_time,
                "successful_games": successful_games_count,
                "total_games": num_games_input,
                "all_guesses_counts": all_individual_guesses_counts,
                "all_game_times": all_individual_game_times,
            }
            logger.info(
                f"Benchmark finished. Median guesses: {median_guesses:.2f} ± {std_guesses:.2f}, Median time: {median_solve_time:.2f}s ± {std_solve_time:.2f}s over {successful_games_count}/{num_games_input} games. (Avg guesses: {average_guesses_log:.2f}, Avg time: {average_solve_time_log:.2f}s)"
            )
        else:
            st.session_state.benchmark_results = {
                "median_guesses": "N/A",
                "std_guesses": "N/A",
                "median_solve_time": "N/A",
                "std_solve_time": "N/A",
                "successful_games": 0,
                "total_games": num_games_input,
                "all_guesses_counts": all_individual_guesses_counts,
                "all_game_times": all_individual_game_times,
            }
            logger.warning("Benchmark finished. No games were successfully solved.")

        progress_bar_placeholder.empty()
        status_text_placeholder.empty()
        st.rerun()

    if st.session_state.benchmark_results and not st.session_state.benchmark_running:
        results = st.session_state.benchmark_results
        st.subheader("Benchmark Results")

        col_metric1, col_metric2 = st.columns(2)

        with col_metric1:
            median_guesses_display = "N/A"
            if isinstance(results["median_guesses"], float) and isinstance(
                results["std_guesses"], float
            ):
                median_guesses_display = (
                    f"{results['median_guesses']:.2f} ± {results['std_guesses']:.2f}"
                )
            elif results["median_guesses"] != "N/A":
                median_guesses_display = f"{results['median_guesses']:.2f}"

            st.metric(
                label="Median Guesses per Solved Game", value=median_guesses_display
            )

        with col_metric2:
            median_time_display = "N/A"
            if isinstance(results["median_solve_time"], float) and isinstance(
                results["std_solve_time"], float
            ):
                median_time_display = f"{results['median_solve_time']:.2f}s ± {results['std_solve_time']:.2f}s"
            elif results["median_solve_time"] != "N/A":
                median_time_display = f"{results['median_solve_time']:.2f}s"

            st.metric(
                label="Median Solve Time per Solved Game", value=median_time_display
            )

        st.write(
            f"Successfully solved games: {results['successful_games']} out of {results['total_games']}"
        )

        if results["all_guesses_counts"] and results["all_game_times"]:
            # Ensure all_game_times has the same length as all_guesses_counts, padding with None if necessary
            # This should be handled by the logic above, but as a safeguard:
            if len(results["all_game_times"]) < len(results["all_guesses_counts"]):
                results["all_game_times"].extend(
                    [None]
                    * (
                        len(results["all_guesses_counts"])
                        - len(results["all_game_times"])
                    )
                )

            df_benchmark_details = pd.DataFrame(
                {
                    "Game Number": range(1, len(results["all_guesses_counts"]) + 1),
                    "Guesses": results["all_guesses_counts"],
                    "Solve Time (s)": [
                        f"{t:.2f}" if t is not None else "N/A"
                        for t in results["all_game_times"]
                    ],
                }
            )

            st.write("Benchmark Details per Game (negative Guesses indicate failures):")
            st.dataframe(df_benchmark_details, use_container_width=True)

            successful_guess_counts = [
                g
                for g in results["all_guesses_counts"]
                if g > 0 and g != MAX_GUESSES_PER_GAME
            ]
            if successful_guess_counts:
                st.write(
                    "Distribution of guesses for successfully solved games (excluding maxed out):"
                )
                st.bar_chart(
                    pd.Series(successful_guess_counts).value_counts().sort_index()
                )

            failure_types = {
                "Initialization Failure": results["all_guesses_counts"].count(-1),
                "Solver Failure (Mid-Game)": results["all_guesses_counts"].count(-2),
                f"Reached Max Guesses ({MAX_GUESSES_PER_GAME})": results[
                    "all_guesses_counts"
                ].count(MAX_GUESSES_PER_GAME),
            }
            failure_df = pd.DataFrame(
                list(failure_types.items()), columns=["Failure Type", "Count"]
            )
            failure_df = failure_df[failure_df["Count"] > 0]
            if not failure_df.empty:
                st.write("Summary of Unsuccessful Games:")
                st.dataframe(failure_df, use_container_width=True)

    elif st.session_state.benchmark_running:
        pass

if st.session_state.error_message and app_mode != "Play Game":
    st.session_state.error_message = None
