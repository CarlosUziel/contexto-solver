import time

import numpy as np
import pandas as pd
import streamlit as st

from config.logger import app_logger as logger
from config.settings import settings
from db.utils import get_qdrant_client
from game.contexto_game import ContextoGame
from solver.contexto_solver import ContextoSolver

st.set_page_config(page_title="Contexto Clone", layout="wide")
st.title("Contexto Clone")


# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all necessary session state variables."""
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
    if "autosolve_start_time" not in st.session_state:
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

    # New session state variables for "Play Real Game" mode
    if "real_game_solver_instance" not in st.session_state:
        st.session_state.real_game_solver_instance = None
    if "real_game_guesses_log" not in st.session_state:
        st.session_state.real_game_guesses_log = []  # List of (word, rank_or_status)
    if "real_game_next_suggestion" not in st.session_state:
        st.session_state.real_game_next_suggestion = None
    if "real_game_error_message" not in st.session_state:
        st.session_state.real_game_error_message = None
    if "real_game_game_won" not in st.session_state:
        st.session_state.real_game_game_won = False
    if "real_game_qdrant_client" not in st.session_state:
        st.session_state.real_game_qdrant_client = None
    # Session state for simulated game Qdrant client
    if "sim_qdrant_client" not in st.session_state:
        st.session_state.sim_qdrant_client = None


# --- Game Initialization ---
def initialize_game():
    """Initializes or re-initializes the game and solver for simulated/benchmark modes."""
    try:
        # Initialize Qdrant client for simulated/benchmark modes if not already done
        if not st.session_state.sim_qdrant_client:
            st.session_state.sim_qdrant_client = get_qdrant_client()

        if not st.session_state.sim_qdrant_client:
            st.session_state.error_message = "Failed to connect to Qdrant for simulated game. Please ensure it's running."
            st.session_state.game_initialized_successfully = False
            return

        # Instantiate the game
        st.session_state.game = ContextoGame(
            client=st.session_state.sim_qdrant_client,
            collection_name=settings.effective_collection_name,  # Use effective name
        )
        # The ContextoGame class initializes the game (e.g., selects a target word)
        # in its __init__ method. Calling a separate start_new_game() is not needed
        # and was causing an AttributeError.

        # Instantiate the solver
        st.session_state.solver = ContextoSolver(
            client=st.session_state.sim_qdrant_client,
            collection_name=settings.effective_collection_name,  # Use effective name
        )
        # Reset solver's past guesses for a new game context
        st.session_state.solver.past_guesses = []

        st.session_state.game_initialized_successfully = True
        st.session_state.error_message = None  # Clear previous errors
        logger.info(
            f"Simulated game initialized. Target: {st.session_state.game.get_target_word()}"
        )

    except Exception as e:
        logger.error(f"Error initializing game: {e}")
        st.session_state.error_message = f"Error initializing game: {str(e)}"
        st.session_state.game_initialized_successfully = False


# --- Sidebar ---
def render_sidebar():
    """Renders the sidebar navigation and returns the selected app mode."""
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Choose a mode:", ("Play Real Game", "Play Simulated Game", "Benchmark Solver")
    )


# --- "Play Simulated Game" Mode Functions ---
def render_play_game_controls_column():
    """Renders the controls in the left column for 'Play Simulated Game' mode."""
    if st.button("Start New Game", key="play_mode_start_new_game"):
        st.session_state.game = None
        st.session_state.solver = None
        st.session_state.game_initialized_successfully = False
        st.session_state.error_message = None
        st.session_state.solver_message = None
        st.session_state.show_win_message = False
        st.session_state.run_full_autosolve = False
        st.session_state.autosolve_start_time = None
        with st.spinner("Setting up new game..."):
            initialize_game()
        if st.session_state.game_initialized_successfully:
            st.success("New game started! Target word has been chosen. Good luck!")
        elif st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None
        st.rerun()

    if st.session_state.game_initialized_successfully and st.session_state.game:
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
                    st.session_state.autosolve_start_time = time.time()
                    if st.session_state.game and st.session_state.solver:
                        st.session_state.solver.past_guesses = []
                        logger.info(
                            "Populating solver with existing game guesses for "
                            "autosolve session."
                        )
                        for item in st.session_state.game.guesses:
                            st.session_state.solver.add_guess(item[0], item[1])
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

        if (
            not st.session_state.game.is_game_won()
            and not st.session_state.show_win_message
        ):
            with st.form(key="guess_form_play_mode"):
                user_guess = st.text_input(
                    "Enter your guess:", key="play_mode_user_guess"
                )
                submit_guess_button = st.form_submit_button(label="Submit Guess")

                if submit_guess_button and user_guess:
                    try:
                        rank = st.session_state.game.make_guess(user_guess)
                        if rank is not None:
                            st.session_state.solver.add_guess(user_guess, rank)
                            st.session_state.solver_message = (
                                f"🤖 You guessed '{user_guess}'. Rank: #{rank + 1}"
                            )
                        # else: word was already guessed or other non-error scenario from make_guess
                        # No specific message here, game.make_guess might provide feedback via exception or return
                    except (
                        ValueError
                    ) as ve:  # Word not in vocab or other game rule violation
                        st.session_state.solver_message = f"⚠️ {str(ve)}"
                    except Exception as e:
                        logger.error(f"Error during manual guess: {e}")
                        st.session_state.error_message = (
                            f"Error processing guess: {str(e)}"
                        )
                    st.rerun()  # Rerun to reflect the guess and message

        if (
            st.session_state.game.is_game_won()
            and not st.session_state.show_win_message
        ):
            st.balloons()
            duration_message = ""
            if st.session_state.autosolve_start_time:
                solve_duration = time.time() - st.session_state.autosolve_start_time
                duration_message = f" in {solve_duration:.2f} seconds"
                st.session_state.autosolve_start_time = None
            st.success(
                f"🎉 Congratulations! You found the word: "
                f"{st.session_state.game.get_target_word()} in "
                f"{len(st.session_state.game.guesses)} guesses{duration_message}."
            )
            st.session_state.show_win_message = True
            st.session_state.run_full_autosolve = False
            # No st.rerun() here, let the natural flow update the UI


def render_play_game_status_column():
    """Renders the game status in the right column for 'Play Simulated Game' mode."""
    if st.session_state.game_initialized_successfully and st.session_state.game:
        st.subheader("Game Status")
        if st.session_state.game.guesses:
            guesses_df = pd.DataFrame(
                st.session_state.game.guesses,
                columns=["Word", "Rank", "_Vector"],
            )
            guesses_df["Position"] = guesses_df["Rank"].apply(
                lambda x: f"#{x + 1}" if pd.notna(x) and isinstance(x, int) else "N/A"
            )
            st.dataframe(
                guesses_df[["Word", "Position"]],
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


def perform_autosolve_step_if_active():
    """Performs a single step of autosolving if active."""
    if (
        st.session_state.get("run_full_autosolve", False)
        and st.session_state.game
        and not st.session_state.game.is_game_won()
        and st.session_state.game_initialized_successfully
        and st.session_state.solver
    ):
        try:
            suggested_word = st.session_state.solver.guess_word()
            if suggested_word:
                rank = st.session_state.game.make_guess(suggested_word)
                if rank is not None:
                    st.session_state.solver_message = (
                        f"🤖 Solver guessed: {suggested_word}"
                    )
                    st.session_state.solver.add_guess(suggested_word, rank)
                else:
                    st.session_state.solver_message = (
                        f"🤖 Solver's guess '{suggested_word}' was not recognized "
                        "by the game."
                    )
                    logger.warning(
                        f"Autosolve: Solver suggested invalid word '{suggested_word}' "
                        "which make_guess returned None for."
                    )
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


def handle_play_game_mode():
    """Handles the UI and logic for the 'Play Simulated Game' mode."""
    st.header("Play Contexto")

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    col1, col2 = st.columns([1, 5])
    with col1:
        render_play_game_controls_column()
    with col2:
        render_play_game_status_column()

    perform_autosolve_step_if_active()


# --- "Benchmark Solver" Mode Functions ---
MAX_GUESSES_PER_GAME_BENCHMARK = 500


def run_benchmark_games(num_games_to_benchmark):
    """Runs the benchmark for the specified number of games."""
    st.session_state.benchmark_running = True
    st.session_state.benchmark_results = None
    st.session_state.benchmark_progress = 0.0
    st.session_state.benchmark_status_text_content = "Initializing benchmark..."

    all_individual_guesses_counts = []
    all_individual_game_times = []
    total_guesses_for_all_solved_games = 0
    total_time_for_all_solved_games = 0.0
    successful_games_count = 0

    progress_bar_placeholder = st.empty()
    status_text_placeholder = st.empty()

    for i in range(num_games_to_benchmark):
        current_game_num = i + 1
        st.session_state.benchmark_progress = i / num_games_to_benchmark
        st.session_state.benchmark_status_text_content = (
            f"Running game {current_game_num}/{num_games_to_benchmark}..."
        )
        status_text_placeholder.text(st.session_state.benchmark_status_text_content)
        progress_bar_placeholder.progress(st.session_state.benchmark_progress)

        initialize_game()
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
                f"Benchmark: Failed to initialize game {current_game_num}. "
                f"Error: {error_msg}"
            )
            all_individual_guesses_counts.append(-1)
            all_individual_game_times.append(None)
            continue

        game_instance = st.session_state.game
        solver_instance = st.session_state.solver
        solver_instance.past_guesses = []
        current_game_guesses_made = 0
        game_start_time = time.time()
        logger.info(
            f"Benchmark Game {current_game_num}: Target word is "
            f"{game_instance.get_target_word()}"
        )

        while (
            not game_instance.is_game_won()
            and current_game_guesses_made < MAX_GUESSES_PER_GAME_BENCHMARK
        ):
            word_to_try = None
            try:
                suggested_word = solver_instance.guess_word()
                if suggested_word:
                    word_to_try = suggested_word
            except Exception as e:
                logger.error(
                    f"Benchmark Game {current_game_num}: Error during "
                    f"solver_instance.guess_word(): {e}"
                )
                break

            if word_to_try:
                try:
                    rank_bench = game_instance.make_guess(word_to_try)
                    if rank_bench is not None:
                        current_game_guesses_made = len(game_instance.guesses)
                        solver_instance.add_guess(word_to_try, rank_bench)
                    else:
                        logger.warning(
                            f"Benchmark Game {current_game_num}: Solver's guess "
                            f"'{word_to_try}' was not recognized. Target: "
                            f"{game_instance.get_target_word()}"
                        )
                        break
                except (
                    Exception
                ) as e:  # Includes ValueError from make_guess if word is invalid
                    logger.error(
                        f"Benchmark Game {current_game_num}: Error during "
                        f"game_instance.make_guess('{word_to_try}'): {e}"
                    )
                    break
            else:
                logger.warning(
                    f"Benchmark Game {current_game_num}: Solver returned no suggestion."
                )
                break

        game_duration = time.time() - game_start_time
        if game_instance.is_game_won():
            num_guesses_this_game = len(game_instance.guesses)
            total_guesses_for_all_solved_games += num_guesses_this_game
            all_individual_guesses_counts.append(num_guesses_this_game)
            successful_games_count += 1
            total_time_for_all_solved_games += game_duration
            all_individual_game_times.append(game_duration)
            logger.info(
                f"Benchmark Game {current_game_num} WON in {num_guesses_this_game} "
                f"guesses, {game_duration:.2f}s. Target: "
                f"{game_instance.get_target_word()}"
            )
        else:
            final_guess_count = len(game_instance.guesses)
            if final_guess_count >= MAX_GUESSES_PER_GAME_BENCHMARK:
                all_individual_guesses_counts.append(MAX_GUESSES_PER_GAME_BENCHMARK)
                all_individual_game_times.append(game_duration)
            else:
                all_individual_guesses_counts.append(
                    -2
                )  # Solver failure or other break
                all_individual_game_times.append(None)
            logger.warning(
                f"Benchmark Game {current_game_num} NOT WON. Guesses: "
                f"{final_guess_count}, Time: {game_duration:.2f}s. Target: "
                f"{game_instance.get_target_word()}"
            )

    st.session_state.benchmark_progress = 1.0
    st.session_state.benchmark_status_text_content = "Benchmark finished!"
    status_text_placeholder.text(st.session_state.benchmark_status_text_content)
    progress_bar_placeholder.progress(st.session_state.benchmark_progress)

    median_guesses, std_guesses, median_solve_time, std_solve_time = (
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    )
    if successful_games_count > 0:
        successful_guesses_list = [
            g for g in all_individual_guesses_counts if g is not None and g > 0
        ]
        successful_times_list = [
            t
            for i, t in enumerate(all_individual_game_times)
            if t is not None
            and all_individual_guesses_counts[i] > 0
            and all_individual_guesses_counts[i] != MAX_GUESSES_PER_GAME_BENCHMARK
        ]
        if successful_guesses_list:
            median_guesses = np.median(successful_guesses_list)
            std_guesses = np.std(successful_guesses_list)
        if successful_times_list:
            median_solve_time = np.median(successful_times_list)
            std_solve_time = np.std(successful_times_list)

        avg_guesses_log = total_guesses_for_all_solved_games / successful_games_count
        avg_time_log = total_time_for_all_solved_games / successful_games_count
        logger.info(
            f"Benchmark finished. Median guesses: "
            f"{median_guesses if isinstance(median_guesses, str) else median_guesses:.2f} "
            f"± {std_guesses if isinstance(std_guesses, str) else std_guesses:.2f}, "
            f"Median time: "
            f"{median_solve_time if isinstance(median_solve_time, str) else median_solve_time:.2f}s "
            f"± {std_solve_time if isinstance(std_solve_time, str) else std_solve_time:.2f}s "
            f"over {successful_games_count}/{num_games_to_benchmark} games. "
            f"(Avg guesses: {avg_guesses_log:.2f}, Avg time: {avg_time_log:.2f}s)"
        )
    else:
        logger.warning("Benchmark finished. No games were successfully solved.")

    st.session_state.benchmark_results = {
        "median_guesses": median_guesses,
        "std_guesses": std_guesses,
        "median_solve_time": median_solve_time,
        "std_solve_time": std_solve_time,
        "successful_games": successful_games_count,
        "total_games": num_games_to_benchmark,
        "all_guesses_counts": all_individual_guesses_counts,
        "all_game_times": all_individual_game_times,
    }
    st.session_state.benchmark_running = False
    progress_bar_placeholder.empty()
    status_text_placeholder.empty()
    st.rerun()


def display_benchmark_results():
    """Displays the results of the benchmark."""
    results = st.session_state.benchmark_results
    st.subheader("Benchmark Results")
    col_metric1, col_metric2 = st.columns(2)

    with col_metric1:
        median_guesses_display = "N/A"
        if isinstance(results["median_guesses"], (float, np.floating)) and isinstance(
            results["std_guesses"], (float, np.floating)
        ):
            median_guesses_display = (
                f"{results['median_guesses']:.2f} ± {results['std_guesses']:.2f}"
            )
        elif results["median_guesses"] != "N/A":
            median_guesses_display = f"{results['median_guesses']:.2f}"
        st.metric(label="Median Guesses per Solved Game", value=median_guesses_display)

    with col_metric2:
        median_time_display = "N/A"
        if isinstance(
            results["median_solve_time"], (float, np.floating)
        ) and isinstance(results["std_solve_time"], (float, np.floating)):
            median_time_display = (
                f"{results['median_solve_time']:.2f}s ± "
                f"{results['std_solve_time']:.2f}s"
            )
        elif results["median_solve_time"] != "N/A":
            median_time_display = f"{results['median_solve_time']:.2f}s"
        st.metric(label="Median Solve Time per Solved Game", value=median_time_display)

    st.write(
        f"Successfully solved games: {results['successful_games']} out of "
        f"{results['total_games']}"
    )

    if results["all_guesses_counts"]:
        # Ensure all_game_times has the same length for DataFrame creation
        game_times_padded = results["all_game_times"][:]
        if len(game_times_padded) < len(results["all_guesses_counts"]):
            game_times_padded.extend(
                [None] * (len(results["all_guesses_counts"]) - len(game_times_padded))
            )

        df_benchmark_details = pd.DataFrame(
            {
                "Game Number": range(1, len(results["all_guesses_counts"]) + 1),
                "Guesses": results["all_guesses_counts"],
                "Solve Time (s)": [
                    f"{t:.2f}" if t is not None else "N/A" for t in game_times_padded
                ],
            }
        )
        st.write("Benchmark Details per Game (negative Guesses indicate failures):")
        st.dataframe(df_benchmark_details, use_container_width=True)

        successful_guess_counts = [
            g
            for g in results["all_guesses_counts"]
            if g > 0 and g != MAX_GUESSES_PER_GAME_BENCHMARK
        ]
        if successful_guess_counts:
            st.write(
                "Distribution of guesses for successfully solved games "
                "(excluding maxed out):"
            )
            st.bar_chart(pd.Series(successful_guess_counts).value_counts().sort_index())

        failure_types = {
            "Initialization Failure": results["all_guesses_counts"].count(-1),
            "Solver Failure (Mid-Game)": results["all_guesses_counts"].count(-2),
            f"Reached Max Guesses ({MAX_GUESSES_PER_GAME_BENCHMARK})": results[
                "all_guesses_counts"
            ].count(MAX_GUESSES_PER_GAME_BENCHMARK),
        }
        failure_df = pd.DataFrame(
            list(failure_types.items()), columns=["Failure Type", "Count"]
        )
        failure_df = failure_df[failure_df["Count"] > 0]
        if not failure_df.empty:
            st.write("Summary of Unsuccessful Games:")
            st.dataframe(failure_df, use_container_width=True)


def handle_benchmark_mode():
    """Handles the UI and logic for the 'Benchmark Solver' mode."""
    st.header("Solver Benchmark")

    num_games_input = st.number_input(
        "Number of games to benchmark:",
        min_value=1,
        value=32,
        step=1,
        key="num_benchmark_games",
        disabled=st.session_state.benchmark_running,
    )

    if st.button(
        "Start Benchmark",
        key="start_benchmark_button",
        disabled=st.session_state.benchmark_running,
    ):
        run_benchmark_games(num_games_input)  # This will eventually call st.rerun()

    if st.session_state.benchmark_results and not st.session_state.benchmark_running:
        display_benchmark_results()


# --- "Play Real Game" Mode Functions ---
def initialize_real_game_solver():
    """Initializes the solver for the 'Play Real Game' mode."""
    if not st.session_state.get("real_game_qdrant_client"):
        st.session_state.real_game_qdrant_client = get_qdrant_client()

    if st.session_state.real_game_qdrant_client:
        try:
            st.session_state.real_game_solver_instance = ContextoSolver(
                client=st.session_state.real_game_qdrant_client,
                collection_name=settings.effective_collection_name,  # Use effective name
            )
            st.session_state.real_game_error_message = None
            logger.info("Real game solver initialized.")
        except Exception as e:
            logger.error(f"Error initializing real game solver: {e}")
            st.session_state.real_game_error_message = (
                f"Could not initialize solver: {e}"
            )
            st.session_state.real_game_solver_instance = None
    else:
        st.session_state.real_game_error_message = (
            "Qdrant connection not available for real game solver."
        )
        st.session_state.real_game_solver_instance = None


def render_real_game_controls():
    """Renders controls for the 'Play Real Game' mode."""
    st.header("Play Real Contexto.me with Solver Assistance")

    if st.button("Start New Real Game Session", key="real_game_start_new"):
        initialize_real_game_solver()
        if st.session_state.real_game_solver_instance:
            st.success("New session started. Solver is ready.")
        elif st.session_state.real_game_error_message:
            st.error(st.session_state.real_game_error_message)
        st.rerun()

    if not st.session_state.real_game_solver_instance:
        st.info(
            "Click 'Start New Real Game Session' to initialize the solver "
            "and get suggestions."
        )
        return

    if st.session_state.real_game_game_won:
        st.balloons()
        st.success(
            "🎉 Congratulations! You found the word in the real Contexto.me game!"
        )
        return

    # Display current suggestion or get a new one
    if not st.session_state.real_game_next_suggestion:
        try:
            suggestion = st.session_state.real_game_solver_instance.guess_word()
            st.session_state.real_game_next_suggestion = suggestion
            if not suggestion:
                st.session_state.real_game_error_message = (
                    "Solver could not provide a new suggestion."
                )
        except Exception as e:
            st.session_state.real_game_error_message = f"Error getting suggestion: {e}"
            logger.error(f"Real Game - Error getting suggestion: {e}")

    if st.session_state.real_game_next_suggestion:
        st.subheader(
            f"Solver Suggestion: `{st.session_state.real_game_next_suggestion}`"
        )
        st.markdown(
            "Enter this word into the real [Contexto.me](https://contexto.me/) game, "
            "then input the rank you receive below, or mark the suggestion as invalid."
        )

        with st.form(key="real_game_rank_form"):
            rank_input = st.number_input(
                "Enter Rank from Contexto.me (1 for the secret word):",
                min_value=1,
                step=1,
                key="real_game_rank_input",
            )
            col_submit, col_ignore = st.columns(2)
            with col_submit:
                submit_rank_button = st.form_submit_button(
                    label="Submit Rank", use_container_width=True
                )
            with col_ignore:
                ignore_suggestion_button = st.form_submit_button(
                    label="Mark as Invalid / Ignore",
                    type="secondary",
                    use_container_width=True,
                )

            if submit_rank_button:
                word_guessed = st.session_state.real_game_next_suggestion
                rank_received = rank_input

                if word_guessed and rank_received is not None:
                    st.session_state.real_game_guesses_log.append(
                        (word_guessed, rank_received)
                    )
                    try:
                        st.session_state.real_game_solver_instance.add_guess(
                            word_guessed, rank_received
                        )
                        if rank_received == 1:
                            st.session_state.real_game_game_won = True
                        st.session_state.real_game_next_suggestion = (
                            None  # Get new suggestion
                        )
                        st.session_state.real_game_error_message = None
                    except Exception as e:
                        st.session_state.real_game_error_message = (
                            f"Error updating solver with guess: {e}"
                        )
                        logger.error(
                            f"Real Game - Error updating solver with guess: {e}"
                        )
                st.rerun()

            if ignore_suggestion_button:
                word_to_ignore = st.session_state.real_game_next_suggestion
                if word_to_ignore:
                    st.session_state.real_game_guesses_log.append(
                        (word_to_ignore, "Ignored")
                    )
                    try:
                        st.session_state.real_game_solver_instance.mark_word_as_not_allowed(
                            word_to_ignore
                        )
                        logger.info(
                            f"Real Game: User marked '{word_to_ignore}' as invalid/ignored."
                        )
                        st.session_state.real_game_next_suggestion = (
                            None  # Get new suggestion
                        )
                        st.session_state.real_game_error_message = None
                    except Exception as e:
                        st.session_state.real_game_error_message = (
                            f"Error marking word as not allowed: {e}"
                        )
                        logger.error(
                            f"Real Game - Error marking word as not allowed: {e}"
                        )
                st.rerun()

    elif not st.session_state.real_game_error_message:
        st.info("Solver is thinking of the next suggestion...")

    if st.session_state.real_game_error_message:
        st.error(st.session_state.real_game_error_message)


def render_real_game_log():
    """Renders the log of guesses for the 'Play Real Game' mode."""
    if st.session_state.real_game_guesses_log:
        st.subheader("Guess Log (Real Game)")
        guesses_df = pd.DataFrame(
            st.session_state.real_game_guesses_log, columns=["Word", "Rank"]
        )
        guesses_df["Position"] = guesses_df["Rank"].apply(
            lambda x: f"#{x}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
        )
        st.dataframe(
            guesses_df[["Word", "Position"]],
            height=300,
            use_container_width=True,
            column_config={
                "Word": st.column_config.TextColumn("Guessed Word"),
                "Position": st.column_config.TextColumn("Rank / Status"),
            },
        )
    else:
        st.write("No guesses made yet in this session.")


def handle_real_game_mode():
    """Handles the UI and logic for the 'Play Real Game' mode."""
    col1, col2 = st.columns([2, 3])  # Adjusted column ratio
    with col1:
        render_real_game_controls()
    with col2:
        render_real_game_log()


# --- Main Application Flow ---
def main():
    initialize_session_state()
    app_mode = render_sidebar()

    if app_mode == "Play Simulated Game":
        handle_play_game_mode()
    elif app_mode == "Benchmark Solver":
        handle_benchmark_mode()
    elif app_mode == "Play Real Game":
        handle_real_game_mode()

    # Clear general error message if it wasn't handled by the current mode's logic
    # and is not relevant anymore (e.g., switching modes)
    if st.session_state.error_message and app_mode != "Play Simulated Game":
        # Specific error handling for Play Simulated Game mode is within handle_play_game_mode
        # This clears errors that might persist from initialization if user switches mode
        st.session_state.error_message = None


if __name__ == "__main__":
    main()
