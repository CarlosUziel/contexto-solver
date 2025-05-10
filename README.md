<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="docs/contexto_gameplay.gif" alt="App Screenshot">

  <h1 align="center">Contexto Solver</h1>
  <h4 align="center">A Python application that simulates and solves the Contexto game.</h4>

  <!-- BADGES -->
  <p align="center">
    <a href="https://github.com/CarlosUziel/contexto-solver/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License Badge"></a>
    <a href="https://github.com/CarlosUziel/contexto-solver/stargazers"><img src="https://img.shields.io/github/stars/CarlosUziel/contexto-solver.svg?style=social" alt="Stars Badge"></a>
  </p>
  
  <p align="center">
    <a href="#-motivation">Motivation</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-contact">Contact</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>📋 Table of Contents</summary>
  <ol>
    <li><a href="#-motivation">🧠 Motivation</a></li>
    <li>
      <a href="#-getting-started">🚀 Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#-algorithm-explanation">💡 Algorithm Explanation</a></li>
    <li><a href="#-configuration-options">⚙️ Configuration Options</a></li>
    <li><a href="#-file-descriptions">📂 File Descriptions</a></li>
    <li><a href="#-license">📄 License</a></li>
    <li><a href="#-additional-notes">📝 Additional Notes</a></li>
    <li><a href="#-contact">👤 Contact</a></li>
    <li><a href="#-acknowledgments">🙏 Acknowledgments</a></li>
  </ol>
</details>

## 🧠 Motivation

[Contexto](https://contexto.me) is a word puzzle game where the primary objective is to discover a secret word. Players make guesses, and after each attempt, the game reveals the guessed word's rank. This rank indicates its position in a list sorted by similarity to the secret word—the lower the number, the closer the guess. The ultimate goal is to identify the secret word using the fewest number of attempts.

The game's ranking system is powered by word embeddings, specifically GloVe (Global Vectors for Word Representation) embeddings. These embeddings capture the semantic relationships between words, meaning that words with similar meanings are located closer to each other in a high-dimensional vector space. The rank provided by Contexto reflects this semantic proximity: a guess semantically closer to the target word will receive a lower rank.

This repository offers a Python-based solution that simulates the Contexto game and includes an automated solver. The user interface is a Streamlit web application, allowing users to play the game manually, observe the solver in action, or run benchmarks. This application connects to a Python backend that manages the game logic and implements the solving algorithm. At the core of this system is [Qdrant](https://qdrant.tech/), a vector database used to store, process, and efficiently query the GloVe word embeddings. Qdrant's capabilities, particularly its discovery API, are fundamental to the solver's strategy for navigating the embedding space and pinpointing the secret word.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Docker**: To run the Qdrant vector database. [Installation Guide](https://docs.docker.com/get-docker/)
*   **uv**: A fast Python package installer and resolver. [Installation Guide](https://github.com/astral-sh/uv)

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/CarlosUziel/contexto-solver.git
    cd contexto-solver
    ```

2.  Install project dependencies using uv:
    ```bash
    uv sync
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Usage

1.  **Set up Environment Variables**:
    Copy the distributed environment file `.env.dist` to `.env` and modify it with your Qdrant settings if necessary (default values should work for a local setup).
    ```bash
    cp .env.dist .env
    ```

2.  **Start Qdrant Docker Instance**:
    Use the provided `docker-compose.yml` to start the Qdrant service.
    ```bash
    docker-compose up -d
    ```

3.  **Populate the Database**:
    Run the setup script to populate the Qdrant database with the word embeddings. Make sure your `PYTHONPATH` includes the `src` directory.
    ```bash
    export PYTHONPATH=./src:$PYTHONPATH
    python src/db/setup_qdrant.py
    ```

4.  **Start the Streamlit App**:
    Launch the Streamlit application.
    ```bash
    streamlit run src/app/contexto_app.py
    ```

The Streamlit application provides the following options:
*   **Normal Game**: Play a simulated Contexto game yourself.
*   **Autosolve**: Watch the solver automatically find the secret word.
*   **Benchmarks**: Run performance tests on the solver.

<br />
<div align="center">
  <img src="docs/app_screenshot.png" alt="App Screenshot" width="700">
</div>
<br />

<p align="right">(<a href="#top">back to top</a>)</p>

## ⚙️ Configuration Options

The application's behavior can be customized through environment variables defined in the `.env` file. Below is a description of these options:

### Qdrant Options

*   `QDRANT_HOST`: Hostname or IP address of the Qdrant server.
    *   Default: `localhost`
*   `QDRANT_GRPC_HOST_PORT`: gRPC port exposed by the Qdrant host for client connections.
    *   Default: `6333`
*   `QDRANT_HTTP_HOST_PORT`: HTTP port exposed by the Qdrant host.
    *   Default: `6334`
*   `QDRANT_GRPC_CONTAINER_PORT`: Internal gRPC port used by the Qdrant container.
    *   Default: `6333`
*   `QDRANT_HTTP_CONTAINER_PORT`: Internal HTTP port used by the Qdrant container.
    *   Default: `6334`
*   `QDRANT_API_KEY`: API key for Qdrant Cloud (optional).
    *   Default: `None`
*   `QDRANT_LOG_LEVEL`: Logging level for the Qdrant service (e.g., 'debug', 'info', 'warning', 'error').
    *   Default: `info`
*   `QDRANT_UUID_NAMESPACE`: Namespace for generating Qdrant point UUIDs. Provide a valid UUID.
    *   Default: `6ba7b810-9dad-11d1-80b4-00c04fd430c8` (corresponds to `uuid.NAMESPACE_DNS`)

### GloVe Dataset

*   `GLOVE_DATASET`: Specifies the GloVe dataset to use for word embeddings. The choice of dataset affects the vocabulary size and the dimensionality of word vectors, which can influence solver performance and resource usage.
    *   Default: `glove.6B.100d`
    *   Available Options:
        *   `glove.6B.50d`
        *   `glove.6B.100d`
        *   `glove.6B.200d`
        *   `glove.6B.300d`
        *   `glove.twitter.27B.25d`
        *   `glove.twitter.27B.50d`
        *   `glove.twitter.27B.100d`
        *   `glove.twitter.27B.200d`
        *   `glove.42B.300d`
        *   `glove.840B.300d`

### Solver Algorithm Parameters

*   `QDRANT_HNSW_EF`: The 'ef' (size of the dynamic list for HNSW) parameter for Qdrant search. This affects search speed and accuracy. Higher values can lead to more accurate searches but may increase latency.
    *   Default: `64`

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- FILE DESCRIPTIONS -->
## 📂 File Descriptions

*   `src/app/contexto_app.py`: Creates the Streamlit web application interface for playing the Contexto game, watching the solver, and running benchmarks.
*   `src/config/logger.py`: Sets up a configured logger for the application, outputting to both the console and a log file.
*   `src/config/settings.py`: Defines and manages application settings using Pydantic, allowing configuration through environment variables or a `.env` file.
*   `src/db/setup_qdrant.py`: Handles the download of GloVe embeddings and their setup in the Qdrant vector database.
*   `src/db/utils.py`: Provides utility functions for interacting with the Qdrant database, such as connecting, fetching data, and performing searches.
*   `src/game/contexto_game.py`: Defines the `ContextoGame` class, which manages the state and logic of a single Contexto game instance.
*   `src/solver/contexto_solver.py`: Contains the `ContextoSolver` class, which implements the logic for automatically solving the Contexto game.

<p align="right">(<a href="#top">back to top</a>)</p>

## 💡 Algorithm Explanation

The Contexto solver employs a multi-stage strategy to pinpoint the secret word. This process heavily relies on Qdrant's vector search capabilities, especially its [*Discovery API*](https://qdrant.tech/documentation/concepts/explore/#discovery-api), to navigate the word-embedding space efficiently. Let $V$ represent the vector embedding of a word.

Here's a breakdown of the solver's process:

1.  **Initial Guess: $G_0$**
    *   If no guesses have been made (i.e., the set of past guesses $\mathcal{G}_{past}$ is empty), the solver selects a word $w_0$ uniformly at random from the entire collection. Its embedding is $V_0$. This serves as the initial anchor point.

2.  **Iterative Refinement via Discovery Search (for guess $G_i, i \ge 1$)**
    *   Once at least one guess $(w_{prev}, r_{prev}, V_{prev})$ exists, the solver primarily uses Qdrant's Discovery API. This requires constructing:
        *   **Context Pairs ($\mathcal{C}$)**: A list of positive and negative example embeddings to guide the search.
        *   **Target Vector ($V_{target}$)**: An optional specific point in the embedding space to steer the search towards.
        *   **Set of Past Guesses ($\mathcal{G}_{past}$)**: To exclude already tried words.

    *   **Context Pair Formation**:
        *   **For the first guess ($G_0$ with embedding $V_0$)**: A single context pair is formed. The positive example is $V_0$, and the negative example is $-V_0$ (the negation of the first guess's embedding). This initial pair helps establish a basic direction.
        *   **For subsequent guesses ($G_i$ with embedding $V_i$, where $i \ge 1$)**: Let $(w_{best}, r_{best}, V_{best})$ be the current best guess (lowest rank) from $\mathcal{G}_{past}$.
            *   If $w_i$ is better than $w_{best}$ (i.e., $r_i < r_{best}$), the new context pair is (positive: $V_i$, negative: $V_{best}$). $w_i$ becomes the new $w_{best}$.
            *   If $w_i$ is not better than $w_{best}$ (i.e., $r_i \ge r_{best}$), the new context pair is (positive: $V_{best}$, negative: $V_i$).
        *   Each new guess $G_i$ results in one additional context pair being added to $\mathcal{C}$.

    *   **Target Vector ($V_{target}$) Determination**:
        *   The solver maintains a list of embeddings, $\mathcal{V}_{positive\_centroid}$, which includes the embedding of the first guess and any subsequent guess that became the new best guess.
            *   **For the second guess ($G_1$, when $|\mathcal{G}_{past}|=1$)**: The `target` vector is `None`. The Discovery API search relies solely on the initial context pair.
            *   **For guesses $G_i$ where $i \ge 2$ (when $|\mathcal{G}_{past}| > 1$)**: The `target` vector $V_{target}$ is the centroid (mean) of all embeddings in $\mathcal{V}_{positive\_centroid}$.
            *   In all discovery searches (i.e., for any guess $G_i$ where $i \ge 1$), the `limit` for search results is 1. This focuses the search on the single best candidate based on the current context and target vector (if present).

    *   **Executing Discovery Search**:
        *   The solver calls Qdrant's `discover` method with the accumulated $\mathcal{C}$, the determined $V_{target}$ (if any), a filter to exclude words in $\mathcal{G}_{past}$, and other search parameters like `hnsw_ef`.

> **Qdrant Discovery API Behavior Notes:**
> 
> The Discovery API is a powerful feature for exploring the vector space. It allows searching for points (word embeddings in this case) that are "semantically similar" to a set of positive example vectors and "dissimilar" to a set of negative example vectors. This similarity/dissimilarity is typically determined by vector distances (e.g., cosine similarity or dot product) in the embedding space.
>   -   When a `target` vector is provided, the search is not only guided by the positive/negative context pairs but is also biased towards this specific `target` point. This dual influence helps to steer the exploration towards a region of interest that the solver has identified as promising (e.g., the centroid of previously successful guesses), effectively refining the search within a known good area. The API attempts to find points that are close to the positive examples, far from the negative examples, and also close to the target vector.
>   -   If no `target` is provided (i.e., `target` is `None`), the API focuses solely on optimizing the search based on the given positive and negative context pairs. It seeks items that best satisfy these relative relationships—close to positives, far from negatives—without a specific directional anchor. This mode is particularly useful in the early stages of the solving process, such as for the second guess, to encourage broader exploration and discover new potential areas of the embedding space.
>   -   The effectiveness of the Discovery API can be influenced by the quality and quantity of the context pairs provided. More pairs can offer finer-grained control over the search direction, helping Qdrant better understand the desired semantic region.
>
> <p align="center">
>  <img src="docs/qdrant_discovery_api.png" alt="Qdrant Discovery API"><br>
>  <em>Image extracted from the <a href="https://qdrant.tech/documentation/concepts/explore/#discovery-api">Qdrant Discovery API documentation</a>.</em>
> </p>

### Failure Condition

*   If the initial random guess cannot be fetched (e.g., collection issue), a `SolverUnableToGuessError` is raised.
*   If the Qdrant Discovery Search, after processing its results, does not yield any new candidate word (i.e., all candidates returned by Qdrant are already in $\mathcal{G}_{past}$ or no candidates are returned), the solver raises a `SolverUnableToGuessError`. This indicates a critical failure in the core guessing strategy, as the solver relies on discovery search to make progress.
*   If the Discovery Search itself encounters an operational error (e.g., an issue with the Qdrant service), a `SolverUnableToGuessError` is raised, wrapping the original exception.

This iterative cycle of guessing, receiving rank feedback, updating context ($\mathcal{C}, V_{target}, \mathcal{V}_{positive\_centroid}$), and strategically querying the vector database allows the solver to progressively narrow the search space towards the secret word.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ADDITIONAL NOTES -->
## 📝 Additional Notes

This codebase is linted and formatted using [Ruff](https://github.com/astral-sh/ruff). The following command is run as a pre-commit hook to ensure code quality:

```bash
ruff check . --fix && ruff format . && ruff check --fix --select I
```

To enable pre-commit hooks, run the following command within the project directory and the *uv* environment:

```bash
pre-commit install
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGEMENTS -->
## 🙏 Acknowledgments

This repository proposes a solution to the [Contexto Challenge](https://gist.github.com/generall/98e18d5afae16bf444eff05c9fc7b74d), which relies on the [Qdrant](https://qdrant.tech/) vector database for efficient vector storage and retrieval. [This other solution](https://github.com/qdrant/contexto) was used as a reference to better understand the Contexto game mechanics and potential strategies.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## 👤 Contact

<div align="center">
  <a href="https://github.com/CarlosUziel"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://scholar.google.co.uk/citations?user=tEz_OeIAAAAJ&hl"><img src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar"></a>
  <a href="https://www.linkedin.com/in/carlosuziel"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://perez-malla.com/"><img src="https://img.shields.io/badge/Homepage-blue?style=for-the-badge&logo=home&logoColor=white" alt="Homepage"></a>
</div>

<p align="right">(<a href="#top">back to top</a>)</p>
