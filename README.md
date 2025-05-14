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
*   **Play Real Game**: Use the solver to assist you in playing the official Contexto.me game.
*   **Play Simulated Game**: Play a simulated Contexto game yourself or watch the solver automatically find the secret word.
*   **Benchmark Solver**: Run performance tests on the solver.

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
*   `TOP_N_POSITIVE_EMBEDDINGS`: The number of top-ranked positive embeddings to consider for calculating the centroid vector. This centroid is used as a target in discovery searches.
    *   Default: `3`
*   `RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK`: The number of random words to sample when fetching an initial word or when discovery search fails. The solver selects the most frequent word from this sample.
    *   Default: `256`
*   `DISCOVERY_SEARCH_LIMIT`: The maximum number of results to retrieve from a Qdrant discovery search. The solver then selects the most frequent word from these results.
    *   Default: `4`
*   `USE_ONLY_NOUNS`: If set to `true`, the application will use a Qdrant collection containing only nouns. The collection name will be the `GLOVE_DATASET` name suffixed with `_nouns` (e.g., `glove.6B.100d_nouns`). This can be useful for a more targeted search if the secret word is known or suspected to be a noun.
    *   Default: `false`

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

The Contexto solver employs a multi-stage strategy to pinpoint the secret word. This process heavily relies on Qdrant's vector search capabilities, especially its [*Discovery API*](https://qdrant.tech/documentation/concepts/explore/#discovery-api), to navigate the word-embedding space efficiently. Word frequency, obtained using the `wordfreq` library, plays a crucial role in prioritizing candidate words at various stages. Let $V$ represent the vector embedding of a word.

Here's a breakdown of the solver's process:

1.  **Initial Guess: $G_0$**
    *   If no guesses have been made (i.e., the set of past guesses $\mathcal{G}_{past}$ is empty), the solver fetches a batch of `RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK` random words from the collection. It then selects the word with the highest frequency in the English language (using the `wordfreq` library) as the initial guess $w_0$. Its embedding is $V_0$. This serves as the initial anchor point.
    *   If this process fails to yield a word (e.g., all sampled words have zero frequency or are invalid), it attempts to fetch a single random point as a fallback.

2.  **Iterative Refinement via Discovery Search (for guess $G_i, i \ge 1$)**
    *   Once at least one guess $(w_{prev}, r_{prev}, V_{prev})$ exists, the solver primarily uses Qdrant's Discovery API. This requires constructing:
        *   **Context Pairs**: For each past guess $(w_j, r_j, V_j)$, a context pair is formed. If $w_j$ improved upon the previous best guess (i.e., $r_j < r_{best\_prev}$), then $V_j$ becomes a positive example and $V_{best\_prev}$ becomes a negative example. Otherwise, $V_{best\_prev}$ remains positive and $V_j$ becomes negative.
        *   **Target Vector (Centroid)**: The solver maintains a list of the top `TOP_N_POSITIVE_EMBEDDINGS` best (lowest rank) positive embeddings encountered so far. A centroid (average vector) is calculated from these embeddings. This centroid serves as the `target` for the discovery search, guiding it towards promising regions of the embedding space.
    *   The Discovery API is then queried with these context pairs and the target centroid. The search is limited to `DISCOVERY_SEARCH_LIMIT` results.
    *   From the returned results, the solver selects the word with the highest frequency in English. This word becomes the next guess $w_i$. This frequency check ensures that more common and relevant words are prioritized among the semantically similar candidates.

3.  **Fallback Strategy**
    *   If the Discovery Search (Step 2) does not return any usable results (e.g., all candidates are already guessed or have zero frequency), the solver falls back to the random word selection strategy described in Step 1 (fetching `RANDOM_SAMPLE_SIZE_FOR_FREQ_CHECK` words and selecting the most frequent one).

> [!NOTE]
> **More about Qdrant Discovery API and Word Frequency:**
>
> The Discovery API is a powerful feature for exploring the vector space. It allows searching for points (word embeddings in this case) that are "semantically similar" to a set of positive example vectors and "dissimilar" to a set of negative example vectors. This similarity/dissimilarity is typically determined by vector distances (e.g., cosine similarity or dot product) in the embedding space.
>   -   When a `target` vector is provided, the search is not only guided by the positive/negative context pairs but is also biased towards this specific `target` point. This dual influence helps to steer the exploration towards a region of interest that the solver has identified as promising (e.g., the centroid of previously successful guesses), effectively refining the search within a known good area. The API attempts to find points that are close to the positive examples, far from the negative examples, and also close to the target vector.
>   -   If no `target` is provided (i.e., `target` is `None`), the API focuses solely on optimizing the search based on the given positive and negative context pairs. It seeks items that best satisfy these relative relationships—close to positives, far from negatives—without a specific directional anchor. This mode is particularly useful in the early stages of the solving process, such as for the second guess, to encourage broader exploration and discover new potential areas of the embedding space.
>   -   The effectiveness of the Discovery API can be influenced by the quality and quantity of the context pairs provided. More pairs can offer finer-grained control over the search direction, helping Qdrant better understand the desired semantic region.
>   -   **Word Frequency Prioritization**: After the Discovery API returns a list of candidate words, the solver uses the `wordfreq` library to determine the frequency of each candidate in the English language. The candidate with the highest frequency is chosen as the next guess. This step is crucial because vector similarity alone might suggest obscure or highly specific words. By prioritizing more common words, the solver aims to make guesses that are more likely to be the target word in a general-knowledge game like Contexto. This heuristic significantly improves the solver's practical performance and relevance of suggestions.
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
