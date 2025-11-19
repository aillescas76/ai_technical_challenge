# Architecture Overview

This document outlines the high-level architecture and design principles of the Airline Policy RAG application.

## System Components

The application is structured as a modular Python package (`app`), leveraging **FastAPI** for the backend and **Docker** for containerized deployment.

### 1. Core Domain & Configuration (`app/core/`)
*   **`config.py`**: Centralized configuration using `pydantic` settings management (conceptually, though currently using `os.getenv`). Loads environment variables for LLM keys, paths, and tuning parameters.
*   **`airlines.py`**: Domain logic for normalizing airline names (e.g., mapping "AA", "American" -> "American Airlines") to ensure consistent filtering and citation.
*   **`telemetry.py`**: Wrapper for **LangFuse** integration, handling distributed tracing and observability without tight coupling to the business logic.

### 2. Infrastructure Components (`app/components/`)
*   **`llm.py`**: An abstraction layer over **LiteLLM**, providing a unified interface for chat completions and embeddings. This allows seamless switching between OpenAI, Anthropic, or local models.
*   **`vector_store.py`**: A wrapper around **FAISS** (Facebook AI Similarity Search). It handles:
    *   Adding embeddings with metadata.
    *   Persisting the index to disk (`data/faiss/`).
    *   Performing similarity searches (`search_by_embedding`).

### 3. Service Layer (`app/services/`)
*   **`ingest.py`**: The offline processing pipeline.
    *   **Load**: Reads Markdown and PDF files from `policies/`.
    *   **Chunk**: splits text into overlapping token windows (default 900 tokens, 150 overlap) using `tiktoken`.
    *   **Embed**: Generates vector embeddings for each chunk.
    *   **Index**: Saves the vectors and metadata to the vector store.
*   **`rag.py`**: The runtime "engine" for Retrieval-Augmented Generation.
    *   **Retrieve**: converts a user question to an embedding and queries the `VectorStore`.
    *   **Generate**: Constructs a prompt with retrieved context and sends it to the LLM.
    *   **Stream**: Handles streaming responses for real-time UI updates.
    *   **Cache**: Implements a short-lived LRU cache for frequent queries.
*   **`prompt.py`**: Manages prompt templates and context formatting, ensuring the LLM receives instructions to cite sources strictly.

### 4. API Layer (`app/api/`)
*   **`server.py`**: The **FastAPI** application entry point.
    *   Defines routes: `/ask`, `/ask/stream`, `/healthz`.
    *   Manages the application lifespan (loading the vector store on startup).
    *   Serves the static/template UI.
*   **`schemas.py`**: **Pydantic** models defining the API contract (request/response shapes) for validation and documentation.

### 5. Evaluation (`app/evaluation/`)
*   **`eval.py`**: A standalone harness for measuring RAG performance.
    *   Loads a "gold standard" dataset (`docs/evals/questions.jsonl`).
    *   Runs queries against the RAG engine.
    *   Computes metrics: Recall@K, MRR (Mean Reciprocal Rank), Citation Precision/Recall.
    *   Logs results to LangFuse for historical tracking.

## Data Flow

1.  **Ingestion (Offline)**:
    `PDF/MD files` -> `Load` -> `Chunk` -> `Embed (OpenAI)` -> `FAISS Index`

2.  **Query (Runtime)**:
    `User Question` -> `Embed` -> `FAISS Search (Top-K)` -> `Context Assembly` -> `LLM (GPT-4o)` -> `Answer with Citations`

## Deployment

The system is containerized using **Docker Compose**.
*   **`app` service**: Runs the FastAPI server (Uvicorn).
*   **`ingest` service**: Ephemeral container for rebuilding the index.

## Key Design Decisions

*   **Separation of Concerns**: Infrastructure (LLM, Vector Store) is isolated from business logic (RAG, Ingest), making it easier to swap backends.
*   **Stateless API**: The API server is stateless (except for the read-only memory-mapped FAISS index), allowing for horizontal scaling if needed.
*   **Groundedness**: The system prompt enforces strict adherence to provided context, reducing hallucinations.
*   **Observability**: Telemetry is a first-class citizen, enabling production monitoring of cost, latency, and quality.
