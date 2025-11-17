## LLM Airline Policy App

### Project Overview

This project implements a small, retrieval‑augmented generation (RAG) application that answers user questions about airline policies (Delta, United, American Airlines) using an LLM and a vector database. Policy documents in `policies/` (Markdown and PDF) are ingested, chunked, embedded, and indexed for similarity search. The API assembles relevant context and asks the LLM to produce concise, grounded answers with citations to the source documents.

### Running Everything with Docker

All workflows (ingestion and the future API) are executed via Docker so you never have to install Python dependencies directly on your host.

1. Copy `.env.example` to `.env` and fill in the required variables (e.g., `OPENAI_API_KEY`, `EMBEDDINGS_MODEL`, `LLM_MODEL`). Docker Compose automatically loads this file.
2. Build the shared image used by every service:
   ```bash
   docker compose build
   ```
3. Run ingestion inside a disposable container whenever policies change:
   ```bash
   docker compose run --rm ingest
   ```
   This performs the entire pipeline inside Docker: load policy files, write `data/processed.jsonl`, and persist the FAISS artifacts under `data/faiss/`.
4. Start the FastAPI backend **and** the lightweight front-end UI with live reload:
   ```bash
   docker compose up app
   ```
   - The `app` service serves both the API (`/ask`, `/ask/stream`, `/healthz`) and the streaming UI hosted at `http://localhost:8000/`.
   - Once the container is running, open `http://localhost:8000` in your browser to use the front-end or `http://localhost:8000/docs` for the interactive API docs.
   - The container mounts the repo so code edits (Python or `app/templates/index.html`) are reflected immediately.
5. Tear everything down when finished:
   ```bash
   docker compose down
   ```

No host-side `pip install` steps are required; the Docker image contains all runtime dependencies.

### Documentation

- Full challenge description (original README.md file): [docs/challenge.md](docs/challenge.md)
- Analysis and proposed solution: [docs/analysis.md](docs/analysis.md)
