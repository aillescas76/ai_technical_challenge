# Python Development Guidelines

These conventions keep the codebase consistent, testable, and fast to iterate on for the LLM Airline Policy App.

## Versions & Environment
- Target Python 3.11+; create a virtual environment (`python -m venv .venv`).
- Manage dependencies via `requirements.txt`; pin direct deps and keep the set small.
- Store secrets in environment variables (use `.env` locally, never commit real keys).

## Project Configuration
- `pyproject.toml` (recommended):
  - Black: line-length 88; target-version py311.
  - isort: profile "black"; known-firstparty for project modules.
  - Ruff: enable common rules; avoid long ignores; fix safe rules automatically.
  - Pytest: default test paths; disable network in tests where possible.
- Optionally enable `mypy` (incremental adoption, strict in new modules).

## Style & Structure
- Format with `black`; sort imports with `isort`; lint with `ruff`.
- Use type hints everywhere; prefer `TypedDict`/`dataclass` for structured data.
- Keep modules small and cohesive; avoid cross-layer imports.
- Public functions/classes get docstrings (one-line summary + key params/returns).
- Prefer pure functions for text processing (chunking, normalization) to simplify tests.

## Application Layout (suggested)
- `app/config.py` – environment parsing, constants.
- `app/ingest.py` – loaders, chunking, indexing CLI.
- `app/vector_store.py` – FAISS wrapper (add/search/persist).
- `app/llm.py` – embeddings + chat clients; swappable provider interface.
- `app/prompt.py` – prompt templates and assembly helpers.
- `app/schemas.py` – Pydantic models (requests/responses).
- `app/server.py` – FastAPI endpoints and startup wiring.

## Error Handling & Logging
- Never use bare `except:`; catch specific exceptions and re-raise meaningful errors.
- Use `logging` with structured fields (airline, doc, chunk_id) where helpful.
- Do not log secrets or full request bodies; redact sensitive data.
- Surface user-friendly messages (e.g., "insufficient evidence to answer").

## RAG Implementation Guidelines
- Ingestion
  - Parse Markdown (preserve headings) and PDFs (`pypdf`); extract inline links.
  - Derive metadata from paths: `airline`, `title`, optional `category`, `source_path`.
  - Select a canonical `source_url` when links exist and persist in metadata.
  - Chunk ≈800–1000 tokens with ≈150 overlap; store `data/processed.jsonl`.
- Retrieval
  - Use FAISS (cosine similarity), `top_k=3–5`; consider MMR for diversity.
  - Light re-ranking: prefer airline match and category alignment.
- Answering
  - Prompt requires citations (airline + doc title) and explicit refusal on low evidence.
  - Stream responses to reduce perceived latency; cap token limits sensibly.
- Models
  - Default: `text-embedding-3-small` for embeddings; `gpt-4o-mini` for chat.
  - Fallbacks: `all-MiniLM-L6-v2` for embeddings; smaller/faster chat models if evals permit.

## Testing
- Use `pytest`; keep tests fast and deterministic.
- Mock network/LLM calls; add stub embedding model for retrieval tests.
- Unit-test loaders, chunking, and vector search; add integration tests for `/ask`.
- Include an eval harness to compare models/params and track latency/cost.

## Performance
- Memory-map FAISS index at startup; reuse clients; batch embeddings.
- Cache frequent queries; limit `top_k`; keep prompts short; enforce timeouts.
- Use streaming responses; measure P50/P95 latencies; avoid unnecessary I/O.

## Security
- Validate and sanitize user input; set conservative CORS if serving a browser UI.
- Keep API keys in env; never in code or logs.
- Consider basic rate limiting to protect endpoints.

## Documentation & CI
- Keep `README.md` up to date; document design choices and limitations.
- Use pre-commit hooks for black/isort/ruff; run tests in CI.

