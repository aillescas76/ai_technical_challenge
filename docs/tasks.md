# Project Tasks Checklist

Use this checklist to track progress while building the LLM Airline Policy App. Check off items as you complete them.

## Repository & Environment
- [ ] Add `.gitignore` entries for `data/`, `.env`, `.venv/`, caches, and build artifacts.
- [ ] Create `requirements.txt` with FastAPI, Uvicorn, FAISS, embeddings/LLM SDKs, PDF parsing, and tooling.
- [ ] Create `.env.example` with required variables (no secrets).
- [ ] Set up local Python 3.11+ virtual environment.

## Data & Ingestion
- [ ] Implement Markdown loader (preserve headings, extract links).
- [ ] Implement PDF text extraction (and link annotation capture where possible).
- [ ] Derive metadata from path: `airline`, `title`, optional `category`, `source_path`.
- [ ] Extract best canonical `source_url` from document content when available.
- [ ] Chunk documents (≈800–1000 tokens, ≈150 overlap) and persist `data/processed.jsonl`.
- [ ] Provide a CLI (`python -m app.ingest`) to (re)build processed data and index.

## Vector Store
- [ ] Implement `app/vector_store.py` (FAISS wrapper): add, search, persist/load.
- [ ] Persist index under `data/faiss/` and load at server startup.
- [ ] Add unit tests for index build/search and persistence.

## Embeddings & LLM
- [ ] Implement embeddings client (`text-embedding-3-small`, fallback to `all-MiniLM-L6-v2`).
- [ ] Implement chat client (`gpt-4o-mini` by default) with streaming support.
- [ ] Create grounded answer prompt with citation and refusal rules.

## API & Server
- [ ] Scaffold FastAPI app in `app/server.py`.
- [ ] Add `/ask` endpoint: retrieve, assemble context, call LLM, return answer + citations.
- [ ] Add `/healthz` and document `/docs` (OpenAPI) for manual testing.
- [ ] Initialize and cache vector store at startup.

## UI
- [ ] Add a minimal HTML page (or small frontend) to submit questions.
- [ ] Display streaming answer, cited sources (airline + title), and links when available.
- [ ] Optional: airline filter control and top-k selector for debugging.

## Evals & QA
- [ ] Create an initial eval set (≥ 30 questions) spanning baggage, pets, children, pregnancy, special cases; include ambiguous airline mentions.
- [ ] Label gold answers, citations (airline + doc title), and expected refusals.
- [ ] Build a `pytest` eval harness to run retrieval + generation and record JSONL results.
- [ ] Compute retrieval metrics (Recall@k, MRR), citation correctness, groundedness, refusals.
- [ ] Track latency (P50/P95) and token/cost per request.
- [ ] Use evals to select models/params (speed/price vs. quality), then lock defaults.

## Performance & Latency
- [ ] Enable response streaming to reduce perceived latency.
- [ ] Add in-memory caching for repeated queries and airline-filtered lookups.
- [ ] Tune `top_k`, chunk size/overlap, and re-ranking to hit P50 ≤ 1.5s, P95 ≤ 3.0s.
- [ ] Add strict timeouts and fallbacks for any optional network activity.

## Link Handling Strategy
- [ ] MVP: Inform-and-cite only (do not fetch external URLs during QA).
- [ ] Feature flag on-demand URL fetch; add caching and ≤ 800 ms budget if enabled.
- [ ] Document legal/compliance considerations and refresh cadence.

## Observability & Errors
- [ ] Add structured logging (no secrets) and meaningful error messages.
- [ ] Surface user-friendly failures (e.g., “insufficient evidence” responses).
- [ ] Basic counters/metrics (requests, errors) for local debugging.

## Security & Privacy
- [ ] Load secrets via environment; never commit API keys.
- [ ] Sanitize/validate user inputs; consider basic rate limiting.

## Documentation
- [ ] Update `README.md` with setup/run instructions and architecture overview.
- [ ] Document design choices and tradeoffs (RAG, models, vector store).
- [ ] Provide example Q&A for the four sample queries with citations.
- [ ] Note limitations and how external links are handled.

## CI / Tooling (Optional but Recommended)
- [ ] Add formatter/linter config (black, isort, ruff) and pre-commit hooks.
- [ ] Add GitHub Actions to run tests and lint on PRs.

## Release Checklist
- [ ] Manual sanity test on key queries; verify citations and links.
- [ ] Ensure `data/` and secrets are ignored; repo is clean.
- [ ] Tag initial version and prepare review notes.

