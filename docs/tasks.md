# Project Tasks Checklist (Parallel Workflows)

Use this checklist to track progress while building the LLM Airline Policy App. Tasks are organized into parallel workflows so multiple people can work independently while still delivering a complete RAG system in this repository.

For each workflow, we list:
- Preconditions: what must exist before starting.
- Outputs: files, modules, and artifacts produced.
- Tasks: concrete, checkable steps.

---

## Workflow A – Repo, Environment & Tooling

**Preconditions**
- Git repository is cloned locally.
- Python 3.11+ installed.

**Outputs**
- `.gitignore` with entries for `data/`, `.env`, `.venv/`, caches, and build artifacts.
- `requirements.txt` with FastAPI, Uvicorn, FAISS, embeddings/LLM SDKs, PDF parsing, and tooling.
- `pyproject.toml` configured for black, isort, ruff, pytest.
- `.env.example` listing required environment variables (no secrets).
- Local virtual environment (e.g., `.venv/`) created and usable.

**Tasks**
- [x] Add `.gitignore` entries for `data/`, `.env`, `.venv/`, caches, and build artifacts.
- [x] Create `requirements.txt` with FastAPI, Uvicorn, FAISS, embeddings/LLM SDKs, PDF parsing, and tooling.
- [x] Create `pyproject.toml` (project metadata; configure black, isort, ruff, pytest).
- [x] Create `.env.example` with required variables (no secrets).
- [x] Set up local Python 3.11+ virtual environment.
- [x] Add formatter/linter config (black, isort, ruff) and pre-commit hooks.
- [x] Add GitHub Actions to run tests and lint on PRs.

This workflow can proceed in parallel with others as soon as the repo is cloned.

---

## Workflow B – Data Ingestion & Vector Store

**Preconditions**
- Workflow A: `requirements.txt` and basic environment are in place.
- `policies/` directory available with PDF and Markdown policy files.

**Outputs**
- Ingestion module: `app/ingest.py`.
- Processed documents file: `data/processed.jsonl`.
- Vector store wrapper: `app/vector_store.py`.
- Persisted FAISS index directory: `data/faiss/`.
- Unit tests for loaders, chunking, and vector store behavior (e.g., under `tests/`).
- CLI entry point: `python -m app.ingest`.

**Tasks**
- [ ] Implement Markdown loader (preserve headings, extract links).
- [ ] Implement PDF text extraction (and link annotation capture where possible).
- [ ] Derive metadata from path: `airline`, `title`, optional `category`, `source_path`.
- [ ] Extract best canonical `source_url` from document content when available.
- [ ] Chunk documents (≈800–1000 tokens, ≈150 overlap) and persist `data/processed.jsonl`.
- [ ] Implement `app/vector_store.py` (FAISS wrapper): add, search, persist/load.
- [ ] Persist index under `data/faiss/` and ensure it can be reloaded.
- [ ] Provide a CLI (`python -m app.ingest`) to (re)build processed data and index.
- [ ] Add unit tests for loaders, chunking, and index build/search/persistence.

This workflow provides the data + index foundation that the API and LLM workflows consume.

---

## Workflow C – Embeddings, LLM Client & Prompting

**Preconditions**
- Workflow A: environment and dependencies installed (including OpenAI SDK or fallback).
- Basic project structure under `app/` created (even if empty modules).

**Outputs**
- Embeddings client module: `app/llm.py` (or equivalent) exposing an embeddings interface.
- LLM chat client in the same module (or a sibling) with a clean API.
- Prompt templates module: `app/prompt.py` with RAG system and user prompts.
- Config module: `app/config.py` defining env-driven model names and keys.
- Stubbed tests for embeddings and LLM clients (network-free where possible).

**Tasks**
- [ ] Implement embeddings client (`text-embedding-3-small`, fallback to `all-MiniLM-L6-v2`).
- [ ] Implement chat client (`gpt-4o-mini` by default) with streaming support.
- [ ] Create grounded answer prompt with citation rules (airline + doc title, optional URL) and refusal behavior.
- [ ] Add configuration support for `OPENAI_API_KEY`, `EMBEDDINGS_MODEL`, `LLM_MODEL`, `VECTOR_STORE_PATH`.
- [ ] Add unit tests with stubbed embeddings/LLM to avoid network usage.

This workflow can run largely in parallel with Workflow B, coordinating only on the embedding vector size for FAISS.

---

## Workflow D – API Server & RAG Orchestration

**Preconditions**
- Workflow A: environment ready.
- Workflow B: vector store interface and index location defined (index does not need to be fully tuned yet).
- Workflow C: embeddings and LLM interfaces agreed, even if mocked initially.

**Outputs**
- FastAPI app: `app/server.py`.
- Pydantic schemas: `app/schemas.py` for request/response models.
- RAG orchestration logic inside `/ask` endpoint.
- Health endpoint `/healthz`.
- Working OpenAPI docs at `/docs`.

**Tasks**
- [ ] Scaffold FastAPI app in `app/server.py`.
- [ ] Define request/response schemas in `app/schemas.py`.
- [ ] Initialize and cache vector store at startup (load FAISS index from `data/faiss/`).
- [ ] Add `/ask` endpoint: retrieve chunks, assemble context, call LLM, return answer + citations.
- [ ] Add `/healthz` endpoint and document `/docs` (OpenAPI) for manual testing.
- [ ] Implement clear error handling (e.g., missing index, empty retrieval, model errors).

This workflow stitches together ingestion, vector store, embeddings, and LLM into a callable backend.

---

## Workflow E – UI & User Experience

**Preconditions**
- Workflow D: FastAPI app with `/ask` endpoint available (even if responses are mocked initially).

**Outputs**
- Minimal HTML/JS page served by FastAPI (e.g., via `app/server.py` or templates directory).
- UI elements: question text box, submit button, answer display area.
- Display of citations (airline + document title, optional URL/preview).

**Tasks**
- [ ] Add a minimal HTML page (or small frontend) to submit questions to `/ask`.
- [ ] Display answer returned by the API.
- [ ] Display cited sources (airline + title, and links when available).
- [ ] (Optional) Display streaming answer as it arrives from the backend.
- [ ] (Optional) Add airline filter control and top-k selector for debugging.

This workflow focuses on usability and can evolve independently once the API contract is stable.

---

## Workflow F – Evals, QA & Performance

**Preconditions**
- Workflow B: ingestion and vector store working.
- Workflow C: LLM pipeline usable (can be mocked for early evals).
- Workflow D: `/ask` endpoint returning structured answers with citations.

**Outputs**
- Evaluation dataset (e.g., `data/eval_questions.jsonl` or similar).
- Pytest-based eval harness (e.g., under `tests/test_eval_*.py`).
- JSONL eval results (e.g., `data/eval_results.jsonl`, not committed).
- Basic performance metrics (latency and cost) captured in logs or eval output.

**Tasks**
- [ ] Create an initial eval set (≥ 30 questions) spanning baggage, pets, children, pregnancy, special cases; include ambiguous airline mentions.
- [ ] Label gold answers, citations (airline + doc title), and expected refusals.
- [ ] Build a `pytest` eval harness to run retrieval + generation and record JSONL results.
- [ ] Compute retrieval metrics (Recall@k, MRR), citation correctness, groundedness, refusals.
- [ ] Track latency (P50/P95) and token/cost per request.
- [ ] Use evals to select models/params (speed/price vs. quality), then lock defaults.
- [ ] Enable response streaming to reduce perceived latency.
- [ ] Add in-memory caching for repeated queries and airline-filtered lookups.
- [ ] Tune `top_k`, chunk size/overlap, and re-ranking to hit reasonable P50/P95 latency targets.
- [ ] Add strict timeouts and fallbacks for any optional network activity.

This workflow makes the RAG system “test-like complete” with measurable quality and latency.

---

## Workflow G – Observability, Security, Documentation & Release

**Preconditions**
- Workflows A–F: core system implemented and manually testable end-to-end.

**Outputs**
- Logging configuration and usage throughout the app (no secrets).
- Input validation and basic protections.
- Updated `README.md` with all required sections.
- Final “release-ready” repo state for the technical challenge.

**Tasks**
- [ ] Add structured logging (no secrets) and meaningful error messages.
- [ ] Surface user-friendly failures (e.g., “insufficient evidence” responses).
- [ ] Add basic counters/metrics (requests, errors) for local debugging.
- [ ] Load secrets via environment; never commit API keys.
- [ ] Sanitize/validate user inputs; consider basic rate limiting.
- [ ] Update `README.md` with setup/run instructions and architecture overview.
- [ ] Document design choices and tradeoffs (RAG, models, vector store).
- [ ] Provide example Q&A for the four sample queries with citations.
- [ ] Note limitations and how external links are handled.
- [ ] Manual sanity test on key queries; verify citations and links.
- [ ] Ensure `data/` and secrets are ignored; repo is clean.
- [ ] Tag initial version and prepare review notes (or equivalent summary for reviewers).

This workflow ensures the repository is cohesive, well-documented, and ready for review as a complete RAG system.
