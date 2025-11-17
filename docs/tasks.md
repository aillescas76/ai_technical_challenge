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
- [x] Create feature branch `feature/data-ingestion` for this workflow.
- [x] Implement Markdown loader (preserve headings, extract links).
- [x] Implement PDF text extraction (and link annotation capture where possible).
- [x] Derive metadata from path: `airline`, `title`, optional `category`, `source_path`.
- [x] Extract best canonical `source_url` from document content when available.
- [x] Chunk documents (≈800–1000 tokens, ≈150 overlap) and persist `data/processed.jsonl`.
- [x] Implement `app/vector_store.py` (FAISS wrapper): add, search, persist/load.
- [x] Persist index under `data/faiss/` and ensure it can be reloaded.
- [x] Provide a CLI (`python -m app.ingest`) to (re)build processed data and index.
- [x] Add unit tests for loaders, chunking, and index build/search/persistence.

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
- [x] Implement embeddings client via LiteLLM (`text-embedding-3-small`, fallback to `all-MiniLM-L6-v2`).
- [x] Implement chat client via LiteLLM (`gpt-4o-mini` by default) with streaming support.
- [x] Create grounded answer prompt with citation rules (airline + doc title, optional URL) and refusal behavior. (Implemented `app/prompt.py` with `build_grounded_answer_messages` and `ContextChunk` helpers.)
- [x] Add configuration support for `OPENAI_API_KEY`, `EMBEDDINGS_MODEL`, `LLM_MODEL`, `VECTOR_STORE_PATH`.
- [x] Add unit tests with stubbed embeddings/LLM to avoid network usage. (See `tests/test_llm_prompt.py` for LiteLLM stubs and prompt assertions.)

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
- [x] Scaffold FastAPI app in `app/server.py`.
- [x] Define request/response schemas in `app/schemas.py`.
- [x] Initialize and cache vector store at startup (load FAISS index from `data/faiss/`).
- [x] Add `/ask` endpoint: retrieve chunks, assemble context, call LLM, return answer + citations.
- [x] Add `/healthz` endpoint and document `/docs` (OpenAPI) for manual testing.
- [x] Implement clear error handling (e.g., missing index, empty retrieval, model errors).

**Notes**
- `app/server.py` loads the FAISS index at startup; if `data/faiss/` is missing the `/ask` route returns `503` instructing operators to run `python -m app.ingest`.
- `/ask` now requires `question`, accepts optional `airline` and `top_k<=8`, and returns `AskResponse` with structured citations (airline, title, snippet, score).
- `/healthz` reports `status` (`ok`, `index_missing`, `error`) plus the vector count so later workflows can confirm readiness.

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
- [ ] Display streaming answer as it arrives from the backend.
- [ ] Add airline filter control and top-k selector for debugging.

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
- LangFuse project + API keys configured via env vars so eval runs and metrics are captured centrally.
- Basic performance metrics (latency and cost) captured in logs or eval output.

**Tasks**
- [x] Create an initial eval set (≥ 30 questions) spanning baggage, pets, children, pregnancy, special cases; include ambiguous airline mentions.
- [x] Label gold answers, citations (airline + doc title), and expected refusals.
- [x] Build a `pytest` eval harness to run retrieval + generation and record JSONL results.
- [x] Compute retrieval metrics (Recall@k, MRR), citation correctness, groundedness, refusals.
- [x] Track latency (P50/P95) and token/cost per request.
- [x] Integrate LangFuse Python SDK: log each eval run, attach metrics, citations, and latency/cost metadata.
- [x] Store LangFuse credentials via `.env` and document the required env vars in `app/config.py` / `.env.example`.
- [x] Use evals to select models/params (speed/price vs. quality), then lock defaults.
- [x] Enable response streaming to reduce perceived latency.
- [x] Add in-memory caching for repeated queries and airline-filtered lookups.
- [x] Tune `top_k`, chunk size/overlap, and re-ranking to hit reasonable P50/P95 latency targets.
- [x] Add strict timeouts and fallbacks for any optional network activity.

**Notes**
- Eval dataset source of truth: `docs/evals/questions.jsonl` (35 labeled prompts). Harness command: `python -m app.eval --dataset docs/evals/questions.jsonl`.
- Eval runs emit JSONL under `data/evals/` and report Recall@k, MRR, citation precision/recall, refusal accuracy, latency P50/P95, token totals, and estimated USD costs. LangFuse logging is optional but ready once `LANGFUSE_*` vars are set.
- `/ask` now supports SSE streaming via `stream: true`, includes caching (configurable via `ASK_CACHE_*` env vars), and enforces LiteLLM timeouts for embeddings + completions.

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

---

## Workflow H – LangFuse Monitoring & Telemetry

**Preconditions**
- Workflows D–F: API server, eval harness, and instrumentation hooks exist.
- LangFuse workspace credentials available via environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).

**Outputs**
- LangFuse SDK initialized inside FastAPI, ingestion, and eval paths with consistent trace IDs.
- LangFuse dashboards tracking request volume, latency, model usage, eval scores, and error rates.
- Runbooks for using LangFuse UI to inspect traces, evals, and anomalies.

**Tasks**
- [ ] Add LangFuse client setup helper (e.g., `app/telemetry.py`) that reads env vars and exposes trace/span utilities.
- [ ] Instrument `/ask` handler to report traces, context chunks, LLM inputs/outputs, and any errors to LangFuse (ensure sensitive data redaction).
- [ ] Log eval harness runs to LangFuse as a dedicated dataset with pass/fail metrics and tags per workflow or dataset version.
- [ ] Configure LangFuse dashboards/alerts for latency (P50/P95), failure rate, and eval regression thresholds; document how to review them in `README.md`.
- [ ] Ensure LangFuse usage is optional (graceful no-op when credentials are missing) so local devs without keys can still run the stack.

### Local LangFuse Setup (for Workflows F & H)

When developing locally, spin up LangFuse via Docker so eval runs and API traces have a destination:

1. `git clone https://github.com/langfuse/langfuse.git && cd langfuse`.
2. Update secrets in that repo’s `docker-compose.yml` (matching the env vars configured in `.env`).  
3. `docker compose up` and wait for `langfuse-web-1` to log “Ready” (~2–3 minutes).  
4. Visit `http://localhost:3000` to access the LangFuse UI and grab the public/secret keys for your local project.

Document the resulting env vars (`LANGFUSE_HOST=http://localhost:3000`, key pair) in `.env.example` and ensure all LangFuse instrumentation handles the “service unavailable” case gracefully if Docker isn’t running.
