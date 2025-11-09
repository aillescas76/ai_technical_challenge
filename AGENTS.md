**Scope**
- Applies to the entire repository rooted at `.`.
- Use these instructions when adding or modifying any files in this project.

**Project Goal**
- Build a small app that answers user questions about airline policies using RAG (retrieval‑augmented generation).
- Source material lives under `policies/` (Markdown and PDF). Users ask free‑form questions; the app retrieves relevant passages and composes grounded answers.

**Recommended Stack**
- Python 3.11+
- API/UI: `FastAPI` + `Uvicorn` (minimal HTML template or simple JS; Streamlit acceptable if faster to deliver).
- Embeddings: OpenAI `text-embedding-3-small` (fallback: `sentence-transformers` `all-MiniLM-L6-v2`).
- LLM for answers: OpenAI `gpt-4o-mini` (fallback: any GPT-4/3.5 tier you have). Keep responses grounded and cite sources.
- Vector store: `FAISS` (local on-disk index). Alternatives like `Chroma` are fine if simpler.

**Proposed Layout**
- `app/config.py` – config, env parsing, constants.
- `app/ingest.py` – load + chunk documents from `policies/`, build/update vector index.
- `app/vector_store.py` – FAISS (or chosen store) wrapper: add, search, persist.
- `app/llm.py` – LLM + embeddings clients; small abstraction to swap providers.
- `app/prompt.py` – system and answer prompts for grounded Q&A with citations.
- `app/schemas.py` – Pydantic models for requests/responses.
- `app/server.py` – FastAPI app: `/ask` endpoint, simple web UI.
- `data/` – generated artifacts (processed docs, vector index). Git-ignored.
- `tests/` – focused unit tests (chunking, retrieval, prompt formatting).

**Core Workflow**
- Ingestion
  - Read Markdown directly; parse PDFs via `pypdf`.
  - Normalize metadata: `airline`, `title`, `source_path`, `chunk_id`.
  - Chunk size ~800–1000 tokens; overlap ~150. Store processed docs to `data/processed.jsonl`.
  - Embed and write/merge FAISS index to `data/faiss/`.
- Retrieval
  - Use cosine similarity; `top_k=5` with MMR/diversity if available.
  - Return snippets with metadata for citation and airline filtering.
- Answering
  - RAG prompt instructs the model to answer concisely, cite airline + document title, and state "no answer found" when evidence is missing.

**Environment & Secrets**
- Required env vars (use `.env` locally; never commit secrets):
  - `OPENAI_API_KEY` – for embeddings and chat completions (if using OpenAI).
  - `EMBEDDINGS_MODEL` (default `text-embedding-3-small`).
  - `LLM_MODEL` (default `gpt-4o-mini`).
  - `VECTOR_STORE_PATH` (default `data/faiss`).
- Add a `.env.example` showing variable names without values.

**Local Setup & Commands**
- Create `requirements.txt` with: `fastapi`, `uvicorn[standard]`, `pydantic`, `faiss-cpu`, `tiktoken`, `pypdf`, `python-dotenv` and chosen LLM SDKs (e.g., `openai`).
- Typical flow
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `python -m app.ingest` (build index)
  - `uvicorn app.server:app --reload` (visit `http://localhost:8000`)

**UI Expectations**
- Simple page with a text box and an answer area showing:
  - Final answer
  - Cited sources (airline + document title, and optionally chunk preview)
- Add a minimal `/docs` (FastAPI) and `/healthz` endpoint.

**Coding Conventions**
- Style: `black` + `isort`; lint with `ruff` or `flake8`. Type hints required.
- Structure modules as small, single‑purpose units; avoid cross‑layer imports.
- Docstrings for public functions; prefer pure functions for chunking logic.
- Log with `logging` (structured if convenient); no secrets in logs.

See also: `docs/python_guidelines.md` for Python‑specific development guidance.

**Testing Guidance**
- Use `pytest`. Add fast tests for:
  - PDF/Markdown loaders return text and metadata.
  - Chunking respects size/overlap and preserves metadata.
  - Retrieval returns expected docs for the sample queries from `README.md` when using a deterministic embedding stub.
- Mock LLM calls; do not require network for tests.

**Data & Privacy**
- Do not commit generated data or indexes. Ensure `data/` and `.env` are git-ignored.
- Never print or log API keys. Handle provider errors gracefully; surface clear messages.

**Performance Notes**
- Reuse the embeddings client; batch embeddings where possible.
- Build the index once and memory-map on server start if supported.
- Keep `top_k` small (3–8); paginate if showing more context.

**Deliverables Alignment**
- Keep `README.md` updated with:
  - Setup and run instructions
  - Design choices (stack, RAG decisions, tradeoffs)
  - Challenges and mitigations
- Provide a short evaluation section demonstrating answers to the 4 sample queries from `README.md` and include citations.

**Git Hygiene**
- Small, focused commits with clear messages. Avoid unrelated refactors.
- Do not commit secrets or large artifacts. Add/maintain `.gitignore` for `data/`, `.env`, caches, and virtualenv.

**When In Doubt**
- Favor simplicity and traceability over features.
- Ask/flag before performing destructive changes, large dependency additions, or migrating the stack.
- If external APIs are unavailable, implement with open-source fallbacks and document limitations.
