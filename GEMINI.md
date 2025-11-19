**Scope**
- Applies to the entire repository rooted at `.`.
- Use these instructions when adding or modifying any files in this project.

**Project Goal**
- Build a small app that answers user questions about airline policies using RAG (retrieval‑augmented generation).
- Source material lives under `policies/` (Markdown and PDF). Users ask free‑form questions; the app retrieves relevant passages and composes grounded answers.

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
