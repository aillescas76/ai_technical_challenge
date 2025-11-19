# Technical Analysis and Proposed Solution

This document analyzes the challenge described in `README.md` and proposes a practical, latency-conscious RAG (retrieval‑augmented generation) solution for answering airline policy questions using the documents in `policies/`. It emphasizes two key considerations for ingestion and outlines evaluation and performance strategies to guide model and system decisions.

## Summary of the Challenge

- Build a small web app where users ask questions about airline policies (Delta, United, American Airlines) and receive grounded answers with references.
- Process policy documents (Markdown and PDF) and store embeddings in a vector database for similarity search.
- Integrate an LLM to compose concise, evidence-based answers.

## Proposed Architecture

- API/UI: FastAPI + Uvicorn (minimal HTML page or JSON endpoint).
- Ingestion: Parse Markdown directly; extract text and links from PDFs using `pypdf`. Normalize metadata.
- Vector DB: FAISS (CPU) persisted on disk for fast local retrieval; alternatives like Chroma are acceptable.
- Embeddings: OpenAI `text-embedding-3-small` for cost/speed; fallback to `sentence-transformers/all-MiniLM-L6-v2` offline.
- LLM: OpenAI `gpt-4o-mini` (fast, cost-effective). Enable streaming responses to reduce perceived latency.
- Prompting: Grounded answering with explicit citation requirements; refuse to speculate when evidence is insufficient.

## Ingestion Strategy

### Emphasis 1 — Use Path-Derived Metadata

Much of the source information is conveyed by the file path and filename, e.g.:

```
policies/Delta/Children Infant Travel.md
policies/United/Checked bags.pdf
policies/AmericanAirlines/Traveling with children.md
```

Leverage this structure explicitly during ingestion:

- airline: taken from the immediate parent directory (e.g., `Delta`, `United`, `AmericanAirlines`).
- title: derived from the filename without extension (e.g., `Children Infant Travel`).
- category: optional heuristic from title keywords (e.g., baggage, pets, children, pregnancy). Useful for routing and re-ranking.
- source_path: original relative file path for traceability.
- source_url: if the document body contains canonical links, store the best candidate URL (see next section).

This metadata improves retrieval (filtering by airline), helps prompt grounding (cite airline and document title), and enables more reliable evaluation splits (by airline/category).

### Document Parsing and Chunking

- Markdown: read as UTF-8, preserve headings; extract inline links.
- PDF: use `pypdf` for text; attempt to capture link annotations when available.
- Chunking: 800–1000 token chunks with ~150 token overlap. Keep chunk-level metadata: `airline`, `title`, `category`, `source_path`, `chunk_index`.
- Store processed artifacts (text + metadata) to `data/processed.jsonl` for reproducibility.

### Indexing and Retrieval

- Create embeddings for each chunk and build a FAISS index persisted under `data/faiss/`.
- Retrieval: cosine similarity, `top_k=5`; optional MMR to diversify contexts.
- Re-ranking: apply light heuristics (same-airline bias if user mentions airline; category match boost).

## Handling Web Links in Documents

Many source documents contain links to the full or canonical policy pages. We need a strategy that balances freshness, complexity, and latency:

1) Proactive crawl and ingest
   - Approach: Fetch known canonical URLs and ingest their content alongside local docs.
   - Pros: Improved coverage; answers remain closer to the latest official policies.
   - Cons: Adds build-time complexity and potential legal/compliance concerns; risk of content drift; increases index size and ingestion time.

2) On-demand fetch (lazy retrieval)
   - Approach: If retrieval confidence is low or answer is incomplete, fetch linked pages at question time and augment the context temporarily.
   - Pros: Keeps index small; more accurate answers when needed.
   - Cons: Adds runtime latency; requires careful caching and timeouts; network failures complicate UX.

3) Inform-and-cite only (no crawl)
   - Approach: Do not fetch external URLs. Cite extracted links and optionally present a “View official policy” link with a brief note.
   - Pros: Simple, fast, and predictable; lowest complexity and latency.
   - Cons: May miss details that aren’t captured in local snippets; relies on the user to click through.

Recommendation for MVP: Option 3 (Inform-and-cite only), with clear citations and “official source” links where available. Add Option 2 as an opt-in feature flag for a v1.1 enhancement, using caching and strict per-request time budgets (e.g., 500–800 ms for fetch + parse) to protect latency.

Implementation notes:

- During ingestion, extract and store the best canonical URL found in each document (Markdown links, PDF link annotations). Persist as `source_url` in metadata. Prefer official airline domains.
- At answer time, show citations as `Airline – Title` and, if `source_url` exists, include the link in the response payload/UI.

## Evals to Guide Model Decisions

We will build a lightweight, repeatable evaluation harness to drive decisions about models, chunking, retrieval parameters, and link-handling behavior.

Evaluation dataset:
- Start with the four sample queries in `README.md` and expand to ~30–50 questions covering baggage, pets, children, pregnancy, special circumstances, and ambiguous airline mentions.
- Label gold answers with short, extractive rationales and required citations (airline + doc title). Include expected failure behavior (e.g., “no answer found” when appropriate).

Metrics and checks:
- Retrieval: Recall@k, MRR@k, and coverage by airline/category.
- Answer quality: Exactness/adequacy judged against gold, citation correctness (points to the right airline/title), groundedness (penalize unsupported claims), refusal correctness (for unknowns).
- Latency and cost: End-to-end 50th/95th percentiles and per-request token/cost tracking.

Automation:
- Implement `pytest`-driven evals with a small harness that runs retrieval + generation and writes JSONL results.
- Use an LLM-as-judge sparingly for qualitative scoring, with seeded prompts and deterministic temperature; keep a small human spot-check loop.

Model selection:
- Start with `gpt-4o-mini` for responses and `text-embedding-3-small` for embeddings (good price/perf). Compare against a larger model on the eval set to check for measurable gains before upgrading.
- Tune `top_k`, chunk size/overlap, and airline/category re-ranking based on retrieval metrics.
- Ensure the chosen approach meets latency budgets (see below) before promoting it.

## Latency Strategy

Targets (per question):
- Retrieval (index search + re-ranking): ≤ 80–120 ms on a modest CPU.
- Prompt assembly + I/O: ≤ 50–100 ms.
- LLM first-token latency: ≤ 300–600 ms for mini models; stream tokens to UI.
- Total P50 under 1.2–1.5 s; P95 under 2.5–3.0 s.

Tactics:
- Precompute and persist embeddings; memory-map FAISS index at startup.
- Keep prompts short and token-efficient; avoid verbose system messages.
- Use smaller, faster models by default; only escalate model size if evals justify it.
- Set `top_k` to 3–5 and MMR to reduce redundant tokens.
- Implement response streaming for early feedback; render citations as soon as the answer stabilizes.
- Add in-memory caching keyed by normalized query + airline filter.
- Apply strict timeouts and fallbacks; if on-demand URL fetch is enabled, keep it off the critical path or cap it with early fallback.

## Risks and Mitigations

- Hallucinations or incorrect citations: Use grounded prompts, require citations, and include a confidence/coverage check. Fall back to “cannot find” when retrieval is low-confidence.
- Stale policies: Prefer local docs as the source of truth for MVP; show official links; optionally support scheduled re-ingestion.
- PDF extraction quality: Validate a sample set; consider `unstructured` if `pypdf` struggles on specific files.
- Scope creep from external crawling: Keep it behind a feature flag; document latency and legal/compliance implications.

## Deliverables and Next Steps

- MVP
  - Implement ingestion with path-derived metadata and link extraction.
  - Build FAISS index and retrieval API with airline/category filters.
  - Add a minimal FastAPI UI; show answers with citations and official links.
  - Add eval harness (dataset + tests) to measure retrieval/answer quality, latency, and cost.

- v1.1 Enhancements
  - Optional on-demand URL fetch with caching and tight time budgets.
  - Re-ranking improvements and airline/category routing.
  - Expanded eval set and automated reporting.

This plan keeps the MVP simple and fast, leverages path-derived metadata for accuracy, and uses evals to guide model choice and parameter tuning while meeting user-facing latency needs.

