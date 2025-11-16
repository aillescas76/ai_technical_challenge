## LLM Airline Policy App

### Project Overview

This project implements a small, retrieval‑augmented generation (RAG) application that answers user questions about airline policies (Delta, United, American Airlines) using an LLM and a vector database. Policy documents in `policies/` (Markdown and PDF) are ingested, chunked, embedded, and indexed for similarity search. The API assembles relevant context and asks the LLM to produce concise, grounded answers with citations to the source documents.

### Ingesting Policies

After installing dependencies (`pip install -r requirements.txt`) and setting `OPENAI_API_KEY` (or another LiteLLM‑supported provider key) in your environment or `.env`, build the processed dataset and FAISS index by running:

```bash
python -m app.ingest
```

This command will:

1. Load all Markdown and PDF files under `policies/`
2. Chunk and write them to `data/processed.jsonl`
3. Generate embeddings via LiteLLM and persist a FAISS index under `data/faiss/`

Run this ingestion step whenever policies change so the RAG system uses the latest content.

We will add setup and run instructions here in a later step.

### Documentation

- Full challenge description (original README.md file): [docs/challenge.md](docs/challenge.md)
- Analysis and proposed solution: [docs/analysis.md](docs/analysis.md)
