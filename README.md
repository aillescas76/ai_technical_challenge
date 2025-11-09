## LLM Airline Policy App

### Project Overview

This project implements a small, retrieval‑augmented generation (RAG) application that answers user questions about airline policies (Delta, United, American Airlines) using an LLM and a vector database. Policy documents in `policies/` (Markdown and PDF) are ingested, chunked, embedded, and indexed for similarity search. The API assembles relevant context and asks the LLM to produce concise, grounded answers with citations to the source documents.

We will add setup and run instructions here in a later step.

### Documentation

- Full challenge description (original README.md file): [docs/challenge.md](docs/challenge.md)
- Analysis and proposed solution: [docs/analysis.md](docs/analysis.md)
