# Langfuse Integration Guide

This document details how the Airline Policy App integrates with [Langfuse](https://langfuse.com/) for full-stack observability, tracing, and evaluation monitoring.

## Overview

We use Langfuse to trace the execution flow of our RAG pipeline, from the initial API request down to individual LLM generations and embedding calls. This allows us to:

1.  **Debug Latency:** Identify bottlenecks in retrieval vs. generation.
2.  **Monitor Costs:** Track token usage and estimated costs per model.
3.  **Evaluate Quality:** Log evaluation runs with metrics like Recall@k and Citation Precision.
4.  **Inspect Traces:** View the exact prompt sent to the LLM and its raw response.

## Configuration

The integration is controlled via environment variables. To enable telemetry, ensure the following are set in your `.env` file:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or http://host.docker.internal:3000 for local self-hosted
```

If these variables are missing or invalid, the application gracefully disables telemetry without crashing.

## Trace Structure

### 1. API Requests (`/ask` and `/ask/stream`)

Every user request triggers a root trace.

*   **Trace Name:** `ask-request` (standard) or `ask-stream` (streaming).
*   **Input:** JSON object containing `question`, `airline`, and `top_k`.
*   **Output:** The final answer text and structured citations.
*   **Metadata:**
    *   `stream`: Boolean indicating if streaming was requested.
    *   `error`: Error message if the request failed (e.g., "Rate limit exceeded").
    *   `metrics`: A dictionary containing `latency_ms`, `tokens` (prompt/completion/embedding), `cost_usd`, and `cache_hit` status.

### 2. RAG Pipeline

Inside the root trace, we capture key spans:

*   **LLM Generations:**
    *   **Name:** Functions decorated with `@observe(as_type="generation")` (e.g., `async_chat_completion`, `embed_texts_with_litellm`).
    *   **Input:** The full list of messages sent to the model.
    *   **Output:** The raw text response from the model.
    *   **Usage:** Token counts (`input`, `output`, `total`) and calculated cost.
    *   **Model:** The specific model name used (e.g., `gpt-4o-mini`).

*   **Vector Search:**
    *   **Name:** `rag-retrieval` (implied by span structure).
    *   **Metadata:** Number of chunks retrieved and their similarity scores.

### 3. Evaluation Runs

When running the evaluation harness (`docker compose run --rm eval`), each test example generates a separate trace.

*   **Trace Name:** `rag-eval`
*   **Tags:** `eval-harness`, plus any category tags from the dataset (e.g., `baggage`, `pets`).
*   **Input:** The test question and example ID.
*   **Output:** The generated answer and citations.
*   **Scores:** Custom metrics attached to the trace:
    *   `retrieval_recall`: (0.0 - 1.0)
    *   `retrieval_mrr`: Mean Reciprocal Rank
    *   `citation_precision` / `citation_recall`
    *   `refusal_accuracy`

## Dashboard Setup

In the Langfuse UI, we recommend creating the following views:

1.  **Cost Monitor:**
    *   Metric: `SUM(cost)` grouped by `model`.
2.  **Latency Tracker:**
    *   Metric: `P95(latency)` for traces named `ask-request`.
3.  **Evaluation Leaderboard:**
    *   Table view filtered by tag `eval-harness`.
    *   Columns: `startTime`, `input`, `score.retrieval_recall`, `score.citation_precision`.

## Troubleshooting

*   **Missing Traces:** Check the logs for `LangFuse SDK not installed` or connection errors to `LANGFUSE_HOST`. Ensure `docker-compose` is forwarding the environment variables correctly.
*   **No Usage Data:** Streaming responses currently do not report token usage due to limitations in accumulating chunks. Non-streaming requests and embeddings should always report usage.
