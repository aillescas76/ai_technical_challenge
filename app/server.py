from __future__ import annotations

import logging
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status

from app.config import VECTOR_STORE_PATH
from app.llm import chat_completion, embed_texts_with_litellm
from app.prompt import build_rag_messages, summarize_context_for_logging
from app.schemas import AskRequest, AskResponse, Citation, HealthResponse
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Airline Policy RAG API",
    description="Retrieval-augmented FastAPI service for airline policy Q&A.",
    version="0.1.0",
)

_vector_store: Optional[VectorStore] = None
_vector_store_error: Optional[str] = None
_vector_store_lock = Lock()


@app.on_event("startup")
def _startup_event() -> None:
    """Attempt to load the FAISS vector store when the API boots."""
    _ensure_vector_store_loaded()


def _ensure_vector_store_loaded() -> Optional[VectorStore]:
    """Load the vector store from disk if it is not already cached."""
    global _vector_store, _vector_store_error
    with _vector_store_lock:
        if _vector_store is not None:
            return _vector_store
        try:
            _vector_store = VectorStore.load(VECTOR_STORE_PATH)
        except FileNotFoundError:
            _vector_store = None
            _vector_store_error = (
                f"Vector index not found in {VECTOR_STORE_PATH}. "
                "Run `python -m app.ingest` to build it."
            )
            logger.warning("%s", _vector_store_error)
        except Exception:
            _vector_store = None
            _vector_store_error = "Failed to load vector index."
            logger.exception("Unable to load vector store from %s", VECTOR_STORE_PATH)
        else:
            _vector_store_error = None
            logger.info(
                "Loaded vector store with %d embeddings from %s",
                _vector_store.size,
                VECTOR_STORE_PATH,
            )
    return _vector_store


def _get_vector_store_or_error() -> VectorStore:
    store = _ensure_vector_store_loaded()
    if store is None:
        detail = _vector_store_error or "Vector index is not available."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return store


def _result_to_citation(result: SearchResult) -> Citation:
    metadata = result.metadata or {}
    snippet = _build_snippet(result.text)
    return Citation(
        id=result.id,
        airline=str(metadata.get("airline", "Unknown Airline")),
        title=str(metadata.get("title", "Unknown Document")),
        source_path=str(metadata.get("source_path", "")),
        source_url=metadata.get("source_url"),
        chunk_index=metadata.get("chunk_index"),
        score=float(result.score),
        snippet=snippet,
    )


def _build_snippet(text: str, limit: int = 320) -> str:
    condensed = " ".join(text.split())
    if len(condensed) <= limit:
        return condensed
    cutoff = condensed.rfind(" ", 0, limit)
    if cutoff == -1:
        cutoff = limit
    return condensed[:cutoff].rstrip() + "..."


def _retrieve_context(
    question: str, *, top_k: int, airline_filter: Optional[str]
) -> List[SearchResult]:
    store = _get_vector_store_or_error()
    embeddings = embed_texts_with_litellm([question])
    if not embeddings:
        raise RuntimeError("Embeddings provider returned no results.")
    query_embedding = embeddings[0]

    search_k = top_k if not airline_filter else top_k * 3
    results = store.search_by_embedding(query_embedding, top_k=search_k)
    if airline_filter:
        normalized_filter = airline_filter.strip().lower()
        results = [
            result
            for result in results
            if str(result.metadata.get("airline", "")).lower() == normalized_filter
        ]
    return results[:top_k]


@app.post("/ask", response_model=AskResponse)
async def ask_route(request: AskRequest) -> AskResponse:
    """Retrieve relevant context, run the LLM, and return an answer with citations."""
    try:
        results = _retrieve_context(
            request.question, top_k=request.top_k, airline_filter=request.airline
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to retrieve context: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supporting context.",
        ) from exc

    if not results:
        return AskResponse(answer="No answer found.", citations=[])

    messages = build_rag_messages(request.question, results)
    try:
        answer = chat_completion(messages).strip()
    except Exception as exc:  # pragma: no cover - network errors
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM generation failed."
        ) from exc

    if not answer:
        answer = "No answer found."

    logger.info(
        "Answered question with %d chunks (%s)",
        len(results),
        summarize_context_for_logging(results),
    )

    citations = [_result_to_citation(result) for result in results]
    return AskResponse(answer=answer, citations=citations)


@app.get("/healthz", response_model=HealthResponse)
async def health_route() -> HealthResponse:
    """Readiness probe exposing vector store status."""
    store = _vector_store
    status_text = "ok" if store else "index_missing"
    if _vector_store_error:
        status_text = "error"
    size = store.size if store else 0
    return HealthResponse(status=status_text, vector_store_size=size)
