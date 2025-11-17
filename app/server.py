from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import VECTOR_STORE_PATH
from app.llm import chat_completion, embed_texts_with_litellm, stream_chat_completion
from app.prompt import ContextChunk, build_grounded_answer_messages, iter_chunk_citations
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
_index_html_cache: Optional[str] = None


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


def _results_to_context_chunks(results: List[SearchResult]) -> List[ContextChunk]:
    """Convert raw search results into prompt-friendly context chunks."""
    contexts: List[ContextChunk] = []
    for result in results:
        metadata = result.metadata or {}
        chunk_id = metadata.get("id")
        if chunk_id is None and metadata.get("chunk_index") is not None:
            chunk_id = str(metadata["chunk_index"])
        contexts.append(
            ContextChunk(
                content=result.text,
                airline=str(metadata.get("airline", "Unknown Airline")),
                title=str(metadata.get("title", "Unknown Document")),
                source_path=str(metadata.get("source_path", "")),
                chunk_id=str(chunk_id) if chunk_id is not None else None,
                source_url=metadata.get("source_url"),
            )
        )
    return contexts


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


def _json_line(payload: dict[str, object]) -> bytes:
    """Serialize a payload to a JSON line for streaming responses."""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


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

    contexts = _results_to_context_chunks(results)
    messages = build_grounded_answer_messages(question=request.question, contexts=contexts)
    try:
        answer = chat_completion(messages).strip()
    except Exception as exc:  # pragma: no cover - network errors
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM generation failed."
        ) from exc

    if not answer:
        answer = "No answer found."

    citation_summary = ", ".join(iter_chunk_citations(contexts)) or "no citations"
    logger.info(
        "Answered question with %d chunks (%s)",
        len(results),
        citation_summary,
    )

    citations = [_result_to_citation(result) for result in results]
    return AskResponse(answer=answer, citations=citations)


@app.post("/ask/stream")
async def ask_stream_route(request: AskRequest) -> StreamingResponse:
    """Stream answer tokens along with citation metadata for the UI."""

    def _respond_with_error(message: str) -> StreamingResponse:
        def _error_stream() -> Iterable[bytes]:
            yield _json_line({"event": "error", "message": message})

        return StreamingResponse(_error_stream(), media_type="application/x-ndjson")

    try:
        results = _retrieve_context(
            request.question, top_k=request.top_k, airline_filter=request.airline
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to retrieve context for streaming answer: %s", exc)
        return _respond_with_error("Failed to retrieve supporting context.")

    if not results:
        def _no_results_stream() -> Iterable[bytes]:
            yield _json_line(
                {"event": "complete", "answer": "No answer found.", "citations": []}
            )

        return StreamingResponse(_no_results_stream(), media_type="application/x-ndjson")

    contexts = _results_to_context_chunks(results)
    messages = build_grounded_answer_messages(question=request.question, contexts=contexts)
    citations = [_result_to_citation(result) for result in results]

    def _stream_answer() -> Iterable[bytes]:
        yield _json_line(
            {
                "event": "citations",
                "citations": [citation.model_dump() for citation in citations],
            }
        )
        deltas: List[str] = []
        try:
            for chunk in stream_chat_completion(messages):
                deltas.append(chunk)
                yield _json_line({"event": "chunk", "delta": chunk})
        except Exception as exc:  # pragma: no cover - network errors
            logger.exception("Streaming LLM generation failed: %s", exc)
            yield _json_line({"event": "error", "message": "LLM generation failed."})
            return

        final_answer = "".join(deltas).strip() or "No answer found."
        citation_summary = ", ".join(iter_chunk_citations(contexts)) or "no citations"
        logger.info(
            "Streamed answer with %d chunks (%s)",
            len(results),
            citation_summary,
        )
        yield _json_line(
            {
                "event": "complete",
                "answer": final_answer,
                "citations": [citation.model_dump() for citation in citations],
            }
        )

    return StreamingResponse(_stream_answer(), media_type="application/x-ndjson")


@app.get("/healthz", response_model=HealthResponse)
async def health_route() -> HealthResponse:
    """Readiness probe exposing vector store status."""
    store = _vector_store
    status_text = "ok" if store else "index_missing"
    if _vector_store_error:
        status_text = "error"
    size = store.size if store else 0
    return HealthResponse(status=status_text, vector_store_size=size)


def _load_index_html() -> str:
    global _index_html_cache
    if _index_html_cache is None:
        templates_dir = Path(__file__).with_name("templates")
        index_path = templates_dir / "index.html"
        try:
            _index_html_cache = index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("UI template missing at %s", index_path)
            _index_html_cache = "<html><body><h1>UI template missing.</h1></body></html>"
    return _index_html_cache


@app.get("/", response_class=HTMLResponse)
async def ui_route() -> HTMLResponse:
    """Serve the lightweight UI for asking questions."""
    return HTMLResponse(content=_load_index_html())
