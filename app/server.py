from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, Optional, cast, List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import VECTOR_STORE_PATH
from app.prompt import iter_chunk_citations
from app.rag import RagAnswer, RagEngine, RagRequest
from app.schemas import AskRequest, AskResponse, Citation, HealthResponse
from app.vector_store import VectorStore


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
_rag_engine = RagEngine(lambda: _get_vector_store_or_error())


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


@app.post("/ask", response_model=AskResponse)
async def ask_route(request: AskRequest) -> AskResponse | StreamingResponse:
    """Retrieve relevant context, run the LLM, and return an answer with citations."""
    rag_request = RagRequest(
        question=request.question,
        top_k=request.top_k,
        airline=request.airline,
    )

    if request.stream:
        return StreamingResponse(
            _iter_streaming_response(rag_request),
            media_type="text/event-stream",
        )

    answer = _run_rag_with_handling(rag_request)
    _log_answer("json", answer, request.airline)
    return AskResponse(answer=answer.answer, citations=answer.citations)


def _run_rag_with_handling(rag_request: RagRequest) -> RagAnswer:
    try:
        return _rag_engine.answer(rag_request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors
        logger.exception("RAG pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM generation failed."
        ) from exc


def _iter_streaming_response(rag_request: RagRequest) -> Iterator[str]:
    try:
        for event_type, payload in _rag_engine.stream(rag_request):
            if event_type == "token":
                yield _format_sse({"type": "token", "text": payload})
                continue
            if event_type == "citations":
                citations = cast(List[Citation], payload)
                yield _format_sse(
                    {
                        "type": "citations",
                        "citations": [
                            citation.model_dump() for citation in citations
                        ],
                    }
                )
                continue
            answer = cast(RagAnswer, payload)
            _log_answer("stream", answer, rag_request.airline)
            response_body = AskResponse(
                answer=answer.answer,
                citations=answer.citations,
            )
            yield _format_sse(
                {
                    "type": "final",
                    "answer": response_body.answer,
                    "citations": [
                        citation.model_dump() for citation in response_body.citations
                    ],
                    "metrics": _serialize_metrics(answer),
                }
            )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Streaming answer failed: %s", exc)
        yield _format_sse({"type": "error", "message": "LLM streaming failed."})


@app.post("/ask/stream")
async def ask_stream_route(request: AskRequest) -> StreamingResponse:
    """Stream NDJSON events for the UI template."""
    rag_request = RagRequest(
        question=request.question,
        top_k=request.top_k,
        airline=request.airline,
    )

    return StreamingResponse(
        _iter_ndjson_stream(rag_request),
        media_type="application/x-ndjson",
    )


def _iter_ndjson_stream(rag_request: RagRequest) -> Iterable[bytes]:
    try:
        for event_type, payload in _rag_engine.stream(rag_request):
            if event_type == "token":
                yield _json_line({"event": "chunk", "delta": payload})
                continue
            if event_type == "citations":
                citations = cast(List[Citation], payload)
                yield _json_line(
                    {
                        "event": "citations",
                        "citations": [
                            citation.model_dump() for citation in citations
                        ],
                    }
                )
                continue
            answer = cast(RagAnswer, payload)
            _log_answer("ndjson", answer, rag_request.airline)
            yield _json_line(
                {
                    "event": "complete",
                    "answer": answer.answer,
                    "citations": [
                        citation.model_dump() for citation in answer.citations
                    ],
                    "metrics": _serialize_metrics(answer),
                }
            )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Streaming answer failed: %s", exc)
        yield _json_line({"event": "error", "message": "LLM streaming failed."})


@app.get("/healthz", response_model=HealthResponse)
async def health_route() -> HealthResponse:
    """Readiness probe exposing vector store status."""
    store = _vector_store
    status_text = "ok" if store else "index_missing"
    if _vector_store_error:
        status_text = "error"
    size = store.size if store else 0
    return HealthResponse(status=status_text, vector_store_size=size)


@app.get("/", response_class=HTMLResponse)
async def ui_route() -> HTMLResponse:
    """Serve the lightweight UI for asking questions."""
    return HTMLResponse(content=_load_index_html())


def _format_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _json_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _serialize_metrics(answer: RagAnswer) -> dict[str, object]:
    return {
        "latency_ms": answer.latency_ms,
        "tokens": {
            "prompt": answer.tokens.prompt,
            "completion": answer.tokens.completion,
            "embedding": answer.tokens.embedding,
        },
        "cost_usd": {
            "prompt": answer.costs.prompt_usd,
            "completion": answer.costs.completion_usd,
            "embedding": answer.costs.embedding_usd,
            "total": answer.costs.total,
        },
        "cache_hit": answer.from_cache,
    }


def _log_answer(channel: str, answer: RagAnswer, airline: Optional[str]) -> None:
    citation_summary = ", ".join(iter_chunk_citations(answer.contexts)) or "no citations"
    logger.info(
        "[%s] Answered in %.1f ms with %d chunks (%s) cache=%s airline_filter=%s",
        channel,
        answer.latency_ms,
        len(answer.citations),
        citation_summary,
        answer.from_cache,
        airline or "*",
    )


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
