from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from contextlib import asynccontextmanager
from typing import Iterable, Iterator, Optional, cast, List, Any

import json_logging
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from app.config import (LOG_LEVEL, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_TTL_SECONDS, VECTOR_STORE_PATH)
from app.prompt import iter_chunk_citations
from app.rag import RagAnswer, RagEngine, RagRequest
from app.schemas import AskRequest, AskResponse, Citation, HealthResponse, Metrics
from app.telemetry import Telemetry
from app.vector_store import VectorStore


logger = logging.getLogger(__name__)


_vector_store: Optional[VectorStore] = None
_vector_store_error: Optional[str] = None
_vector_store_lock = Lock()
_index_html_cache: Optional[str] = None

_requests_total: int = 0
_requests_current: int = 0
_errors_total: int = 0
_start_time: float = time.time()
_metrics_lock = Lock()

_rate_limit_cache: TTLCache[str, int] = TTLCache(
    maxsize=RATE_LIMIT_MAX_REQUESTS * 4, ttl=RATE_LIMIT_TTL_SECONDS
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attempt to load the FAISS vector store when the API boots."""
    _ensure_vector_store_loaded()
    global _index_html_cache
    if _index_html_cache is None:
        templates_dir = Path(__file__).with_name("templates")
        index_path = templates_dir / "index.html"
        try:
            _index_html_cache = index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("UI template missing at %s", index_path)
            _index_html_cache = "<html><body><h1>UI template missing.</h1></body></html>"

    json_logging.init_fastapi(enable_json=True, custom_formatter=json_logging.UnifiedJSONFormatter)
    json_logging.init_request_instrument(app)
    logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)
    logger.setLevel(LOG_LEVEL)
    yield


app = FastAPI(
    title="Airline Policy RAG API",
    description="Retrieval-augmented FastAPI service for airline policy Q&A.",
    version="0.1.0",
    lifespan=lifespan,
)


def _increment_counter(counter_name: str) -> None:
    global _requests_total, _requests_current, _errors_total
    with _metrics_lock:
        if counter_name == "requests_total":
            _requests_total += 1
        elif counter_name == "requests_current":
            _requests_current += 1
        elif counter_name == "errors_total":
            _errors_total += 1


def _decrement_counter(counter_name: str) -> None:
    global _requests_current
    with _metrics_lock:
        if counter_name == "requests_current":
            _requests_current -= 1


async def rate_limit_dependency(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    with _metrics_lock:
        count = _rate_limit_cache.get(client_ip, 0)
        if count >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
        _rate_limit_cache[client_ip] = count + 1

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


_rag_engine = RagEngine(store_getter=_get_vector_store_or_error)


@app.post("/ask", response_model=AskResponse)
async def ask_route(request: AskRequest, rate_limit: None = Depends(rate_limit_dependency)) -> AskResponse | StreamingResponse:
    """Retrieve relevant context, run the LLM, and return an answer with citations."""
    _increment_counter("requests_total")
    _increment_counter("requests_current")
    
    # Start LangFuse trace
    telemetry = Telemetry.get_instance()
    trace = telemetry.trace(
        name="ask-request",
        input={"question": request.question, "airline": request.airline, "top_k": request.top_k},
        metadata={"stream": request.stream}
    )

    rag_request = RagRequest(
        question=request.question,
        top_k=request.top_k,
        airline=request.airline,
    )

    if request.stream:
        # Pass trace to streaming handler to update it on completion
        return StreamingResponse(
            _iter_streaming_response(rag_request, trace),
            media_type="text/event-stream",
        )

    try:
        answer = await _run_rag_with_handling(rag_request)
        _log_answer("json", answer, request.airline)
        
        # Update trace with success
        trace.update(
            output={"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations]},
            metadata=_serialize_metrics(answer)
        )
        telemetry.flush()
        
        return AskResponse(answer=answer.answer, citations=answer.citations)
    except HTTPException as e:
        _increment_counter("errors_total")
        trace.update(metadata={"error": str(e)})
        telemetry.flush()
        raise e
    except Exception as e:
        _increment_counter("errors_total")
        trace.update(metadata={"error": str(e)})
        telemetry.flush()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error."
        ) from e
    finally:
        _decrement_counter("requests_current")


async def _run_rag_with_handling(rag_request: RagRequest) -> RagAnswer:
    try:
        answer = await _rag_engine.answer(rag_request)
        # If the LLM explicitly states no answer, return a 404.
        # This handles cases where the RAG system determines it lacks sufficient evidence.
        if any(term in answer.answer.lower() for term in ["no answer found", "insufficient information", "could not find"]):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No answer found based on the provided policies."
            )
        return answer
    except HTTPException:
        _increment_counter("errors_total")
        raise
    except Exception as exc:  # pragma: no cover - network errors
        _increment_counter("errors_total")
        logger.exception("RAG pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM generation failed."
        ) from exc


async def _iter_streaming_response(rag_request: RagRequest, trace: Any = None) -> Iterator[str]:
    try:
        async for event_type, payload in _rag_engine.stream(rag_request):
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
            
            if trace:
                trace.update(
                    output={"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations]},
                    metadata=_serialize_metrics(answer)
                )
                Telemetry.get_instance().flush()

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
        _increment_counter("errors_total")
        if trace:
            trace.update(metadata={"error": "HTTPException"})
            Telemetry.get_instance().flush()
        raise
    except Exception as exc:  # pragma: no cover
        _increment_counter("errors_total")
        logger.exception("Streaming answer failed: %s", exc)
        if trace:
            trace.update(metadata={"error": str(exc)})
            Telemetry.get_instance().flush()
        yield _format_sse({"type": "error", "message": "LLM streaming failed."})
    finally:
        _decrement_counter("requests_current")

@app.post("/ask/stream")
async def ask_stream_route(request: AskRequest, rate_limit: None = Depends(rate_limit_dependency)) -> StreamingResponse:
    """Stream NDJSON events for the UI template."""
    _increment_counter("requests_total")
    _increment_counter("requests_current")
    
    telemetry = Telemetry.get_instance()
    trace = telemetry.trace(
        name="ask-stream",
        input={"question": request.question, "airline": request.airline, "top_k": request.top_k},
        metadata={"stream": True, "format": "ndjson"}
    )

    rag_request = RagRequest(
        question=request.question,
        top_k=request.top_k,
        airline=request.airline,
    )

    return StreamingResponse(
        _iter_ndjson_stream(rag_request, trace),
        media_type="application/x-ndjson",
    )


async def _iter_ndjson_stream(rag_request: RagRequest, trace: Any = None) -> Iterable[bytes]:
    try:
        async for event_type, payload in _rag_engine.stream(rag_request):
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
            
            if trace:
                trace.update(
                    output={"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations]},
                    metadata=_serialize_metrics(answer)
                )
                Telemetry.get_instance().flush()

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
        _increment_counter("errors_total")
        if trace:
            trace.update(metadata={"error": "HTTPException"})
            Telemetry.get_instance().flush()
        raise
    except Exception as exc:  # pragma: no cover
        _increment_counter("errors_total")
        logger.exception("Streaming answer failed: %s", exc)
        if trace:
            trace.update(metadata={"error": str(exc)})
            Telemetry.get_instance().flush()
        yield _json_line({"event": "error", "message": "LLM streaming failed."})
    finally:
        _decrement_counter("requests_current")

@app.get("/healthz", response_model=HealthResponse)
async def health_route() -> HealthResponse:
    """Readiness probe exposing vector store status."""
    store = _vector_store
    status_text = "ok" if store else "index_missing"
    if _vector_store_error:
        status_text = "error"
    size = store.size if store else 0
    return HealthResponse(status=status_text, vector_store_size=size)


@app.get("/metrics", response_model=Metrics)
async def metrics_route() -> Metrics:
    """Return basic monitoring metrics for the service."""
    with _metrics_lock:
        return Metrics(
            requests_total=_requests_total,
            requests_current=_requests_current,
            errors_total=_errors_total,
            uptime_seconds=time.time() - _start_time,
        )


@app.get("/", response_class=HTMLResponse)
async def ui_route() -> HTMLResponse:
    """Serve the lightweight UI for asking questions."""
    return _index_html_cache


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
