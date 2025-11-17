from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

import tiktoken

from app.airlines import normalize_airline_key
from app.config import (
    ASK_CACHE_MAX_ITEMS,
    ASK_CACHE_TTL_SECONDS,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_TIMEOUT_SECONDS,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    MODEL_COST_LOOKUP,
    TOKEN_ENCODING_NAME,
)
from app.llm import chat_completion, embed_texts_with_litellm, stream_chat_completion
from app.prompt import ContextChunk, build_grounded_answer_messages
from app.schemas import Citation
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


@dataclass
class RagRequest:
    """Normalized request parameters for the RAG pipeline."""

    question: str
    top_k: int
    airline: Optional[str] = None


@dataclass
class TokenUsage:
    prompt: int
    completion: int
    embedding: int


@dataclass
class CostBreakdown:
    prompt_usd: float
    completion_usd: float
    embedding_usd: float

    @property
    def total(self) -> float:
        return self.prompt_usd + self.completion_usd + self.embedding_usd


@dataclass
class RagAnswer:
    answer: str
    citations: List[Citation]
    contexts: List[ContextChunk]
    retrievals: List[SearchResult]
    latency_ms: float
    tokens: TokenUsage
    costs: CostBreakdown
    from_cache: bool = False

    def to_record(self) -> dict:
        """Return a JSON-serializable record for eval logs."""
        return {
            "answer": self.answer,
            "citations": [citation.model_dump() for citation in self.citations],
            "latency_ms": self.latency_ms,
            "tokens": {
                "prompt": self.tokens.prompt,
                "completion": self.tokens.completion,
                "embedding": self.tokens.embedding,
            },
            "cost_usd": {
                "prompt": self.costs.prompt_usd,
                "completion": self.costs.completion_usd,
                "embedding": self.costs.embedding_usd,
                "total": self.costs.total,
            },
            "from_cache": self.from_cache,
            "retrieval_ids": [result.id for result in self.retrievals],
        }


@dataclass
class _CacheEntry:
    value: RagAnswer
    expires_at: float


class _ResponseCache:
    """Simple LRU cache with TTL for RAG answers."""

    def __init__(self, *, max_items: int, ttl_seconds: int) -> None:
        self._max_items = max(1, max_items)
        self._ttl = max(1, ttl_seconds)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[RagAnswer]:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: RagAnswer) -> None:
        expires_at = time.time() + self._ttl
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_items:
                self._entries.popitem(last=False)


class RagEngine:
    """Shared retrieval + generation pipeline with caching and metrics."""

    def __init__(self, store_getter: Callable[[], VectorStore]) -> None:
        self._store_getter = store_getter
        self._cache = _ResponseCache(
            max_items=ASK_CACHE_MAX_ITEMS, ttl_seconds=ASK_CACHE_TTL_SECONDS
        )
        self._encoding = tiktoken.get_encoding(TOKEN_ENCODING_NAME)

    def answer(self, request: RagRequest) -> RagAnswer:
        """Return a grounded answer (non-streaming)."""
        normalized_request = self._normalize_request(request)
        cache_key = self._cache_key(normalized_request)
        cached = self._cache.get(cache_key)
        if cached:
            cached.from_cache = True
            return cached

        retrievals, contexts, citations = self._retrieve_records(normalized_request)
        if not retrievals:
            empty_answer = RagAnswer(
                answer="No answer found.",
                citations=[],
                contexts=[],
                retrievals=[],
                latency_ms=0.0,
                tokens=TokenUsage(prompt=0, completion=0, embedding=0),
                costs=CostBreakdown(prompt_usd=0.0, completion_usd=0.0, embedding_usd=0.0),
            )
            self._cache.set(cache_key, empty_answer)
            return empty_answer

        messages = build_grounded_answer_messages(
            question=normalized_request.question, contexts=contexts
        )
        embedding_tokens = self._estimate_tokens(normalized_request.question)

        start = perf_counter()
        completion = chat_completion(
            messages, model=LLM_MODEL, timeout=LLM_TIMEOUT_SECONDS
        ).strip()
        latency_ms = (perf_counter() - start) * 1000

        answer_text = completion or "No answer found."
        prompt_tokens = self._estimate_message_tokens(messages)
        completion_tokens = self._estimate_tokens(answer_text)
        answer = RagAnswer(
            answer=answer_text,
            citations=citations,
            contexts=contexts,
            retrievals=retrievals,
            latency_ms=latency_ms,
            tokens=TokenUsage(
                prompt=prompt_tokens,
                completion=completion_tokens,
                embedding=embedding_tokens,
            ),
            costs=self._calculate_costs(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                embedding_tokens=embedding_tokens,
            ),
        )
        self._cache.set(cache_key, answer)
        return answer

    def stream(
        self, request: RagRequest
    ) -> Iterator[Tuple[Literal["token", "final"], str | RagAnswer]]:
        """Yield streaming events (token chunks, then final answer)."""
        normalized_request = self._normalize_request(request)
        cache_key = self._cache_key(normalized_request)
        cached = self._cache.get(cache_key)
        if cached:
            cached.from_cache = True
            yield ("final", cached)
            return

        retrievals, contexts, citations = self._retrieve_records(normalized_request)
        if not retrievals:
            empty_answer = RagAnswer(
                answer="No answer found.",
                citations=[],
                contexts=[],
                retrievals=[],
                latency_ms=0.0,
                tokens=TokenUsage(prompt=0, completion=0, embedding=0),
                costs=CostBreakdown(prompt_usd=0.0, completion_usd=0.0, embedding_usd=0.0),
            )
            self._cache.set(cache_key, empty_answer)
            yield ("final", empty_answer)
            return

        messages = build_grounded_answer_messages(
            question=normalized_request.question, contexts=contexts
        )
        embedding_tokens = self._estimate_tokens(normalized_request.question)
        prompt_tokens = self._estimate_message_tokens(messages)

        start = perf_counter()
        chunks: List[str] = []
        for piece in stream_chat_completion(
            messages, model=LLM_MODEL, timeout=LLM_TIMEOUT_SECONDS
        ):
            if piece:
                chunks.append(piece)
                yield ("token", piece)

        answer_text = "".join(chunks).strip() or "No answer found."
        latency_ms = (perf_counter() - start) * 1000
        completion_tokens = self._estimate_tokens(answer_text)
        answer = RagAnswer(
            answer=answer_text,
            citations=citations,
            contexts=contexts,
            retrievals=retrievals,
            latency_ms=latency_ms,
            tokens=TokenUsage(
                prompt=prompt_tokens,
                completion=completion_tokens,
                embedding=embedding_tokens,
            ),
            costs=self._calculate_costs(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                embedding_tokens=embedding_tokens,
            ),
        )
        self._cache.set(cache_key, answer)
        yield ("final", answer)

    def _retrieve_records(
        self, request: RagRequest
    ) -> Tuple[List[SearchResult], List[ContextChunk], List[Citation]]:
        store = self._store_getter()
        embeddings = embed_texts_with_litellm(
            [request.question],
            model=EMBEDDINGS_MODEL,
            timeout=EMBEDDINGS_TIMEOUT_SECONDS,
        )
        if not embeddings:
            raise RuntimeError("Embeddings provider returned no vectors for the query.")
        query_embedding = embeddings[0]

        search_k = request.top_k if request.airline is None else request.top_k * 3
        results = store.search_by_embedding(query_embedding, top_k=search_k)
        if request.airline:
            normalized_airline = normalize_airline_key(request.airline)
            results = [
                result
                for result in results
                if normalize_airline_key(str(result.metadata.get("airline", "")))
                == normalized_airline
            ]
        trimmed = results[: request.top_k]
        contexts = _results_to_context_chunks(trimmed)
        citations = [_result_to_citation(result) for result in trimmed]
        return trimmed, contexts, citations

    def _calculate_costs(
        self, *, prompt_tokens: int, completion_tokens: int, embedding_tokens: int
    ) -> CostBreakdown:
        prompt_costs = MODEL_COST_LOOKUP.get(LLM_MODEL, {})
        embed_costs = MODEL_COST_LOOKUP.get(EMBEDDINGS_MODEL, {})
        prompt_usd = _cost_from_tokens(prompt_tokens, prompt_costs.get("prompt"))
        completion_usd = _cost_from_tokens(
            completion_tokens, prompt_costs.get("completion")
        )
        embedding_usd = _cost_from_tokens(embedding_tokens, embed_costs.get("prompt"))
        return CostBreakdown(
            prompt_usd=prompt_usd,
            completion_usd=completion_usd,
            embedding_usd=embedding_usd,
        )

    def _normalize_request(self, request: RagRequest) -> RagRequest:
        question = request.question.strip()
        airline = request.airline.strip() if request.airline else None
        top_k = max(1, min(8, request.top_k))
        return RagRequest(question=question, airline=airline, top_k=top_k)

    def _cache_key(self, request: RagRequest) -> str:
        normalized_question = " ".join(request.question.lower().split())
        airline_key = normalize_airline_key(request.airline) if request.airline else ""
        airline = airline_key or "*"
        return json.dumps(
            {
                "question": normalized_question,
                "airline": airline,
                "top_k": request.top_k,
            },
            sort_keys=True,
        )

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def _estimate_message_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        flattened = "\n\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        return self._estimate_tokens(flattened)


def _cost_from_tokens(tokens: int, rate_per_1k: Optional[float]) -> float:
    if not tokens or rate_per_1k is None:
        return 0.0
    return (tokens / 1000.0) * rate_per_1k


def _results_to_context_chunks(results: Sequence[SearchResult]) -> List[ContextChunk]:
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
