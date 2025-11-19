from __future__ import annotations

from typing import List

import pytest

from app.components.vector_store import SearchResult
from app.services import rag
from app.services.rag import RagEngine, RagRequest


class _FakeVectorStore:
    def __init__(self, results: List[SearchResult]) -> None:
        self.results = results
        self.search_calls = 0

    @property
    def size(self) -> int:
        return len(self.results)

    def search_by_embedding(self, embedding, top_k: int) -> List[SearchResult]:
        self.search_calls += 1
        return self.results[:top_k]


def test_build_snippet_truncates() -> None:
    text = "Carry-on luggage must weigh under eight kilograms. " * 5
    snippet = rag._build_snippet(text, limit=60)
    assert snippet.endswith("...")
    assert len(snippet) <= 63


@pytest.mark.asyncio
async def test_rag_engine_caches_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        SearchResult(
            id="chunk-1",
            score=0.9,
            text="Carry-on bags under 8 kg.",
            metadata={
                "airline": "SkyFly",
                "title": "Baggage Rules",
                "source_path": "policies/skyfly/baggage.md",
                "chunk_index": 0,
            },
        )
    ]
    store = _FakeVectorStore(results)
    engine = RagEngine(lambda: store)

    # Mock async embeddings
    async def _fake_embed(texts, **_):
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(rag, "async_embed_texts_with_litellm", _fake_embed)

    call_counter = {"count": 0}

    # Mock async chat completion
    async def _fake_chat(messages, **kwargs):
        call_counter["count"] += 1
        return "Carry-on bags must fit under the seat."

    monkeypatch.setattr(rag, "async_chat_completion", _fake_chat)

    request = RagRequest(question="Carry-on?", top_k=1, airline=None)

    # Await the async answer method
    first = await engine.answer(request)
    second = await engine.answer(request)

    assert first.answer.startswith("Carry-on bags")
    assert second.from_cache
    assert call_counter["count"] == 1
