from __future__ import annotations

import asyncio
from typing import List, Sequence

import pytest
from fastapi.testclient import TestClient

from app import server
from app.vector_store import SearchResult


class _FakeVectorStore:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = list(results)
        self.size = len(self._results)
        self.search_calls: List[tuple[Sequence[float], int]] = []

    def search_by_embedding(self, embedding: Sequence[float], top_k: int) -> List[SearchResult]:
        self.search_calls.append((tuple(embedding), top_k))
        return list(self._results)


@pytest.fixture(autouse=True)
def _reset_vector_store_state() -> None:
    original_store = server._vector_store
    original_error = server._vector_store_error
    server._vector_store = None
    server._vector_store_error = None
    yield
    server._vector_store = original_store
    server._vector_store_error = original_error


def test_build_snippet_truncates_and_preserves_words() -> None:
    long_sentence = "Carry-on luggage must weigh under eight kilograms." * 10

    snippet = server._build_snippet(long_sentence, limit=80)

    assert snippet.endswith("...")
    assert len(snippet) <= 83
    assert "\n" not in snippet


def test_retrieve_context_applies_airline_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        SearchResult(
            id="chunk-a",
            score=0.9,
            text="Allowance notes",
            metadata={"airline": "SkyFly"},
        ),
        SearchResult(
            id="chunk-b",
            score=0.8,
            text="Pet policy",
            metadata={"airline": "AeroJet"},
        ),
    ]
    fake_store = _FakeVectorStore(results)
    server._vector_store = fake_store

    monkeypatch.setattr(server, "embed_texts_with_litellm", lambda texts: [[0.1, 0.2, 0.3]])

    filtered = server._retrieve_context("pets?", top_k=1, airline_filter="AEROJET")

    assert [result.id for result in filtered] == ["chunk-b"]
    assert fake_store.search_calls[-1][1] == 3  # multiplies top_k when filtering


def test_ask_route_returns_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        SearchResult(
            id="skyfly-0",
            score=0.87,
            text="Carry-on bags must weigh under 8 kg and fit overhead bins.",
            metadata={
                "airline": "SkyFly",
                "title": "Baggage Rules",
                "source_path": "policies/skyfly/baggage.md",
                "chunk_index": 0,
            },
        ),
        SearchResult(
            id="skyfly-1",
            score=0.81,
            text="Personal items under 2 kg are exempt from baggage fees.",
            metadata={
                "airline": "SkyFly",
                "title": "Baggage Rules",
                "source_path": "policies/skyfly/baggage.md",
                "chunk_index": 1,
            },
        ),
    ]
    fake_store = _FakeVectorStore(results)
    server._vector_store = fake_store

    monkeypatch.setattr(server, "embed_texts_with_litellm", lambda texts: [[0.01, 0.02, 0.03]])
    recorded_messages: dict[str, object] = {}

    def _fake_chat(messages: List[dict[str, str]]) -> str:
        recorded_messages["messages"] = messages
        return "SkyFly allows one carry-on under 8 kg. Sources: SkyFly – Baggage Rules."

    monkeypatch.setattr(server, "chat_completion", _fake_chat)

    with TestClient(server.app) as client:
        response = client.post("/ask", json={"question": "Carry-on limit?", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("SkyFly allows one carry-on")
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["airline"] == "SkyFly"
    assert payload["citations"][0]["source_path"].endswith("baggage.md")

    messages = recorded_messages["messages"]
    assert isinstance(messages, list)
    assert "SkyFly" in messages[1]["content"]


def test_health_route_reports_error_status() -> None:
    server._vector_store = None
    server._vector_store_error = "Vector index missing"

    response = asyncio.run(server.health_route())

    assert response.status == "error"
    assert response.vector_store_size == 0
