from __future__ import annotations

from typing import Iterator, List, Sequence

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.api.schemas import Citation
from app.services.prompt import ContextChunk
from app.services.rag import CostBreakdown, RagAnswer, RagRequest, TokenUsage


class _FakeEngine:
    def __init__(self, *, stream_events: Sequence) -> None:
        self._stream_events = list(stream_events)
        self.requests: List[RagRequest] = []
        self.answer_payload = _make_answer(
            "Carry-on bags must fit in the overhead bin."
        )

    async def answer(self, request: RagRequest) -> RagAnswer:
        self.requests.append(request)
        return self.answer_payload

    async def stream(self, request: RagRequest) -> Iterator:
        self.requests.append(request)
        for event in self._stream_events:
            yield event


def _make_answer(text: str) -> RagAnswer:
    citation = Citation(
        id="chunk-1",
        airline="SkyFly",
        title="Baggage Rules",
        source_path="policies/skyfly/baggage.md",
        source_url=None,
        chunk_index=0,
        score=0.9,
        snippet="Carry-on bags must weigh under 8 kg.",
    )
    context = ContextChunk(
        content="Carry-on bags must weigh under 8 kg.",
        airline="SkyFly",
        title="Baggage Rules",
        source_path="policies/skyfly/baggage.md",
        chunk_id="0",
        source_url=None,
    )
    return RagAnswer(
        answer=text,
        citations=[citation],
        contexts=[context],
        retrievals=[],
        latency_ms=12.5,
        tokens=TokenUsage(prompt=20, completion=10, embedding=5),
        costs=CostBreakdown(
            prompt_usd=0.0002, completion_usd=0.0004, embedding_usd=0.00005
        ),
    )


@pytest.fixture(autouse=True)
def _reset_server_state() -> Iterator[None]:
    original_store = server._vector_store
    original_error = server._vector_store_error
    original_engine = server._rag_engine
    yield
    server._vector_store = original_store
    server._vector_store_error = original_error
    server._rag_engine = original_engine


def test_ask_route_returns_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = _FakeEngine(stream_events=[])
    server._rag_engine = fake_engine  # type: ignore[assignment]

    with TestClient(server.app) as client:
        response = client.post("/ask", json={"question": "Carry-on limit?", "top_k": 4})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Carry-on bags must fit")
    assert len(payload["citations"]) == 1
    assert fake_engine.requests[-1].top_k == 4


def test_ask_route_streams_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        ("token", "Sky"),
        ("token", "Fly"),
        ("final", _make_answer("SkyFly allows one carry-on.")),
    ]
    fake_engine = _FakeEngine(stream_events=events)
    server._rag_engine = fake_engine  # type: ignore[assignment]

    with TestClient(server.app) as client:
        response = client.post("/ask", json={"question": "Carry-on?", "stream": True})

    assert response.status_code == 200
    body = response.text
    assert 'data: {"type": "token", "text": "Sky"}' in body
    assert '"type": "final"' in body
    assert '"metrics"' in body


def test_health_route_reports_error_status() -> None:
    with TestClient(server.app) as client:
        server._vector_store = None
        server._vector_store_error = "Vector index missing"

        response = client.get("/healthz")

    payload = response.json()
    assert payload["status"] == "error"
    assert payload["vector_store_size"] == 0
