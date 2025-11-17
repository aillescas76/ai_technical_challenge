from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pytest

from app import llm
from app.llm import ChatMessage
from app.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    ContextChunk,
    build_grounded_answer_messages,
    iter_chunk_citations,
)


class _FakeLiteLLMClient:
    def __init__(self) -> None:
        self.last_completion_payload: Dict[str, Any] | None = None
        self.last_embedding_payload: Dict[str, Any] | None = None
        self._stream_chunks: Iterable[Any] | None = None
        self._completion_response: Any | None = None
        self._embedding_response: Any | None = None

    def set_completion_response(self, response: Any) -> None:
        self._completion_response = response

    def set_stream_chunks(self, chunks: Iterable[Any]) -> None:
        self._stream_chunks = chunks

    def set_embedding_response(self, response: Any) -> None:
        self._embedding_response = response

    def completion(self, **payload: Any) -> Any:
        self.last_completion_payload = payload
        if payload.get("stream"):
            return self._stream_chunks
        return self._completion_response

    def embedding(self, **payload: Any) -> Any:
        self.last_embedding_payload = payload
        return self._embedding_response


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeLiteLLMClient:
    client = _FakeLiteLLMClient()
    monkeypatch.setattr(llm, "_client", client)
    return client


def test_chat_completion_returns_first_choice(fake_client: _FakeLiteLLMClient) -> None:
    fake_client.set_completion_response(
        {"choices": [{"message": {"content": "Grounded answer"}}]}
    )

    messages: List[ChatMessage] = [ChatMessage(role="user", content="Hello?")]
    result = llm.chat_completion(messages, temperature=0.0)

    assert result == "Grounded answer"
    assert fake_client.last_completion_payload is not None
    assert fake_client.last_completion_payload.get("stream") is False


def test_stream_chat_completion_yields_chunks(fake_client: _FakeLiteLLMClient) -> None:
    fake_client.set_stream_chunks(
        [
            {"choices": [{"delta": {"content": "Part"}}]},
            {"choices": [{"delta": {"content": "ial"}}]},
        ]
    )

    messages: List[ChatMessage] = [ChatMessage(role="user", content="Stream")]
    result = "".join(llm.stream_chat_completion(messages))

    assert result == "Partial"
    assert fake_client.last_completion_payload is not None
    assert fake_client.last_completion_payload.get("stream") is True


def test_embed_texts_with_litellm(fake_client: _FakeLiteLLMClient) -> None:
    fake_client.set_embedding_response(
        {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }
    )

    vectors = llm.embed_texts_with_litellm(["a", "b"], model="dummy")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_client.last_embedding_payload is not None
    assert fake_client.last_embedding_payload["model"] == "dummy"


def test_build_grounded_answer_messages_formats_context() -> None:
    contexts = [
        ContextChunk(
            content="Carry-on under 8kg",
            airline="SkyFly",
            title="Baggage",
            source_path="policies/skyfly/baggage.md",
            chunk_id="skyfly-bag-1",
            source_url="https://skyfly.com/baggage",
        )
    ]

    messages = build_grounded_answer_messages(question="What is the limit?", contexts=contexts)

    assert messages[0] == ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT.strip())
    user_content = messages[1]["content"]
    assert "SkyFly" in user_content
    assert "baggage.md" in user_content
    assert "Carry-on under 8kg" in user_content
    assert "No answer found" in user_content


def test_iter_chunk_citations_returns_unique_labels() -> None:
    contexts = [
        ContextChunk(
            content="",
            airline="SkyFly",
            title="Pets",
            source_path="policies/skyfly/pets.md",
        ),
        ContextChunk(
            content="",
            airline="SkyFly",
            title="Pets",
            source_path="policies/skyfly/pets.md",
        ),
        ContextChunk(
            content="",
            airline="Aero",
            title="Fees",
            source_path="policies/aero/fees.md",
        ),
    ]

    labels = list(iter_chunk_citations(contexts))

    assert labels == ["SkyFly – Pets", "Aero – Fees"]
