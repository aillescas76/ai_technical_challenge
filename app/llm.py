from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Literal, Optional, Sequence, TypedDict

import litellm

from app.config import EMBEDDINGS_MODEL, LLM_MODEL


logger = logging.getLogger(__name__)


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    """Typed representation of a chat message compatible with LiteLLM."""

    role: Role
    content: str


def _get_model_name(explicit_model: Optional[str]) -> str:
    """Return the model name to use for chat completions."""
    model_name = explicit_model or LLM_MODEL
    if not model_name:
        raise RuntimeError(
            "LLM model name is not configured. Set the LLM_MODEL environment variable."
        )
    return model_name


def _get_embeddings_model_name(explicit_model: Optional[str]) -> str:
    """Return the model name to use for embeddings."""
    model_name = explicit_model or EMBEDDINGS_MODEL
    if not model_name:
        raise RuntimeError(
            "Embeddings model name is not configured. "
            "Set the EMBEDDINGS_MODEL environment variable."
        )
    return model_name


def chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: object,
) -> str:
    """Call LiteLLM for a non-streaming chat completion and return the text content."""
    model_name = _get_model_name(model)
    try:
        response = litellm.completion(
            model=model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
    except Exception:
        logger.exception("LiteLLM chat completion failed for model %s", model_name)
        raise

    message = response["choices"][0]["message"]
    return message.get("content", "") or ""


def stream_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: object,
) -> Iterable[str]:
    """Call LiteLLM for a streaming chat completion and yield content chunks."""
    model_name = _get_model_name(model)
    try:
        response_stream = litellm.completion(
            model=model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
    except Exception:
        logger.exception("LiteLLM streaming chat completion failed for model %s", model_name)
        raise

    for chunk in response_stream:
        try:
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            content_piece = delta.get("content")
            if content_piece:
                yield content_piece
        except (KeyError, IndexError, TypeError):
            logger.debug("Unexpected streaming chunk format from LiteLLM: %r", chunk)
            continue


def embed_texts_with_litellm(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
) -> List[List[float]]:
    """Create embeddings for a batch of texts using LiteLLM's embeddings API."""
    model_name = _get_embeddings_model_name(model)
    if not texts:
        return []

    try:
        response = litellm.embedding(model=model_name, input=list(texts))
    except Exception:
        logger.exception("LiteLLM embeddings request failed for model %s", model_name)
        raise

    data = getattr(response, "data", None)
    if data is None:
        data = response.get("data", [])

    embeddings: List[List[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None:
            embedding = item.get("embedding")
        if embedding is None:
            continue
        embeddings.append(list(map(float, embedding)))

    return embeddings

