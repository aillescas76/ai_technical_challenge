from __future__ import annotations

import logging
from typing import Any, Iterable, List, Literal, Optional, Sequence, TypedDict

import litellm
from litellm.exceptions import AuthenticationError as LiteLLMAuthenticationError

from app.core.config import (
    EMBEDDINGS_MODEL,
    EMBEDDINGS_TIMEOUT_SECONDS,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MODEL_FALLBACKS,
    LLM_TIMEOUT_SECONDS,
)

try:
    import langfuse
    from langfuse.decorators import observe
except ImportError:
    langfuse = None

    # No-op decorator if langfuse is not present
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


def _update_trace_usage(response: Any) -> None:
    """Helper to update the current Langfuse observation with usage and cost metrics."""
    if not langfuse:
        return

    try:
        # Calculate cost
        # litellm.completion_cost usually handles ModelResponse or EmbeddingResponse if standard keys exist
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None

        # Extract usage
        usage = getattr(response, "usage", None)
        usage_details = {}
        if usage:
            # specific mapping for litellm usage object to langfuse
            # Litellm Usage object often has attributes matching OpenAI standard
            usage_details = {
                "input": getattr(usage, "prompt_tokens", 0),
                "output": getattr(usage, "completion_tokens", 0),
                "total": getattr(usage, "total_tokens", 0),
            }

        # Update observation
        # We use update_current_observation which finds the active generation created by @observe
        langfuse.update_current_observation(
            usage_details=usage_details,
            cost_details={"total": float(cost)} if cost is not None else None,
            model=getattr(response, "model", None),
        )
    except Exception:
        logger.warning(
            "Failed to update Langfuse trace with usage metrics", exc_info=True
        )


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    """Typed representation of a chat message compatible with LiteLLM."""

    role: Role
    content: str


class LiteLLMClient:
    """Small wrapper to avoid mutating LiteLLM's module-level configuration."""

    def __init__(self, *, api_base: Optional[str]) -> None:
        self._api_base = api_base

    def completion(self, **kwargs: Any) -> Any:
        return litellm.completion(**self._with_api_base(kwargs))

    async def acompletion(self, **kwargs: Any) -> Any:
        return await litellm.acompletion(**self._with_api_base(kwargs))

    def embedding(self, **kwargs: Any) -> Any:
        return litellm.embedding(**self._with_api_base(kwargs))

    async def aembedding(self, **kwargs: Any) -> Any:
        return await litellm.aembedding(**self._with_api_base(kwargs))

    def _with_api_base(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._api_base and "api_base" not in kwargs:
            # When using a custom API base (likely a proxy), default to OpenAI protocol
            # unless a provider is already explicitly set (e.g. via overrides).
            updates = {"api_base": self._api_base}
            if "custom_llm_provider" not in kwargs:
                updates["custom_llm_provider"] = "openai"
            return {**kwargs, **updates}
        return kwargs


_client = LiteLLMClient(api_base=LLM_BASE_URL)


def _get_embeddings_model_name(explicit_model: Optional[str]) -> str:
    """Return the model name to use for embeddings."""
    model_name = explicit_model or EMBEDDINGS_MODEL
    if not model_name:
        raise RuntimeError(
            "Embeddings model name is not configured. "
            "Set the EMBEDDINGS_MODEL environment variable."
        )
    return model_name


@observe(as_type="generation")
async def async_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    **kwargs: object,
) -> str:
    """Call LiteLLM for a non-streaming chat completion asynchronously."""
    candidate_models = _candidate_models(model)
    request_timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        response, _ = await _run_async_completion_with_fallbacks(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            timeout=request_timeout,
            extra_kwargs=dict(kwargs),
            candidate_models=candidate_models,
        )
    except Exception:
        logger.exception(
            "LiteLLM async chat completion failed after trying models: %s",
            candidate_models,
        )
        raise

    _update_trace_usage(response)
    return _extract_message_content(response)


@observe(as_type="generation")
async def async_stream_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    **kwargs: object,
) -> Any:
    """Call LiteLLM for a streaming chat completion asynchronously."""
    candidate_models = _candidate_models(model)
    request_timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        response_stream, _ = await _run_async_completion_with_fallbacks(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=request_timeout,
            extra_kwargs=dict(kwargs),
            candidate_models=candidate_models,
        )
    except Exception:
        logger.exception(
            "LiteLLM async streaming chat completion failed after trying models: %s",
            candidate_models,
        )
        raise

    async for chunk in response_stream:
        content_piece = _extract_delta_content(chunk)
        if content_piece:
            yield content_piece


@observe(as_type="generation")
def chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    **kwargs: object,
) -> str:
    """Call LiteLLM for a non-streaming chat completion and return the text content."""
    candidate_models = _candidate_models(model)
    request_timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        response, _ = _run_completion_with_fallbacks(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            timeout=request_timeout,
            extra_kwargs=dict(kwargs),
            candidate_models=candidate_models,
        )
    except Exception:
        logger.exception(
            "LiteLLM chat completion failed after trying models: %s", candidate_models
        )
        raise

    _update_trace_usage(response)
    return _extract_message_content(response)


@observe(as_type="generation")
def stream_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    **kwargs: object,
) -> Iterable[str]:
    """Call LiteLLM for a streaming chat completion and yield content chunks."""
    candidate_models = _candidate_models(model)
    request_timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        response_stream, _ = _run_completion_with_fallbacks(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=request_timeout,
            extra_kwargs=dict(kwargs),
            candidate_models=candidate_models,
        )
    except Exception:
        logger.exception(
            "LiteLLM streaming chat completion failed after trying models: %s",
            candidate_models,
        )
        raise

    for chunk in response_stream:
        content_piece = _extract_delta_content(chunk)
        if content_piece:
            yield content_piece


@observe(as_type="generation")
async def async_embed_texts_with_litellm(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[List[float]]:
    """Create embeddings for a batch of texts asynchronously using LiteLLM."""
    model_name = _get_embeddings_model_name(model)
    if not texts:
        return []

    try:
        request_timeout = timeout or EMBEDDINGS_TIMEOUT_SECONDS
        payload: dict[str, Any] = {
            "model": model_name,
            "input": list(texts),
            "timeout": request_timeout,
        }
        response = await _client.aembedding(**payload)
        _update_trace_usage(response)
    except Exception:
        logger.exception(
            "LiteLLM async embeddings request failed for model %s", model_name
        )
        raise

    data = _get_attribute(response, "data", default=[])
    return [
        [float(value) for value in embedding]
        for item in data
        if (embedding := _get_attribute(item, "embedding")) is not None
    ]


@observe(as_type="generation")
def embed_texts_with_litellm(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[List[float]]:
    """Create embeddings for a batch of texts using LiteLLM's embeddings API."""
    model_name = _get_embeddings_model_name(model)
    if not texts:
        return []

    try:
        request_timeout = timeout or EMBEDDINGS_TIMEOUT_SECONDS
        payload: dict[str, Any] = {
            "model": model_name,
            "input": list(texts),
            "timeout": request_timeout,
        }
        response = _client.embedding(**payload)
        _update_trace_usage(response)
    except Exception:
        logger.exception("LiteLLM embeddings request failed for model %s", model_name)
        raise

    data = _get_attribute(response, "data", default=[])
    return [
        [float(value) for value in embedding]
        for item in data
        if (embedding := _get_attribute(item, "embedding")) is not None
    ]


async def _run_async_completion(
    messages: Sequence[ChatMessage],
    *,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
    timeout: float,
    extra_kwargs: dict[str, object],
) -> Any:
    if "stream" in extra_kwargs:
        raise TypeError("stream argument is managed internally by app.llm")
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "timeout": timeout,
    }
    payload.update(extra_kwargs)
    return await _client.acompletion(**payload)


async def _run_async_completion_with_fallbacks(
    messages: Sequence[ChatMessage],
    *,
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
    timeout: float,
    extra_kwargs: dict[str, object],
    candidate_models: Sequence[str],
) -> tuple[Any, str]:
    last_auth_error: LiteLLMAuthenticationError | None = None
    for candidate in candidate_models:
        try:
            response = await _run_async_completion(
                messages,
                model_name=candidate,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                timeout=timeout,
                extra_kwargs=extra_kwargs,
            )
            return response, candidate
        except LiteLLMAuthenticationError as exc:
            last_auth_error = exc
            logger.warning(
                "LiteLLM async denied access to model %s, trying next fallback",
                candidate,
            )
    if last_auth_error is not None:
        raise last_auth_error
    raise RuntimeError("No LLM models available for async completion")


def _run_completion(
    messages: Sequence[ChatMessage],
    *,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
    timeout: float,
    extra_kwargs: dict[str, object],
) -> Any:
    if "stream" in extra_kwargs:
        raise TypeError("stream argument is managed internally by app.llm")
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "timeout": timeout,
    }
    payload.update(extra_kwargs)
    return _client.completion(**payload)


def _run_completion_with_fallbacks(
    messages: Sequence[ChatMessage],
    *,
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
    timeout: float,
    extra_kwargs: dict[str, object],
    candidate_models: Sequence[str],
) -> tuple[Any, str]:
    last_auth_error: LiteLLMAuthenticationError | None = None
    for candidate in candidate_models:
        try:
            response = _run_completion(
                messages,
                model_name=candidate,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                timeout=timeout,
                extra_kwargs=extra_kwargs,
            )
            return response, candidate
        except LiteLLMAuthenticationError as exc:
            last_auth_error = exc
            logger.warning(
                "LiteLLM denied access to model %s, trying next fallback", candidate
            )
    if last_auth_error is not None:
        raise last_auth_error
    raise RuntimeError("No LLM models available for completion")


def _extract_message_content(response: Any) -> str:
    choice = _first_choice(response)
    message = _get_attribute(choice, "message", default={})
    content = _get_attribute(message, "content")
    return str(content or "")


def _extract_delta_content(chunk: Any) -> Optional[str]:
    try:
        choice = _first_choice(chunk)
    except ValueError:
        logger.debug("LiteLLM chunk missing choices: %r", chunk)
        return None
    delta = _get_attribute(choice, "delta", default={})
    content_piece = _get_attribute(delta, "content")
    return str(content_piece) if content_piece else None


def _first_choice(response: Any) -> Any:
    choices = _get_attribute(response, "choices", default=())
    if not choices:
        raise ValueError("LiteLLM response contained no choices")
    return choices[0]


def _get_attribute(obj: Any, attribute: str, *, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attribute, default)
    return getattr(obj, attribute, default)


def _candidate_models(explicit_model: Optional[str]) -> List[str]:
    """Return ordered candidate models, ensuring at least one is configured."""
    models: List[str] = []
    seen: set[str] = set()
    for candidate in (explicit_model, LLM_MODEL, *LLM_MODEL_FALLBACKS):
        if candidate and candidate not in seen:
            models.append(candidate)
            seen.add(candidate)
    if not models:
        raise RuntimeError(
            "LLM model name is not configured. Set the LLM_MODEL environment variable."
        )
    return models
