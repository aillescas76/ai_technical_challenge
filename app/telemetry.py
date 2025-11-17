from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class LangfuseReporter:
    """Thin wrapper that emits eval traces to LangFuse when credentials are present."""

    def __init__(self) -> None:
        self._client = _build_client()

    def is_enabled(self) -> bool:
        return self._client is not None

    def log_eval(
        self,
        *,
        run_name: str,
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        client = self._client
        if not client:
            return
        try:
            trace = client.trace(  # type: ignore[call-arg]
                name=run_name,
                input=input_payload,
                output=output_payload,
                metadata=metrics,
            )
            if hasattr(trace, "event"):
                trace.event(name="eval-metrics", metadata=metrics)
            client.flush()
        except Exception:  # pragma: no cover - network failures
            logger.exception("Failed to send LangFuse trace")


def _build_client() -> Optional["Langfuse"]:
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return None
    if Langfuse is None:
        logger.debug("LangFuse SDK not installed; telemetry disabled.")
        return None
    try:
        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception:  # pragma: no cover
        logger.exception("Failed to initialize LangFuse client")
        return None
