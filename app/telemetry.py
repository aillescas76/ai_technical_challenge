from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class Telemetry:
    """
    Central telemetry handler for LangFuse.
    Gracefully handles missing credentials or dependencies by performing no-ops.
    """

    _instance: Optional[Telemetry] = None

    def __init__(self) -> None:
        self._client = self._build_client()

    @classmethod
    def get_instance(cls) -> Telemetry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_client(self) -> Optional["Langfuse"]:
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

    @property
    def client(self) -> Optional["Langfuse"]:
        return self._client

    def is_enabled(self) -> bool:
        return self._client is not None

    def trace(self, **kwargs: Any) -> Any:
        """Create a trace if enabled, otherwise return a dummy object."""
        if self._client:
            try:
                return self._client.trace(**kwargs)
            except Exception:
                logger.exception("Error creating trace")
        return _DummyTrace()

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass


class _DummyTrace:
    """No-op trace object for when telemetry is disabled."""
    def span(self, **kwargs: Any) -> _DummyTrace:
        return self
    
    def event(self, **kwargs: Any) -> _DummyTrace:
        return self
        
    def generation(self, **kwargs: Any) -> _DummyTrace:
        return self
        
    def update(self, **kwargs: Any) -> None:
        pass
    
    def score(self, **kwargs: Any) -> None:
        pass
    
    def __enter__(self) -> _DummyTrace:
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class LangfuseReporter:
    """
    Legacy wrapper for evaluation reporting.
    """

    def __init__(self) -> None:
        self._telemetry = Telemetry.get_instance()

    def is_enabled(self) -> bool:
        return self._telemetry.is_enabled()

    def log_eval(
        self,
        *,
        run_name: str,
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        if not self.is_enabled():
            return
        
        # Use the centralized telemetry client
        try:
            trace = self._telemetry.trace(
                name=run_name,
                input=input_payload,
                output=output_payload,
                metadata=metrics,
                tags=tags or [],
            )
            trace.event(name="eval-metrics", metadata=metrics)
            self._telemetry.flush()
        except Exception:  # pragma: no cover
            logger.exception("Failed to send LangFuse trace")
