from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Incoming request payload for the /ask endpoint."""

    question: str = Field(
        ..., description="Natural-language policy question to answer."
    )
    top_k: int = Field(
        5,
        ge=1,
        le=8,
        description="Maximum number of chunks to retrieve for grounding.",
    )
    airline: Optional[str] = Field(
        default=None,
        description="Optional airline filter; restricts retrieval to a specific carrier.",
    )
    stream: bool = Field(
        default=False,
        description="When true, return a server-sent event stream of answer tokens.",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class Citation(BaseModel):
    """Grounding metadata returned with an answer."""

    id: str
    airline: str
    title: str
    source_path: str
    source_url: Optional[str] = None
    chunk_index: Optional[int] = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    """Response payload for grounded answers."""

    answer: str
    citations: List[Citation]


class HealthResponse(BaseModel):
    """Payload returned by /healthz for readiness probes."""

    status: str
    vector_store_size: int


class Metrics(BaseModel):
    requests_total: int
    requests_current: int
    errors_total: int
    uptime_seconds: float
