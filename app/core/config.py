from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

POLICIES_DIR: Path = BASE_DIR / "policies"
DATA_DIR: Path = BASE_DIR / "data"
PROCESSED_DOCS_PATH: Path = DATA_DIR / "processed.jsonl"

EVAL_DATASET_PATH: Path = Path(
    os.getenv("EVAL_DATASET_PATH", str(BASE_DIR / "docs" / "evals" / "questions.jsonl"))
)

_default_vector_store_dir = DATA_DIR / "faiss"
VECTOR_STORE_PATH: Path = Path(
    os.getenv("VECTOR_STORE_PATH", str(_default_vector_store_dir))
)


def _parse_model_list(env_var: str, default: Sequence[str]) -> tuple[str, ...]:
    raw_value = os.getenv(env_var)
    if raw_value:
        cleaned = [item.strip() for item in raw_value.split(",")]
        parsed = [item for item in cleaned if item]
        if parsed:
            return tuple(parsed)
    return tuple(default)


def _parse_provider_overrides(env_var: str) -> Dict[str, str]:
    raw_value = os.getenv(env_var)
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = None
    overrides: Dict[str, str] = {}
    if isinstance(parsed, dict):
        for model_name, provider_name in parsed.items():
            model = str(model_name).strip()
            provider = str(provider_name).strip()
            if model and provider:
                overrides[model] = provider
        if overrides:
            return overrides
    for entry in raw_value.split(","):
        if ":" not in entry:
            continue
        model, provider = entry.split(":", 1)
        model = model.strip()
        provider = provider.strip()
        if model and provider:
            overrides[model] = provider
    return overrides


EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL") or None
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_MODEL_FALLBACKS: tuple[str, ...] = _parse_model_list(
    "LLM_MODEL_FALLBACKS",
    ("gpt-5-mini", "gpt-4.1"),
)
LLM_PROVIDER_OVERRIDES: Dict[str, str] = _parse_provider_overrides(
    "LLM_PROVIDER_OVERRIDES"
)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
EMBEDDINGS_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDINGS_TIMEOUT_SECONDS", "30"))

TOKEN_ENCODING_NAME: str = os.getenv("TOKEN_ENCODING_NAME", "cl100k_base")

ASK_CACHE_MAX_ITEMS: int = int(os.getenv("ASK_CACHE_MAX_ITEMS", "128"))
ASK_CACHE_TTL_SECONDS: int = int(os.getenv("ASK_CACHE_TTL_SECONDS", "600"))

RATE_LIMIT_TTL_SECONDS: int = int(os.getenv("RATE_LIMIT_TTL_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str | None = os.getenv("LANGFUSE_HOST")

_DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"prompt": 0.00000025, "completion": 0.000002},
    "gpt-4.1-mini": {"prompt": 0.0000004, "completion": 0.0000016},
    "gpt-4.1": {"prompt": 0.000002, "completion": 0.000008},
    "text-embedding-3-small": {"prompt": 0.00000002},
    "claude-3-5-haiku": {"prompt": 0.0000008, "completion": 0.000004}
}


def _load_model_costs() -> Dict[str, Dict[str, float]]:
    raw = os.getenv("MODEL_COST_OVERRIDES")
    if not raw:
        return _DEFAULT_MODEL_COSTS
    try:
        overrides: Dict[str, Dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError:
        return _DEFAULT_MODEL_COSTS
    merged: Dict[str, Dict[str, float]] = dict(_DEFAULT_MODEL_COSTS)
    for model, values in overrides.items():
        costs: Dict[str, float] = {}
        for cost_key, cost_value in values.items():
            try:
                costs[str(cost_key)] = float(cost_value)
            except (TypeError, ValueError):
                continue
        if costs:
            merged[model] = costs
    return merged


MODEL_COST_LOOKUP: Dict[str, Dict[str, float]] = _load_model_costs()
