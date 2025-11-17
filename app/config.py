from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent.parent

POLICIES_DIR: Path = BASE_DIR / "policies"
DATA_DIR: Path = BASE_DIR / "data"
PROCESSED_DOCS_PATH: Path = DATA_DIR / "processed.jsonl"

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


EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL", "http://localhost")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-5-haiku")
LLM_MODEL_FALLBACKS: tuple[str, ...] = _parse_model_list(
    "LLM_MODEL_FALLBACKS",
    ("gpt-5-mini", "gpt-4.1-mini", "gpt-4.1"),
)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
