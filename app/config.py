from __future__ import annotations

import os
from pathlib import Path

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

EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL", "http://localhost")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
