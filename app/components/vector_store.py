from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class VectorStore:
    """FAISS-based vector store supporting add, search, and persistence."""

    def __init__(self, dimension: int, index: Optional[faiss.Index] = None) -> None:
        self.dimension = dimension
        self.index = index or faiss.IndexFlatIP(dimension)
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return the number of vectors stored."""
        return len(self._ids)

    def add_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Add embeddings with associated metadata and optional IDs and texts."""
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must have the same length")

        num_vectors = len(embeddings)
        if ids is None:
            ids = [str(i + self.size) for i in range(num_vectors)]
        if texts is None:
            texts = ["" for _ in range(num_vectors)]

        if len(ids) != num_vectors or len(texts) != num_vectors:
            raise ValueError("ids, texts, embeddings, and metadatas must align")

        embeddings_array = np.asarray(embeddings, dtype="float32")
        if embeddings_array.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError("embedding dimension does not match index dimension")

        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)

        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(list(metadatas))
        return list(ids)

    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for nearest neighbors given a query embedding."""
        if self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError("query embedding dimension does not match index dimension")

        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= self.size:
                continue
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    text=self._texts[idx],
                    metadata=self._metadatas[idx],
                )
            )
        return results

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and metadata to the given directory."""
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.jsonl"

        faiss.write_index(self.index, str(index_path))

        with meta_path.open("w", encoding="utf-8") as f:
            for doc_id, text, metadata in zip(
                self._ids, self._texts, self._metadatas, strict=False
            ):
                record = {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                }
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a FAISS index and metadata from the given directory."""
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.jsonl"

        index = faiss.read_index(str(index_path))
        dimension = index.d

        store = cls(dimension=dimension, index=index)

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                store._ids.append(record["id"])
                store._texts.append(record.get("text", ""))
                store._metadatas.append(record.get("metadata", {}))

        if store.size != index.ntotal:
            raise ValueError("Loaded metadata count does not match index size")

        return store
