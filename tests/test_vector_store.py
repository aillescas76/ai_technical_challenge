from __future__ import annotations

from pathlib import Path

from app.components.vector_store import VectorStore


def test_vector_store_add_and_search_by_embedding() -> None:
    store = VectorStore(dimension=2)

    texts = ["cat", "dog", "airplane"]
    metadatas = [
        {"label": "cat"},
        {"label": "dog"},
        {"label": "airplane"},
    ]
    ids = ["cat", "dog", "airplane"]
    embeddings = [
        [1.0, 0.0],  # cat
        [0.0, 1.0],  # dog
        [0.7, 0.7],  # airplane (somewhere in between)
    ]

    store.add_embeddings(embeddings=embeddings, metadatas=metadatas, ids=ids, texts=texts)

    results = store.search_by_embedding([1.0, 0.0], top_k=2)

    assert results
    assert results[0].id == "cat"
    assert results[0].metadata["label"] == "cat"


def test_vector_store_persist_and_reload(tmp_path: Path) -> None:
    store = VectorStore(dimension=2)

    texts = ["cat", "dog"]
    metadatas = [{"label": "cat"}, {"label": "dog"}]
    ids = ["cat", "dog"]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add_embeddings(embeddings=embeddings, metadatas=metadatas, ids=ids, texts=texts)

    store_dir = tmp_path / "faiss"
    store.save(store_dir)

    loaded_store = VectorStore.load(store_dir)

    assert loaded_store.size == store.size

    results = loaded_store.search_by_embedding([0.0, 1.0], top_k=1)

    assert results
    assert results[0].id == "dog"
    assert results[0].metadata["label"] == "dog"

