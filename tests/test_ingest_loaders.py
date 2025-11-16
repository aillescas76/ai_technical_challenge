from __future__ import annotations

from pathlib import Path

from app.ingest import (
    DocumentMetadata,
    RawDocument,
    chunk_document,
    get_token_encoder,
    load_markdown_document,
    load_pdf_document,
)


def test_markdown_loader_returns_text_and_metadata() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "policies" / "AmericanAirlines" / "Checked bag policy.md"
    relative_path = str(path.relative_to(base_dir))

    document = load_markdown_document(path, relative_path)

    assert document.text.strip()
    assert document.metadata.airline == "American Airlines"
    assert document.metadata.title == "Checked bag policy"
    assert document.metadata.category == "baggage"
    assert document.metadata.source_path == relative_path
    if document.metadata.source_url is not None:
        assert document.metadata.source_url.startswith("http")


def test_pdf_loader_returns_text_and_metadata() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "policies" / "United" / "Checked bags.pdf"
    relative_path = str(path.relative_to(base_dir))

    document = load_pdf_document(path, relative_path)

    assert document.text.strip()
    assert document.metadata.airline == "United Airlines"
    assert document.metadata.title == "Checked bags"
    assert document.metadata.category == "baggage"
    assert document.metadata.source_path == relative_path
    if document.metadata.source_url is not None:
        assert document.metadata.source_url.startswith("http")


def test_chunking_respects_token_limits_and_overlap() -> None:
    encoding = get_token_encoder()
    base_text = "hello world "
    text = base_text * 100

    metadata = DocumentMetadata(
        airline="Test Airline",
        title="Test Document",
        category="test",
        source_path="policies/Test/Test Document.md",
        source_url=None,
    )
    document = RawDocument(text=text, metadata=metadata)

    chunk_size_tokens = 50
    overlap_tokens = 10

    chunks = chunk_document(
        document=document,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        encoding=encoding,
    )

    assert len(chunks) > 1

    first_chunk_tokens = encoding.encode(chunks[0].text)
    second_chunk_tokens = encoding.encode(chunks[1].text)

    assert len(first_chunk_tokens) <= chunk_size_tokens
    assert len(second_chunk_tokens) <= chunk_size_tokens

    assert first_chunk_tokens[-overlap_tokens:] == second_chunk_tokens[:overlap_tokens]

    for idx, chunk in enumerate(chunks):
        assert chunk.airline == metadata.airline
        assert chunk.title == metadata.title
        assert chunk.category == metadata.category
        assert chunk.source_path == metadata.source_path
        assert chunk.source_url == metadata.source_url
        assert chunk.chunk_index == idx

