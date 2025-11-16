from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import tiktoken
from pypdf import PdfReader

from app.config import EMBEDDINGS_MODEL, POLICIES_DIR, PROCESSED_DOCS_PATH, VECTOR_STORE_PATH
from app.llm import embed_texts_with_litellm
from app.vector_store import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    airline: str
    title: str
    category: Optional[str]
    source_path: str
    source_url: Optional[str] = None


@dataclass
class RawDocument:
    text: str
    metadata: DocumentMetadata


@dataclass
class DocumentChunk:
    id: str
    text: str
    airline: str
    title: str
    category: Optional[str]
    source_path: str
    source_url: Optional[str]
    chunk_index: int

    def to_record(self) -> dict:
        """Serialize chunk to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "airline": self.airline,
            "title": self.title,
            "category": self.category,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
        }

    def to_metadata(self) -> dict:
        """Return metadata dictionary for indexing."""
        return {
            "id": self.id,
            "airline": self.airline,
            "title": self.title,
            "category": self.category,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
        }


AIRLINE_NAME_MAP = {
    "AmericanAirlines": "American Airlines",
    "Delta": "Delta Air Lines",
    "United": "United Airlines",
}


URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")


def iter_policy_files(policies_dir: Path) -> Iterable[Path]:
    """Yield all Markdown and PDF policy files under the given directory."""
    for path in sorted(policies_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".md", ".pdf"}:
            yield path


def infer_airline_from_path(path: Path) -> str:
    """Infer normalized airline name from a policy file path."""
    airline_key = path.parent.name
    return AIRLINE_NAME_MAP.get(airline_key, airline_key)


def infer_title_from_path(path: Path) -> str:
    """Infer document title from filename without extension."""
    return path.stem.strip()


def infer_category_from_title(title: str) -> Optional[str]:
    """Derive a coarse category from the document title."""
    lower = title.lower()
    if "bag" in lower or "baggage" in lower or "fee" in lower:
        return "baggage"
    if "pet" in lower or "pets" in lower:
        return "pets"
    if "child" in lower or "infant" in lower or "kids" in lower or "family" in lower:
        return "children"
    if "pregnan" in lower:
        return "pregnancy"
    if "special" in lower:
        return "special_circumstances"
    return None


def extract_first_url(text: str) -> Optional[str]:
    """Extract the first HTTP(S) URL from the given text, if any."""
    match = URL_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def extract_pdf_text_and_links(path: Path) -> tuple[str, Optional[str]]:
    """Extract text and first link from a PDF file."""
    reader = PdfReader(str(path))
    texts: List[str] = []
    links: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
        # Best-effort link extraction from annotations.
        try:
            if "/Annots" in page:
                for annot_ref in page["/Annots"]:
                    annot = annot_ref.get_object()
                    action = annot.get("/A")
                    if action is None:
                        continue
                    uri = action.get("/URI")
                    if isinstance(uri, str):
                        links.append(uri)
        except Exception:  # pragma: no cover - defensive for PDF quirks
            continue

    full_text = "\n\n".join(t for t in texts if t)

    source_url = links[0] if links else extract_first_url(full_text)
    return full_text, source_url


def load_markdown_document(path: Path, relative_path: str) -> RawDocument:
    """Load a Markdown policy document into a RawDocument."""
    airline = infer_airline_from_path(path)
    title = infer_title_from_path(path)
    category = infer_category_from_title(title)

    text = path.read_text(encoding="utf-8")
    source_url = extract_first_url(text)

    metadata = DocumentMetadata(
        airline=airline,
        title=title,
        category=category,
        source_path=relative_path,
        source_url=source_url,
    )
    return RawDocument(text=text, metadata=metadata)


def load_pdf_document(path: Path, relative_path: str) -> RawDocument:
    """Load a PDF policy document into a RawDocument."""
    airline = infer_airline_from_path(path)
    title = infer_title_from_path(path)
    category = infer_category_from_title(title)

    text, source_url = extract_pdf_text_and_links(path)

    metadata = DocumentMetadata(
        airline=airline,
        title=title,
        category=category,
        source_path=relative_path,
        source_url=source_url,
    )
    return RawDocument(text=text, metadata=metadata)


def load_documents(policies_dir: Path = POLICIES_DIR) -> List[RawDocument]:
    """Load all policy documents from Markdown and PDF sources."""
    documents: List[RawDocument] = []
    base_dir = policies_dir.parent

    for path in iter_policy_files(policies_dir):
        relative_path = str(path.relative_to(base_dir))
        if path.suffix.lower() == ".md":
            document = load_markdown_document(path, relative_path)
        elif path.suffix.lower() == ".pdf":
            document = load_pdf_document(path, relative_path)
        else:
            continue
        documents.append(document)

    return documents


def get_token_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Return a tiktoken encoder for the given encoding name."""
    return tiktoken.get_encoding(encoding_name)


def chunk_document(
    document: RawDocument,
    chunk_size_tokens: int = 900,
    overlap_tokens: int = 150,
    encoding: Optional[tiktoken.Encoding] = None,
) -> List[DocumentChunk]:
    """Chunk a single document into overlapping token-based chunks."""
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    encoding = encoding or get_token_encoder()
    tokens = encoding.encode(document.text)

    chunks: List[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size_tokens)
        token_slice = tokens[start:end]
        text_chunk = encoding.decode(token_slice)

        chunk_id = f"{document.metadata.airline}:{document.metadata.title}:{chunk_index}"

        chunk = DocumentChunk(
            id=chunk_id,
            text=text_chunk,
            airline=document.metadata.airline,
            title=document.metadata.title,
            category=document.metadata.category,
            source_path=document.metadata.source_path,
            source_url=document.metadata.source_url,
            chunk_index=chunk_index,
        )
        chunks.append(chunk)

        if end >= len(tokens):
            break

        start = max(0, end - overlap_tokens)
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: Sequence[RawDocument],
    chunk_size_tokens: int = 900,
    overlap_tokens: int = 150,
    encoding_name: str = "cl100k_base",
) -> List[DocumentChunk]:
    """Chunk all documents into overlapping token-based chunks."""
    encoding = get_token_encoder(encoding_name)
    all_chunks: List[DocumentChunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
                encoding=encoding,
            )
        )
    return all_chunks


def write_processed_chunks(chunks: Sequence[DocumentChunk], path: Path) -> None:
    """Write processed document chunks to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = chunk.to_record()
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def embed_texts(texts: Sequence[str], model: str = EMBEDDINGS_MODEL) -> List[List[float]]:
    """Create embeddings for texts using the LiteLLM embeddings API."""
    return embed_texts_with_litellm(texts, model=model)


def build_vector_index(
    chunks: Sequence[DocumentChunk],
    vector_store_dir: Path = VECTOR_STORE_PATH,
) -> None:
    """Build and persist a FAISS vector index from document chunks."""
    if not chunks:
        logger.warning("No chunks provided; skipping index build.")
        return

    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.to_metadata() for chunk in chunks]
    ids = [chunk.id for chunk in chunks]

    embeddings = embed_texts(texts)
    if not embeddings:
        logger.warning("Received no embeddings; skipping index build.")
        return

    dim = len(embeddings[0])

    store = VectorStore(dimension=dim)
    store.add_embeddings(embeddings=embeddings, metadatas=metadatas, ids=ids, texts=texts)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    store.save(vector_store_dir)


def run_ingestion() -> None:
    """Run the full ingestion pipeline: load, chunk, persist, and index."""
    logger.info("Loading policy documents from %s", POLICIES_DIR)
    documents = load_documents(POLICIES_DIR)
    logger.info("Loaded %d documents", len(documents))

    logger.info("Chunking documents")
    chunks = chunk_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Writing processed chunks to %s", PROCESSED_DOCS_PATH)
    write_processed_chunks(chunks, PROCESSED_DOCS_PATH)

    logger.info("Building vector index in %s", VECTOR_STORE_PATH)
    build_vector_index(chunks, VECTOR_STORE_PATH)
    logger.info("Ingestion completed.")


def main() -> None:
    """CLI entry point for rebuilding processed data and index."""
    logging.basicConfig(level=logging.INFO)
    run_ingestion()


if __name__ == "__main__":
    main()
