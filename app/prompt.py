from __future__ import annotations

from typing import Iterable, Sequence

from app.llm import ChatMessage
from app.vector_store import SearchResult


SYSTEM_PROMPT = """You are an assistant that answers questions about airline policies.
You must follow these rules:
- Use only the facts contained in the supplied context sections.
- Cite supporting evidence inline as [Airline – Title] after each relevant sentence.
- If the context is empty or does not contain the answer, respond with "No answer found."
- Keep answers short (2–4 sentences or concise bullets) and avoid speculation.
- Prefer the airline name and document title exactly as provided in the context metadata."""


def format_context_sections(results: Sequence[SearchResult]) -> str:
    """Convert search results into a numbered context block for prompting."""
    sections = []
    for idx, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        airline = metadata.get("airline", "Unknown Airline")
        title = metadata.get("title", "Unknown Document")
        chunk_index = metadata.get("chunk_index")
        header_parts = [f"[{idx}] Airline: {airline}", f"Title: {title}"]
        if chunk_index is not None:
            header_parts.append(f"Chunk: {chunk_index}")
        header = " | ".join(header_parts)
        body = result.text.strip()
        if not body:
            continue
        section_text = f"{header}\n{body}"
        sections.append(section_text)
    return "\n\n".join(sections)


def build_rag_messages(question: str, results: Sequence[SearchResult]) -> list[ChatMessage]:
    """Create chat messages for the RAG interaction with the LLM."""
    context = format_context_sections(results)
    if not context:
        context = "No context available."
    user_prompt = (
        "Use the context to answer the airline policy question.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def summarize_context_for_logging(results: Iterable[SearchResult]) -> str:
    """Return a compact summary of context used, suitable for logs."""
    parts = []
    for result in results:
        metadata = result.metadata or {}
        airline = metadata.get("airline", "Unknown")
        title = metadata.get("title", "Unknown")
        parts.append(f"{airline} – {title}")
    return ", ".join(parts)
