"""Prompt templates for grounded RAG answers with citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from app.components.llm import ChatMessage

DEFAULT_SYSTEM_PROMPT = """You are an airline policy specialist.
Use ONLY the provided policy excerpts to answer.
If the excerpts do not answer the question, reply with: "No answer found based on available policies." without additional text.
Write concise paragraphs or short bullet points.
Every factual statement must cite the airline and document title in brackets like [Airline – Document Title].
Never fabricate airlines, titles, or URLs.
"""


@dataclass(slots=True)
class ContextChunk:
    """Small container describing a retrieved policy chunk."""

    content: str
    airline: str
    title: str
    source_path: str
    chunk_id: str | None = None
    source_url: str | None = None


def build_grounded_answer_messages(
    *,
    question: str,
    contexts: Sequence[ContextChunk],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[ChatMessage]:
    """Return chat messages for answering a user question with grounded context."""

    context_block = _format_context_block(contexts)
    user_content = "\n\n".join(
        (
            f"Question:\n{question.strip()}" if question else "Question:\n",
            "Policy Excerpts:\n" + context_block,
            "Instructions:\n"
            "- Base the answer ONLY on the excerpts above.\n"
            "- Cite each statement with [Airline – Document Title].\n"
            '- If nothing answers the question, respond with "No answer found based on available policies."',
        )
    )
    return [
        ChatMessage(role="system", content=system_prompt.strip()),
        ChatMessage(role="user", content=user_content),
    ]


def _format_context_block(contexts: Sequence[ContextChunk]) -> str:
    if not contexts:
        return "No policy excerpts available."

    formatted_chunks: List[str] = []
    for index, chunk in enumerate(contexts, start=1):
        lines: List[str] = [
            f"[{index}] Airline: {chunk.airline}",
            f"Title: {chunk.title}",
            f"Source path: {chunk.source_path}",
        ]
        if chunk.chunk_id:
            lines.append(f"Chunk ID: {chunk.chunk_id}")
        if chunk.source_url:
            lines.append(f"Source URL: {chunk.source_url}")
        lines.append("Excerpt:\n" + chunk.content.strip())
        formatted_chunks.append("\n".join(lines).strip())
    return "\n\n".join(formatted_chunks)


def iter_chunk_citations(contexts: Sequence[ContextChunk]) -> Iterable[str]:
    """Return unique citation labels for the provided contexts."""

    seen: set[str] = set()
    for chunk in contexts:
        label = f"{chunk.airline} – {chunk.title}"
        if label not in seen:
            seen.add(label)
            yield label
