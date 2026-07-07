from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float


def _normalize_snippet(text: str, limit: int = 260) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page_label = chunk.page if chunk.page is not None else "unknown"
        score_label = f"{float(chunk.score):.4f}"
        context_blocks.append(
            f"[Doc {index} | source={chunk.source} | page={page_label} | "
            f"score={score_label}]\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    return dedent(
        f"""
        You are a technical assistant for electrical engineering documents.

        Context:
        {context}

        Question:
        {question}

        Rules:
        - Use only the context above.
        - Answer in the user's requested language, otherwise use the question language.
        - Translate relevant facts when the context language is different.
        - Answer directly, concisely, and professionally.
        - Identify exactly what the question requests and include only facts needed to answer it.
        - Ignore unrelated details, even when they appear in the context.
        - Preserve exact protocol, interface, model, unit, range, number, and sensor names.
        - For requested technical names, list only names that directly answer the question.
        - For protocol questions, distinguish protocols from physical interfaces.
        - Mention interfaces only when they clarify how a protocol is transported.
        - Modbus RTU/TCP and BACnet MS/TP/IP are protocol examples.
        - RS-485, Ethernet, RJ-11, RJ-12, and USB are interfaces or connectors, not protocols.
        - Never answer a protocol question with only interfaces, connectors, or unrelated features.
        - Do not replace exact technical names or values with vague wording.
        - Do not say a fact is unavailable if it appears in any context block.
        - Prefer the most specific context block over broad or generic context.
        - Start directly; use short bullets only when several facts are required.
        - Include necessary technical facts, but remove repetition and unsupported inference.
        - Synthesize across chunks; do not answer chunk by chunk.
        - Do not mention sources, citations, context, retrieval, or internal process.
        - If the answer is not supported, say it is not available in the documents.

        """
    ).strip()


def build_citations(chunks: list[RetrievedChunk], max_items: int = 5) -> list[dict[str, object]]:
    citations: list[dict[str, object]] = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "score": round(float(chunk.score), 4),
                "snippet": _normalize_snippet(chunk.text),
            }
        )
        if len(citations) >= max_items:
            break

    return citations
