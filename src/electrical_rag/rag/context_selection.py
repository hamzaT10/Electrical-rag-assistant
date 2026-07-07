from __future__ import annotations

from electrical_rag.rag.prompting import RetrievedChunk


def limit_chunks_per_source_page(
    chunks: list[RetrievedChunk],
    max_per_source_page: int,
) -> list[RetrievedChunk]:
    page_counts: dict[tuple[str, int | None], int] = {}
    selected_chunks: list[RetrievedChunk] = []

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        count = page_counts.get(key, 0)
        if count >= max_per_source_page:
            continue

        page_counts[key] = count + 1
        selected_chunks.append(chunk)

    return selected_chunks


def _trim_text_to_limit(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped
    if max_chars <= 3:
        return stripped[:max_chars]
    return stripped[: max_chars - 3].rstrip() + "..."


def apply_context_budget(
    chunks: list[RetrievedChunk],
    max_context_chars: int,
    max_chunk_chars: int,
) -> list[RetrievedChunk]:
    selected_chunks: list[RetrievedChunk] = []
    used_context_chars = 0

    for chunk in chunks:
        text = _trim_text_to_limit(chunk.text, max_chunk_chars)

        if max_context_chars > 0:
            remaining_chars = max_context_chars - used_context_chars
            if remaining_chars <= 0:
                break

            text = _trim_text_to_limit(text, remaining_chars)

        if not text:
            continue

        selected_chunks.append(
            RetrievedChunk(
                text=text,
                source=chunk.source,
                page=chunk.page,
                score=chunk.score,
            )
        )
        used_context_chars += len(text)

    return selected_chunks
