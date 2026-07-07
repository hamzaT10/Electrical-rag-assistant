from __future__ import annotations

import re

_SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\"')\]]*)")
_COMPLETE_END_PATTERN = re.compile(r"[.!?](?:[\"')\]]*)$")


def trim_incomplete_trailing_sentence(answer: str) -> str:
    """Remove a final broken sentence caused by max-token truncation."""
    stripped = answer.strip()
    if not stripped or _COMPLETE_END_PATTERN.search(stripped):
        return stripped

    matches = list(_SENTENCE_END_PATTERN.finditer(stripped))
    if not matches:
        return stripped

    return stripped[: matches[-1].end()].rstrip()
