from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from langchain_core.documents import Document

DEVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "power meter": ("power meter", "power analyzer", "energy meter"),
    "protection relay": ("protection relay", "relay module", "feeder relay"),
}

SOURCE_DEVICE_OVERRIDES: dict[str, str] = {}


def normalize_device_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def detect_query_devices(
    query: str,
    alias_map: Mapping[str, Iterable[str]] = DEVICE_ALIASES,
) -> list[str]:
    normalized_query = normalize_device_text(query)
    matches: list[str] = []

    for device_name, aliases in alias_map.items():
        normalized_aliases = [normalize_device_text(alias) for alias in aliases]
        if any(alias and alias in normalized_query for alias in normalized_aliases):
            matches.append(device_name)

    return matches


def infer_device_name_from_text(
    text: str,
    alias_map: Mapping[str, Iterable[str]] = DEVICE_ALIASES,
) -> str | None:
    matches = detect_query_devices(text, alias_map)
    if not matches:
        return None
    return matches[0]


def infer_device_name_from_source(
    source: str,
    alias_map: Mapping[str, Iterable[str]] = DEVICE_ALIASES,
) -> str | None:
    normalized_source = normalize_device_text(source)
    overridden_device = SOURCE_DEVICE_OVERRIDES.get(normalized_source)
    if overridden_device is not None:
        return overridden_device

    for device_name, aliases in alias_map.items():
        normalized_aliases = [normalize_device_text(alias) for alias in aliases]
        if any(alias and alias in normalized_source for alias in normalized_aliases):
            return device_name

    return None


def extract_document_device_name(doc: Document) -> str | None:
    metadata_value = doc.metadata.get("device_name")
    if isinstance(metadata_value, str) and metadata_value.strip():
        return metadata_value

    source = str(doc.metadata.get("source", ""))
    if source:
        inferred_from_source = infer_device_name_from_source(source)
        if inferred_from_source is not None:
            return inferred_from_source

    return infer_device_name_from_text(doc.page_content)


def rerank_results_by_devices(
    results: list[tuple[Document, float]],
    detected_devices: Iterable[str],
) -> list[tuple[Document, float]]:
    preferred_devices = set(detected_devices)
    if not preferred_devices:
        return results

    return sorted(
        results,
        key=lambda item: 0 if extract_document_device_name(item[0]) in preferred_devices else 1,
    )