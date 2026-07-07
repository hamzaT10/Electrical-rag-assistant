from __future__ import annotations

import re

_TECHNICAL_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([0-9OoIlS]+(?:[.,][0-9OoIlS]+)?)"
    r"(\s*)"
    r"(kVA|kVAR|kVAr|kV|kA|kW|mA|MW|MVA|VA|VAR|VAr|Hz|V|A|W|%|PF)"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

_DECIMAL_VALUE_PATTERN = re.compile(r"(?<![A-Za-z0-9])([Oo])([.,][0-9]+)(?![A-Za-z0-9])")


def _normalize_number_like_text(value: str) -> str:
    return value.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"}))


def normalize_ocr_technical_text(text: str) -> str:
    """Fix OCR/PDF character confusion only inside technical numeric values."""

    def replace_technical_value(match: re.Match[str]) -> str:
        number, spacing, unit = match.groups()
        normalized_number = _normalize_number_like_text(number)
        return f"{normalized_number}{spacing}{unit}"

    normalized = _TECHNICAL_VALUE_PATTERN.sub(replace_technical_value, text)
    return _DECIMAL_VALUE_PATTERN.sub(r"0\2", normalized)
