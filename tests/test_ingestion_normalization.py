from electrical_rag.rag.text_normalization import normalize_ocr_technical_text


def test_normalize_ocr_technical_text_fixes_electrical_numeric_units() -> None:
    text = (
        "TRMS current measurement up to 60OOA "
        "Direct voltage measurement up to 60OV (L-L) "
        "Frequency up to 5OHz"
    )

    normalized = normalize_ocr_technical_text(text)

    assert "6000A" in normalized
    assert "600V" in normalized
    assert "50Hz" in normalized


def test_normalize_ocr_technical_text_keeps_normal_words_unchanged() -> None:
    text = "Power Quality and Voltage measurements are available."

    normalized = normalize_ocr_technical_text(text)

    assert normalized == text


def test_normalize_ocr_technical_text_handles_decimal_leading_o() -> None:
    text = "Power factor is O.95 PF and voltage is 23O V."

    normalized = normalize_ocr_technical_text(text)

    assert "0.95 PF" in normalized
    assert "230 V" in normalized
