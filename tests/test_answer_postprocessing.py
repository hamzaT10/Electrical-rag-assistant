from electrical_rag.rag.answer_postprocessing import trim_incomplete_trailing_sentence


def test_trim_incomplete_trailing_sentence_removes_broken_tail() -> None:
    answer = "The data field contains register addresses and byte counts. In an"

    cleaned = trim_incomplete_trailing_sentence(answer)

    assert cleaned == "The data field contains register addresses and byte counts."


def test_trim_incomplete_trailing_sentence_keeps_complete_answer() -> None:
    answer = "The direct voltage measurement range is up to 600 V line-to-line."

    cleaned = trim_incomplete_trailing_sentence(answer)

    assert cleaned == answer


def test_trim_incomplete_trailing_sentence_keeps_answer_without_sentence_boundary() -> None:
    answer = "power meter"

    cleaned = trim_incomplete_trailing_sentence(answer)

    assert cleaned == answer
