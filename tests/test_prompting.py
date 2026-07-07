from electrical_rag.rag.prompting import RetrievedChunk, build_citations, build_prompt


def test_build_prompt_contains_question_and_source() -> None:
    chunks = [
        RetrievedChunk(
            text="IEEE 519 defines recommended harmonic current limits.",
            source="Standards/IEEE_519_Harmonic_standard.pdf",
            page=7,
            score=0.34,
        )
    ]

    prompt = build_prompt("What are harmonic current limits?", chunks)

    assert "What are harmonic current limits?" in prompt
    assert "Standards/IEEE_519_Harmonic_standard.pdf" in prompt
    assert "score=0.3400" in prompt


def test_build_prompt_instructs_exact_technical_fact_extraction() -> None:
    prompt = build_prompt("What is the power meter?", [])

    assert len(prompt) < 2200
    assert "Use only the context above." in prompt
    assert "Preserve exact protocol, interface" in prompt
    assert "Do not replace exact technical names or values" in prompt


def test_build_prompt_requires_answer_relevance() -> None:
    prompt = build_prompt("What communication protocols does protection relay support?", [])

    assert "include only facts needed to answer it" in prompt
    assert "Ignore unrelated details" in prompt
    assert "list only names that directly answer the question" in prompt


def test_build_prompt_distinguishes_protocols_from_interfaces() -> None:
    prompt = build_prompt("What communication protocols does protection relay support?", [])

    assert "distinguish protocols from physical interfaces" in prompt
    assert "only when they clarify how a protocol is transported" in prompt
    assert "Modbus RTU/TCP and BACnet MS/TP/IP are protocol examples" in prompt
    assert "RJ-11, RJ-12, and USB are interfaces or connectors, not protocols" in prompt
    assert "Never answer a protocol question with only interfaces" in prompt


def test_build_citations_deduplicates_same_source_and_page() -> None:
    chunks = [
        RetrievedChunk(text="A", source="doc-a.pdf", page=1, score=0.1),
        RetrievedChunk(text="B", source="doc-a.pdf", page=1, score=0.2),
        RetrievedChunk(text="C", source="doc-b.pdf", page=2, score=0.3),
    ]

    citations = build_citations(chunks, max_items=5)

    assert len(citations) == 2
    assert citations[0]["source"] == "doc-a.pdf"
    assert citations[1]["source"] == "doc-b.pdf"
