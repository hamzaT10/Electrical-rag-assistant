from langchain_core.documents import Document

from electrical_rag.rag.device_metadata import (
    detect_query_devices,
    infer_device_name_from_source,
    infer_device_name_from_text,
    rerank_results_by_devices,
)


def test_detect_query_devices_handles_aliases() -> None:
    devices = detect_query_devices("What is the voltage range of the power meter analyzer?")

    assert devices == ["power meter"]


def test_infer_device_name_from_source_detects_known_device() -> None:
    device_name = infer_device_name_from_source("Standards/protection-relay-Eng (1).pdf")

    assert device_name == "protection relay"


def test_infer_device_name_from_text_detects_device_from_content() -> None:
    device_name = infer_device_name_from_text(
        "The power meter portable analyzer supports power quality measurements."
    )

    assert device_name == "power meter"


def test_rerank_results_by_devices_prioritizes_matching_device_documents() -> None:
    electrical_rag_doc = Document(
        page_content="power meter specifications",
        metadata={"source": "Standards/power meter_Eng-.pdf", "device_name": "power meter"},
    )
    generic_doc = Document(
        page_content="Generic voltage guidance",
        metadata={"source": "Standards/IEEE_519_Harmonic_standard.pdf"},
    )

    results = [(generic_doc, 0.1), (electrical_rag_doc, 0.2)]
    reranked = rerank_results_by_devices(results, ["power meter"])

    assert reranked[0][0].metadata["source"] == "Standards/power meter_Eng-.pdf"
    assert reranked[1][0].metadata["source"] == "Standards/IEEE_519_Harmonic_standard.pdf"


def test_rerank_results_by_devices_preserves_vector_store_order_within_groups() -> None:
    first_electrical_rag = Document(
        page_content="First power meter result",
        metadata={"source": "power meter_Eng-.pdf", "device_name": "power meter"},
    )
    second_electrical_rag = Document(
        page_content="Second power meter result",
        metadata={"source": "power meter_Eng-.pdf", "device_name": "power meter"},
    )

    reranked = rerank_results_by_devices(
        [(first_electrical_rag, 0.9), (second_electrical_rag, 0.7)],
        ["power meter"],
    )

    assert [item[1] for item in reranked] == [0.9, 0.7]
