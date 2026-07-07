from electrical_rag.core.settings import Settings
from electrical_rag.observability.tracing import TraceManager
from electrical_rag.rag.context_selection import apply_context_budget, limit_chunks_per_source_page
from electrical_rag.rag.prompting import RetrievedChunk
from electrical_rag.services.qa_service import NO_RELEVANT_CONTEXT_ANSWER, RAGService


class EmptyRetriever:
    def search(self, query: str, k: int, document_id: int | None = None):
        return []


class FailingLLM:
    def ask(self, prompt: str) -> str:
        raise AssertionError("LLM should not be called when retrieval returns no context")

    def stream(self, prompt: str):
        raise AssertionError("LLM should not stream when retrieval returns no context")


def _service_with_empty_retriever() -> RAGService:
    service = RAGService.__new__(RAGService)
    service.settings = Settings(rag_min_retrieval_score=0.3)
    service.tracing = TraceManager(Settings(enable_langfuse=False))
    service.retriever = EmptyRetriever()
    service.llm = FailingLLM()
    return service


def test_qa_service_returns_fallback_when_retrieval_has_no_context() -> None:
    service = _service_with_empty_retriever()

    answer, citations = service.ask("Who won the World Cup?")

    assert answer == NO_RELEVANT_CONTEXT_ANSWER
    assert citations == []


def test_qa_service_streams_fallback_when_retrieval_has_no_context() -> None:
    service = _service_with_empty_retriever()

    events = list(service.ask_stream("Who won the World Cup?"))

    assert events == [
        {"type": "token", "content": NO_RELEVANT_CONTEXT_ANSWER},
        {"type": "citations", "citations": []},
    ]


def test_limit_chunks_per_source_page_keeps_bounded_context_in_order() -> None:
    chunks = [
        RetrievedChunk(text="A", source="power-meter.pdf", page=1, score=0.1),
        RetrievedChunk(text="B", source="power-meter.pdf", page=1, score=0.2),
        RetrievedChunk(text="C", source="power-meter.pdf", page=1, score=0.3),
        RetrievedChunk(text="D", source="power-meter.pdf", page=2, score=0.4),
        RetrievedChunk(text="E", source="manual.pdf", page=1, score=0.5),
    ]

    selected = limit_chunks_per_source_page(chunks, max_per_source_page=2)

    assert [chunk.text for chunk in selected] == ["A", "B", "D", "E"]


def test_apply_context_budget_trims_each_chunk() -> None:
    chunks = [
        RetrievedChunk(text="A" * 20, source="power-meter.pdf", page=1, score=0.1),
        RetrievedChunk(text="B" * 8, source="power-meter.pdf", page=2, score=0.2),
    ]

    selected = apply_context_budget(
        chunks,
        max_context_chars=0,
        max_chunk_chars=10,
    )

    assert [chunk.text for chunk in selected] == ["A" * 7 + "...", "B" * 8]
    assert selected[0].source == "power-meter.pdf"
    assert selected[0].page == 1


def test_apply_context_budget_respects_total_context_limit() -> None:
    chunks = [
        RetrievedChunk(text="A" * 10, source="a.pdf", page=1, score=0.1),
        RetrievedChunk(text="B" * 10, source="b.pdf", page=1, score=0.2),
        RetrievedChunk(text="C" * 10, source="c.pdf", page=1, score=0.3),
    ]

    selected = apply_context_budget(
        chunks,
        max_context_chars=15,
        max_chunk_chars=0,
    )

    assert [chunk.text for chunk in selected] == ["A" * 10, "BB..."]
    assert sum(len(chunk.text) for chunk in selected) <= 15
