from langchain_core.documents import Document

from electrical_rag.rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        assert pairs == [
            ("question", "less relevant"),
            ("question", "most relevant"),
        ]
        return [0.1, 0.9]


def test_cross_encoder_reranker_sorts_by_reranker_score() -> None:
    reranker = CrossEncoderReranker("fake-model")
    reranker._model = FakeCrossEncoder()
    low_doc = Document(page_content="less relevant", metadata={"source": "a.pdf"})
    high_doc = Document(page_content="most relevant", metadata={"source": "b.pdf"})

    results = reranker.rerank(
        "question",
        [
            (low_doc, 0.8),
            (high_doc, 0.2),
        ],
    )

    assert results == [(high_doc, 0.2), (low_doc, 0.8)]
    assert high_doc.metadata["reranker_score"] == 0.9
    assert low_doc.metadata["reranker_score"] == 0.1
