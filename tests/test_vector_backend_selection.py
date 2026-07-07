from __future__ import annotations

import pytest
from langchain_core.documents import Document

import electrical_rag.rag.ingestion as ingestion_module
import electrical_rag.rag.retriever as retriever_module
from electrical_rag.core.settings import Settings
from electrical_rag.rag.retriever import VectorRetriever


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text: str):
        FakeEmbeddings.last_embedded_text = text
        return [0.1, 0.2, 0.3]


class FakeQdrantStore:
    search_calls = []

    def __init__(self, settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings

    def is_ready(self) -> bool:
        return True

    def ensure_payload_indexes(self) -> None:
        FakeQdrantStore.payload_indexes_ensured = True

    def search(
        self,
        query: str,
        k: int,
        document_id: int | None = None,
        device_names: list[str] | None = None,
        score_threshold: float | None = None,
    ):
        FakeQdrantStore.last_document_id = document_id
        FakeQdrantStore.last_device_names = device_names
        FakeQdrantStore.last_score_threshold = score_threshold
        FakeQdrantStore.search_calls.append(device_names)
        return [
            (
                Document(
                    page_content="power meter text",
                    metadata={"source": "power-meter.pdf", "page": 1},
                ),
                0.1,
            )
        ]

    def upsert_documents(self, chunks):
        FakeQdrantStore.last_upsert_count = len(chunks)
        FakeQdrantStore.last_chunks = chunks


def test_retriever_uses_qdrant_when_backend_is_qdrant(monkeypatch):
    FakeQdrantStore.search_calls = []
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", FakeQdrantStore)
    settings = Settings(vector_backend="qdrant")

    retriever = VectorRetriever(settings)
    results = retriever.search("What is power meter?", k=1)

    assert results[0][0].page_content == "power meter text"
    assert FakeQdrantStore.last_device_names == ["power meter"]


def test_retriever_warmup_validates_embedding_dimension(monkeypatch):
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", FakeQdrantStore)
    settings = Settings(vector_backend="qdrant")

    retriever = VectorRetriever(settings)

    assert retriever.warmup_embeddings() == 3
    assert FakeEmbeddings.last_embedded_text == "electrical_rag embedding model warmup"


def test_retriever_passes_document_filter_to_qdrant(monkeypatch):
    FakeQdrantStore.search_calls = []
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", FakeQdrantStore)
    settings = Settings(vector_backend="qdrant")

    retriever = VectorRetriever(settings)
    retriever.search("What is power meter?", k=1, document_id=12)

    assert FakeQdrantStore.last_document_id == 12
    assert FakeQdrantStore.last_device_names is None


def test_retriever_passes_score_threshold_to_qdrant(monkeypatch):
    FakeQdrantStore.search_calls = []
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", FakeQdrantStore)
    settings = Settings(vector_backend="qdrant", rag_min_retrieval_score=0.3)

    retriever = VectorRetriever(settings)
    retriever.search("What is power meter?", k=1)

    assert FakeQdrantStore.last_score_threshold == 0.3


def test_retriever_uses_reranker_when_enabled(monkeypatch):
    class MultiResultQdrantStore(FakeQdrantStore):
        def search(
            self,
            query: str,
            k: int,
            document_id: int | None = None,
            device_names: list[str] | None = None,
            score_threshold: float | None = None,
        ):
            return [
                (
                    Document(
                        page_content="generic text",
                        metadata={"source": "generic.pdf", "page": 1},
                    ),
                    0.8,
                ),
                (
                    Document(
                        page_content="specific Modbus BACnet RS-485 text",
                        metadata={"source": "specific.pdf", "page": 2},
                    ),
                    0.5,
                ),
            ]

    class FakeReranker:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def rerank(self, query, results):
            FakeReranker.last_query = query
            return [results[1], results[0]]

    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", MultiResultQdrantStore)
    monkeypatch.setattr(retriever_module, "CrossEncoderReranker", FakeReranker)
    settings = Settings(vector_backend="qdrant", enable_reranker=True)

    retriever = VectorRetriever(settings)
    results = retriever.search("What protocols are supported?", k=1)

    assert FakeReranker.last_query == "What protocols are supported?"
    assert results[0][0].metadata["source"] == "specific.pdf"
    assert results[0][1] == 0.5


def test_retriever_falls_back_when_device_filter_has_no_results(monkeypatch):
    class FallbackQdrantStore(FakeQdrantStore):
        def search(
            self,
            query: str,
            k: int,
            document_id: int | None = None,
            device_names: list[str] | None = None,
            score_threshold: float | None = None,
        ):
            FallbackQdrantStore.search_calls.append(device_names)
            if device_names:
                return []
            return [
                (
                    Document(
                        page_content="Fallback power meter text",
                        metadata={"source": "power-meter.pdf", "device_name": "power meter"},
                    ),
                    0.7,
                )
            ]

    FallbackQdrantStore.search_calls = []
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", FallbackQdrantStore)

    retriever = VectorRetriever(Settings(vector_backend="qdrant"))
    results = retriever.search("What is power meter?", k=1)

    assert FallbackQdrantStore.search_calls == [["power meter"], None]
    assert results[0][0].page_content == "Fallback power meter text"


def test_retriever_raises_when_qdrant_collection_is_missing(monkeypatch):
    class MissingQdrantStore(FakeQdrantStore):
        def is_ready(self) -> bool:
            return False

    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever_module, "QdrantVectorStore", MissingQdrantStore)
    settings = Settings(vector_backend="qdrant", qdrant_collection="missing")

    with pytest.raises(FileNotFoundError, match="Qdrant collection"):
        VectorRetriever(settings)


def test_ingestion_upserts_chunks_to_qdrant(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(ingestion_module, "QdrantVectorStore", FakeQdrantStore)
    monkeypatch.setattr(ingestion_module, "discover_pdfs", lambda data_dir: [tmp_path / "a.pdf"])
    monkeypatch.setattr(
        ingestion_module,
        "extract_pdf_documents",
        lambda **kwargs: ([Document(page_content="hello", metadata={"source": "a.pdf"})], 0),
    )
    settings = Settings(
        data_dir=tmp_path,
        vector_backend="qdrant",
        chunk_size=100,
        chunk_overlap=0,
    )

    stats = ingestion_module.run_ingestion(settings)

    assert stats.chunks_created == 1
    assert FakeQdrantStore.last_upsert_count == 1


def test_document_ingestion_adds_document_id_to_qdrant_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(ingestion_module, "QdrantVectorStore", FakeQdrantStore)

    pdf_path = tmp_path / "uploaded.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ncontent")

    monkeypatch.setattr(
        ingestion_module,
        "extract_pdf_documents",
        lambda **kwargs: (
            [
                Document(
                    page_content="uploaded text",
                    metadata={"source": "uploaded.pdf", "page": 1},
                )
            ],
            0,
        ),
    )
    settings = Settings(
        data_dir=tmp_path,
        vector_backend="qdrant",
        chunk_size=100,
        chunk_overlap=0,
    )

    stats = ingestion_module.run_document_ingestion(
        settings=settings,
        pdf_path=pdf_path,
        document_id=42,
    )

    assert stats.pdf_files == 1
    assert FakeQdrantStore.last_upsert_count == 1
    assert FakeQdrantStore.last_chunks[0].metadata["document_id"] == 42
