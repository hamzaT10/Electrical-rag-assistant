from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from electrical_rag.core.settings import Settings
from electrical_rag.rag.qdrant_store import QdrantVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.1, 0.2] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.1, 0.2]


@dataclass
class FakePoint:
    payload: dict[str, object]
    score: float


class FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class FakeCollectionInfo:
    def __init__(self, payload_schema: dict[str, object] | None = None):
        self.payload_schema = payload_schema or {}


class FakeQdrantClient:
    def __init__(self):
        self.collections: set[str] = set()
        self.points = []
        self.query_vector = None
        self.query_filter = None
        self.score_threshold = None
        self.payload_indexes: dict[str, object] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config):
        self.collections.add(collection_name)
        self.vectors_config = vectors_config

    def get_collection(self, collection_name: str):
        return FakeCollectionInfo(self.payload_indexes)

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema,
        wait: bool,
    ):
        self.payload_indexes[field_name] = field_schema

    def upsert(self, collection_name: str, points, wait: bool):
        self.points.extend(points)

    def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        with_payload: bool,
        query_filter=None,
        score_threshold=None,
    ):
        self.query_vector = query
        self.query_filter = query_filter
        self.score_threshold = score_threshold
        return FakeQueryResponse(
            [
                FakePoint(
                    payload={
                        "text": "Voltage unbalance text",
                        "source": "manual.pdf",
                        "page": 4,
                        "chunk_id": 2,
                    },
                    score=0.42,
                )
            ]
        )


def test_qdrant_store_upserts_document_payloads():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    store.upsert_documents(
        [
            Document(
                page_content="chunk text",
                metadata={"source": "manual.pdf", "page": 1, "chunk_id": 7},
            )
        ]
    )

    assert "test_chunks" in client.collections
    assert "document_id" in client.payload_indexes
    assert "device_name" in client.payload_indexes
    assert len(client.points) == 1
    assert client.points[0].payload == {
        "text": "chunk text",
        "source": "manual.pdf",
        "page": 1,
        "chunk_id": 7,
    }


def test_qdrant_store_search_returns_documents_with_scores():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    client.collections.add("test_chunks")
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    results = store.search("What is voltage unbalance?", k=5)

    assert len(results) == 1
    document, score = results[0]
    assert document.page_content == "Voltage unbalance text"
    assert document.metadata == {
        "source": "manual.pdf",
        "page": 4,
        "chunk_id": 2,
    }
    assert score == 0.42


def test_qdrant_store_search_filters_by_document_id():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    client.collections.add("test_chunks")
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    store.search("What is voltage unbalance?", k=5, document_id=42)

    assert client.query_filter is not None
    condition = client.query_filter.must[0]
    assert condition.key == "document_id"
    assert condition.match.value == 42


def test_qdrant_store_search_filters_by_device_names():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    client.collections.add("test_chunks")
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    store.search("What is the power meter voltage range?", k=5, device_names=["power meter"])

    assert client.query_filter is not None
    condition = client.query_filter.must[0]
    assert condition.key == "device_name"
    assert condition.match.any == ["power meter"]


def test_qdrant_store_search_passes_score_threshold():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    client.collections.add("test_chunks")
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    store.search("What is voltage unbalance?", k=5, score_threshold=0.3)

    assert client.score_threshold == 0.3


def test_qdrant_store_creates_only_missing_payload_indexes():
    settings = Settings(vector_backend="qdrant", qdrant_collection="test_chunks")
    client = FakeQdrantClient()
    client.collections.add("test_chunks")
    client.payload_indexes["document_id"] = "integer"
    store = QdrantVectorStore(settings, embeddings=FakeEmbeddings(), client=client)

    store.ensure_collection(vector_size=3)

    assert client.payload_indexes["document_id"] == "integer"
    assert "device_name" in client.payload_indexes
