from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from electrical_rag.core.settings import Settings


class QdrantVectorStore:
    def __init__(
        self,
        settings: Settings,
        embeddings: HuggingFaceEmbeddings | None = None,
        client: QdrantClient | None = None,
    ):
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self._embeddings = embeddings
        self.client = client or QdrantClient(url=settings.qdrant_url)

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model_name
            )
        return self._embeddings

    def is_ready(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.ensure_payload_indexes()
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        collection_info = self.client.get_collection(self.collection_name)
        payload_schema = getattr(collection_info, "payload_schema", {}) or {}
        required_indexes = {
            "document_id": models.PayloadSchemaType.INTEGER,
            "device_name": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, field_schema in required_indexes.items():
            if field_name in payload_schema:
                continue
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def upsert_documents(self, documents: list[Document], batch_size: int = 64) -> None:
        if not documents:
            return

        texts = [document.page_content for document in documents]
        vectors = self.embeddings.embed_documents(texts)
        self.ensure_collection(len(vectors[0]))

        for start in range(0, len(documents), batch_size):
            batch_documents = documents[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=self._point_id(document),
                    vector=vector,
                    payload=self._payload(document),
                )
                for document, vector in zip(batch_documents, batch_vectors, strict=True)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query: str,
        k: int,
        document_id: int | None = None,
        device_names: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)
        conditions: list[models.FieldCondition] = []
        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )
        if device_names:
            conditions.append(
                models.FieldCondition(
                    key="device_name",
                    match=models.MatchAny(any=device_names),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            score_threshold=score_threshold,
            limit=k,
            with_payload=True,
        )

        points = getattr(response, "points", response)
        return [self._document_from_point(point) for point in points]

    @staticmethod
    def _payload(document: Document) -> dict[str, Any]:
        payload = dict(document.metadata)
        payload["text"] = document.page_content
        return payload

    @staticmethod
    def _point_id(document: Document) -> str:
        source = str(document.metadata.get("source", "unknown"))
        page = str(document.metadata.get("page", "unknown"))
        chunk_id = str(document.metadata.get("chunk_id", "unknown"))
        stable_key = f"{source}:{page}:{chunk_id}:{document.page_content[:120]}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

    @staticmethod
    def _document_from_point(point: object) -> tuple[Document, float]:
        payload = getattr(point, "payload", None) or {}
        score = float(getattr(point, "score", 0.0) or 0.0)
        text = str(payload.get("text", ""))
        metadata = dict(payload)
        metadata.pop("text", None)
        return Document(page_content=text, metadata=metadata), score
