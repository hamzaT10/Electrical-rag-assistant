from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from electrical_rag.core.settings import Settings
from electrical_rag.observability.tracing import TraceManager
from electrical_rag.rag.device_metadata import detect_query_devices, rerank_results_by_devices
from electrical_rag.rag.qdrant_store import QdrantVectorStore
from electrical_rag.rag.reranker import CrossEncoderReranker


class VectorRetriever:
    def __init__(
        self,
        settings: Settings,
        trace_manager: TraceManager | None = None,
    ):
        self.settings = settings
        self.tracing = trace_manager or TraceManager(settings)
        self.vectorstore = None
        self.qdrant_store: QdrantVectorStore | None = None
        self.reranker = (
            CrossEncoderReranker(self.settings.reranker_model_name)
            if self.settings.enable_reranker
            else None
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name
        )

        if self.settings.vector_backend == "qdrant":
            self.qdrant_store = QdrantVectorStore(
                settings,
                embeddings=self.embeddings,
            )
            if not self.qdrant_store.is_ready():
                raise FileNotFoundError(
                    f"Qdrant collection '{self.settings.qdrant_collection}' not found. "
                    "Run ingestion first."
                )
            self.qdrant_store.ensure_payload_indexes()
            return

        if not self.settings.vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at '{self.settings.vectorstore_dir}'. "
                "Run ingestion first."
            )

        self.vectorstore = FAISS.load_local(
            str(self.settings.vectorstore_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def warmup_embeddings(self) -> int:
        vector = self.embeddings.embed_query("electrical_rag embedding model warmup")
        if not vector:
            raise RuntimeError("Embedding model returned an empty warmup vector")
        return len(vector)

    def search(
        self,
        query: str,
        k: int | None = None,
        document_id: int | None = None,
    ) -> list[tuple[Document, float]]:
        top_k = k or self.settings.retrieval_top_k
        detected_devices = detect_query_devices(query)

        if self.qdrant_store is not None:
            candidate_k = max(top_k * 3, top_k + 4)
            score_threshold = self.settings.rag_min_retrieval_score or None
            device_filter = detected_devices if document_id is None else None
            raw_results = self.qdrant_store.search(
                query,
                candidate_k,
                document_id=document_id,
                device_names=device_filter,
                score_threshold=score_threshold,
            )
            if device_filter and not raw_results:
                raw_results = self.qdrant_store.search(
                    query,
                    candidate_k,
                    document_id=document_id,
                    score_threshold=score_threshold,
                )
        else:
            if self.vectorstore is None:
                return []
            candidate_k = min(max(top_k * 3, top_k + 4), self.vectorstore.index.ntotal)
            raw_results = self.vectorstore.similarity_search_with_score(query, k=candidate_k)

        reranked_results = rerank_results_by_devices(raw_results, detected_devices)
        if self.reranker is not None:
            with self.tracing.span(
                name="rerank-candidates",
                as_type="retriever",
                metadata={
                    "model": self.settings.reranker_model_name,
                    "candidate_count": len(reranked_results),
                },
            ) as reranker_span:
                reranked_results = self.reranker.rerank(query, reranked_results)
                reranker_span.update(
                    output={
                        "result_count": min(len(reranked_results), top_k),
                        "reranker_scores": [
                            document.metadata.get("reranker_score")
                            for document, _ in reranked_results[:top_k]
                        ],
                    }
                )

        return reranked_results[:top_k]
