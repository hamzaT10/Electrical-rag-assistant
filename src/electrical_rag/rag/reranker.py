from __future__ import annotations

import time

from langchain_core.documents import Document

from electrical_rag.observability.metrics import record_reranker_duration


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        if len(results) <= 1:
            return results

        pairs = [(query, document.page_content) for document, _ in results]
        started_at = time.perf_counter()
        reranker_scores = self.model.predict(pairs)
        record_reranker_duration(time.perf_counter() - started_at)
        scored_results = []
        for (document, original_score), reranker_score in zip(
            results,
            reranker_scores,
            strict=True,
        ):
            document.metadata["reranker_score"] = round(float(reranker_score), 4)
            scored_results.append((document, original_score, float(reranker_score)))

        scored_results.sort(key=lambda item: item[2], reverse=True)
        return [(document, original_score) for document, original_score, _ in scored_results]
