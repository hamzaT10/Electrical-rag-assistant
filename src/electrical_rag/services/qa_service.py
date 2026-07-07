from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

from electrical_rag.core.messages import NO_RELEVANT_CONTEXT_ANSWER
from electrical_rag.core.settings import Settings
from electrical_rag.observability.metrics import record_rag_request
from electrical_rag.observability.tracing import TraceManager
from electrical_rag.providers.openai_compatible import OpenAICompatibleClient
from electrical_rag.rag.answer_postprocessing import trim_incomplete_trailing_sentence
from electrical_rag.rag.context_selection import apply_context_budget, limit_chunks_per_source_page
from electrical_rag.rag.device_metadata import detect_query_devices
from electrical_rag.rag.prompting import RetrievedChunk, build_citations, build_prompt
from electrical_rag.rag.retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _score_summary(results: list[tuple[Document, float]]) -> dict[str, float | None]:
    scores = [float(score) for _, score in results]
    if not scores:
        return {"top": None, "avg": None, "min": None}

    return {
        "top": max(scores),
        "avg": sum(scores) / len(scores),
        "min": min(scores),
    }


class RAGService:
    def __init__(
        self,
        settings: Settings,
        trace_manager: TraceManager | None = None,
    ):
        self.settings = settings
        self.tracing = trace_manager or TraceManager(settings)
        self.retriever = VectorRetriever(settings, trace_manager=self.tracing)
        self.llm = OpenAICompatibleClient(settings)

    def warmup_embeddings(self) -> int:
        return self.retriever.warmup_embeddings()

    @staticmethod
    def _to_chunk(item: tuple[Document, float]) -> RetrievedChunk:
        doc, score = item
        page_value = doc.metadata.get("page")
        page: int | None = None
        if isinstance(page_value, int):
            page = page_value
        elif isinstance(page_value, str) and page_value.isdigit():
            page = int(page_value)
        source = str(doc.metadata.get("source", "unknown"))
        return RetrievedChunk(text=doc.page_content, source=source, page=page, score=float(score))

    def ask(
        self,
        question: str,
        request_id: str = "unknown",
        document_id: int | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        total_start = time.perf_counter()
        detected_devices = detect_query_devices(question)

        retrieval_start = time.perf_counter()
        with self.tracing.span(
            name="retrieve-context",
            as_type="retriever",
            input_data=self.tracing.content(question, "question"),
            metadata={
                "document_id": document_id,
                "top_k": self.settings.retrieval_top_k,
                "vector_backend": self.settings.vector_backend,
            },
        ) as retrieval_span:
            results = self.retriever.search(
                question,
                self.settings.retrieval_top_k,
                document_id=document_id,
            )
            retrieval_span.update(
                output={
                    "result_count": len(results),
                    "sources": [
                        str(document.metadata.get("source", "unknown"))
                        for document, _ in results
                    ],
                    "scores": [round(float(score), 4) for _, score in results],
                }
            )
        retrieval_seconds = time.perf_counter() - retrieval_start
        score_summary = _score_summary(results)

        if not results:
            total_seconds = time.perf_counter() - total_start
            record_rag_request(
                mode="sync",
                outcome="no_context",
                retrieval_seconds=retrieval_seconds,
                total_seconds=total_seconds,
                chunk_count=0,
            )
            logger.info(
                "rag_request_no_context",
                extra={
                    "request_id": request_id,
                    "mode": "sync",
                    "question_chars": len(question),
                    "detected_devices": detected_devices,
                    "document_id": document_id,
                    "retrieval_seconds": round(retrieval_seconds, 6),
                    "total_seconds": round(total_seconds, 6),
                    "min_retrieval_score": self.settings.rag_min_retrieval_score,
                    "scores": score_summary,
                },
            )
            return NO_RELEVANT_CONTEXT_ANSWER, []

        with self.tracing.span(
            name="select-context",
            as_type="span",
            metadata={
                "max_context_chars": self.settings.max_context_chars,
                "max_chunk_chars": self.settings.max_chunk_chars,
            },
        ) as context_span:
            chunks = [self._to_chunk(item) for item in results]
            chunks = limit_chunks_per_source_page(
                chunks,
                self.settings.max_chunks_per_source_page,
            )
            prompt_chunks = apply_context_budget(
                chunks,
                self.settings.max_context_chars,
                self.settings.max_chunk_chars,
            )
            prompt = build_prompt(question, prompt_chunks)
            context_span.update(
                output={
                    "retrieved_chunks": len(results),
                    "selected_chunks": len(prompt_chunks),
                    "context_chars": sum(len(chunk.text) for chunk in prompt_chunks),
                    "sources": [chunk.source for chunk in prompt_chunks],
                }
            )

        llm_start = time.perf_counter()
        with self.tracing.span(
            name="generate-answer",
            as_type="generation",
            input_data=self.tracing.content(prompt, "prompt"),
            model=self.settings.llm_model,
            model_parameters={"temperature": self.settings.llm_temperature},
        ) as generation:
            answer = self.llm.ask(prompt)
            generation.update(output=self.tracing.content(answer or "", "answer"))
        llm_seconds = time.perf_counter() - llm_start

        citation_chunks = chunks[: len(prompt_chunks)]
        citations = build_citations(citation_chunks, max_items=self.settings.retrieval_top_k)

        if not answer:
            answer = "I could not generate an answer from the retrieved context."
        else:
            answer = trim_incomplete_trailing_sentence(answer)

        total_seconds = time.perf_counter() - total_start
        record_rag_request(
            mode="sync",
            outcome="answered",
            retrieval_seconds=retrieval_seconds,
            llm_seconds=llm_seconds,
            total_seconds=total_seconds,
            chunk_count=len(prompt_chunks),
        )
        logger.info(
            "rag_request_completed",
            extra={
                "request_id": request_id,
                "mode": "sync",
                "outcome": "answered",
                "question_chars": len(question),
                "detected_devices": detected_devices,
                "document_id": document_id,
                "retrieval_seconds": round(retrieval_seconds, 6),
                "min_retrieval_score": self.settings.rag_min_retrieval_score,
                "scores": score_summary,
                "llm_seconds": round(llm_seconds, 6),
                "total_seconds": round(total_seconds, 6),
                "prompt_chars": len(prompt),
                "context_chars": sum(len(chunk.text) for chunk in prompt_chunks),
                "chunk_count": len(prompt_chunks),
                "sources": [chunk.source for chunk in prompt_chunks],
            },
        )

        return answer, citations

    def ask_stream(
        self,
        question: str,
        request_id: str = "unknown",
        document_id: int | None = None,
    ):
        total_start = time.perf_counter()
        detected_devices = detect_query_devices(question)

        retrieval_start = time.perf_counter()
        with self.tracing.span(
            name="retrieve-context",
            as_type="retriever",
            input_data=self.tracing.content(question, "question"),
            metadata={
                "document_id": document_id,
                "top_k": self.settings.retrieval_top_k,
                "vector_backend": self.settings.vector_backend,
            },
        ) as retrieval_span:
            results = self.retriever.search(
                question,
                self.settings.retrieval_top_k,
                document_id=document_id,
            )
            retrieval_span.update(
                output={
                    "result_count": len(results),
                    "sources": [
                        str(document.metadata.get("source", "unknown"))
                        for document, _ in results
                    ],
                    "scores": [round(float(score), 4) for _, score in results],
                }
            )
        retrieval_seconds = time.perf_counter() - retrieval_start
        score_summary = _score_summary(results)

        if not results:
            total_seconds = time.perf_counter() - total_start
            record_rag_request(
                mode="stream",
                outcome="no_context",
                retrieval_seconds=retrieval_seconds,
                total_seconds=total_seconds,
                chunk_count=0,
            )
            logger.info(
                "rag_request_no_context",
                extra={
                    "request_id": request_id,
                    "mode": "stream",
                    "question_chars": len(question),
                    "detected_devices": detected_devices,
                    "document_id": document_id,
                    "retrieval_seconds": round(retrieval_seconds, 6),
                    "total_seconds": round(total_seconds, 6),
                    "min_retrieval_score": self.settings.rag_min_retrieval_score,
                    "scores": score_summary,
                },
            )
            yield {"type": "token", "content": NO_RELEVANT_CONTEXT_ANSWER}
            yield {"type": "citations", "citations": []}
            return

        with self.tracing.span(
            name="select-context",
            as_type="span",
            metadata={
                "max_context_chars": self.settings.max_context_chars,
                "max_chunk_chars": self.settings.max_chunk_chars,
            },
        ) as context_span:
            chunks = [self._to_chunk(item) for item in results]
            chunks = limit_chunks_per_source_page(
                chunks,
                self.settings.max_chunks_per_source_page,
            )
            prompt_chunks = apply_context_budget(
                chunks,
                self.settings.max_context_chars,
                self.settings.max_chunk_chars,
            )
            prompt = build_prompt(question, prompt_chunks)
            context_span.update(
                output={
                    "retrieved_chunks": len(results),
                    "selected_chunks": len(prompt_chunks),
                    "context_chars": sum(len(chunk.text) for chunk in prompt_chunks),
                    "sources": [chunk.source for chunk in prompt_chunks],
                }
            )
        citation_chunks = chunks[: len(prompt_chunks)]
        citations = build_citations(citation_chunks, max_items=self.settings.retrieval_top_k)

        llm_start = time.perf_counter()
        answer_parts: list[str] = []
        ttft_seconds: float | None = None
        llm_ttft_seconds: float | None = None
        with self.tracing.span(
            name="generate-answer",
            as_type="generation",
            input_data=self.tracing.content(prompt, "prompt"),
            model=self.settings.llm_model,
            model_parameters={"temperature": self.settings.llm_temperature},
        ) as generation:
            for token in self.llm.stream(prompt):
                if ttft_seconds is None:
                    now = time.perf_counter()
                    ttft_seconds = now - total_start
                    llm_ttft_seconds = now - llm_start
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            generation.update(
                output=self.tracing.content("".join(answer_parts).strip(), "answer"),
                metadata={
                    "ttft_seconds": round(ttft_seconds or 0.0, 6),
                    "llm_ttft_seconds": round(llm_ttft_seconds or 0.0, 6),
                },
            )

        llm_seconds = time.perf_counter() - llm_start
        answer = "".join(answer_parts).strip()
        if not answer:
            fallback = "I could not generate an answer from the retrieved context."
            if ttft_seconds is None:
                now = time.perf_counter()
                ttft_seconds = now - total_start
                llm_ttft_seconds = now - llm_start
            yield {"type": "token", "content": fallback}
        else:
            cleaned_answer = trim_incomplete_trailing_sentence(answer)
            if cleaned_answer != answer:
                yield {"type": "final", "answer": cleaned_answer}

        yield {"type": "citations", "citations": citations}

        total_seconds = time.perf_counter() - total_start
        record_rag_request(
            mode="stream",
            outcome="answered",
            retrieval_seconds=retrieval_seconds,
            llm_seconds=llm_seconds,
            total_seconds=total_seconds,
            chunk_count=len(prompt_chunks),
            ttft_seconds=ttft_seconds,
        )
        logger.info(
            "rag_request_completed",
            extra={
                "request_id": request_id,
                "mode": "stream",
                "outcome": "answered",
                "question_chars": len(question),
                "detected_devices": detected_devices,
                "document_id": document_id,
                "retrieval_seconds": round(retrieval_seconds, 6),
                "min_retrieval_score": self.settings.rag_min_retrieval_score,
                "scores": score_summary,
                "ttft_seconds": round(ttft_seconds or 0.0, 6),
                "llm_ttft_seconds": round(llm_ttft_seconds or 0.0, 6),
                "llm_seconds": round(llm_seconds, 6),
                "total_seconds": round(total_seconds, 6),
                "prompt_chars": len(prompt),
                "context_chars": sum(len(chunk.text) for chunk in prompt_chunks),
                "chunk_count": len(prompt_chunks),
                "sources": [chunk.source for chunk in prompt_chunks],
            },
        )
