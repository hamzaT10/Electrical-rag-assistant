from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "electrical_rag_http_requests_total",
    "Total HTTP requests completed by the API.",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "electrical_rag_http_request_duration_seconds",
    "End-to-end HTTP response duration, including streamed response bodies.",
    ("method", "route"),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "electrical_rag_http_requests_in_progress",
    "HTTP requests currently being processed.",
)

CHAT_CACHE_EVENTS = Counter(
    "electrical_rag_chat_cache_events_total",
    "Chat cache lookups by outcome.",
    ("result",),
)
RAG_REQUESTS = Counter(
    "electrical_rag_rag_requests_total",
    "RAG requests by response mode and outcome.",
    ("mode", "outcome"),
)
RAG_STAGE_DURATION = Histogram(
    "electrical_rag_rag_stage_duration_seconds",
    "Duration of retrieval, reranking, LLM, and full RAG stages.",
    ("stage", "mode"),
)
RAG_RETRIEVED_CHUNKS = Histogram(
    "electrical_rag_rag_retrieved_chunks",
    "Number of chunks included in the final LLM context.",
    ("mode",),
    buckets=(0, 1, 2, 3, 5, 8, 13, 20),
)
RAG_COMPONENT_INITIALIZATION = Histogram(
    "electrical_rag_rag_component_initialization_seconds",
    "Startup initialization duration for RAG components.",
    ("component",),
)
RAG_SERVICE_READY = Gauge(
    "electrical_rag_rag_service_ready",
    "Whether the in-process RAG service is initialized and ready.",
)


def record_http_request(
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status_code)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, route=route).observe(duration_seconds)


def record_cache_event(result: str) -> None:
    CHAT_CACHE_EVENTS.labels(result=result).inc()


def record_rag_request(
    *,
    mode: str,
    outcome: str,
    retrieval_seconds: float,
    total_seconds: float,
    chunk_count: int,
    llm_seconds: float | None = None,
    ttft_seconds: float | None = None,
) -> None:
    RAG_REQUESTS.labels(mode=mode, outcome=outcome).inc()
    RAG_STAGE_DURATION.labels(stage="retrieval", mode=mode).observe(retrieval_seconds)
    RAG_STAGE_DURATION.labels(stage="total", mode=mode).observe(total_seconds)
    RAG_RETRIEVED_CHUNKS.labels(mode=mode).observe(chunk_count)
    if llm_seconds is not None:
        RAG_STAGE_DURATION.labels(stage="llm", mode=mode).observe(llm_seconds)
    if ttft_seconds is not None:
        RAG_STAGE_DURATION.labels(stage="ttft", mode=mode).observe(ttft_seconds)


def record_reranker_duration(duration_seconds: float) -> None:
    RAG_STAGE_DURATION.labels(stage="rerank", mode="shared").observe(duration_seconds)


def record_component_initialization(component: str, duration_seconds: float) -> None:
    RAG_COMPONENT_INITIALIZATION.labels(component=component).observe(duration_seconds)


def set_rag_service_ready(ready: bool) -> None:
    RAG_SERVICE_READY.set(1 if ready else 0)
