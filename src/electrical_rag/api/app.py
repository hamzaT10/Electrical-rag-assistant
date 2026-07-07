from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from electrical_rag.api.schemas import (
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    ChatSessionMessagesResponse,
    ChatSessionSummary,
    Citation,
    DocumentSummary,
    DocumentUploadResponse,
    HealthResponse,
    IngestionJobCreate,
    IngestionJobResponse,
)
from electrical_rag.cache.redis_cache import ChatCache
from electrical_rag.core.settings import Settings
from electrical_rag.db.models import IngestionJob
from electrical_rag.db.repositories import ChatRepository, IngestionJobRepository
from electrical_rag.db.session import get_db_session
from electrical_rag.observability.logging import configure_logging
from electrical_rag.observability.metrics import (
    record_cache_event,
    record_component_initialization,
    set_rag_service_ready,
)
from electrical_rag.observability.middleware import ObservabilityMiddleware
from electrical_rag.observability.tracing import TraceManager
from electrical_rag.providers.lmstudio import LMStudioClient, LMStudioUnavailableError
from electrical_rag.rag.qdrant_store import QdrantVectorStore
from electrical_rag.security.rate_limit import RedisRateLimiter
from electrical_rag.services.qa_service import RAGService
from electrical_rag.workers.ingestion_tasks import ingest_documents

settings = Settings()
chat_cache = ChatCache(settings)
rate_limiter = RedisRateLimiter(settings)
trace_manager = TraceManager(settings)

configure_logging(settings.app_log_level, settings.app_log_format)
logger = logging.getLogger(__name__)

service: RAGService | None = None
service_bootstrap_lock = threading.Lock()
service_bootstrap_error: str | None = None
service_bootstrap_seconds: float | None = None
embedding_dimension: int | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.preload_rag_service:
        await asyncio.to_thread(_bootstrap_service)
    try:
        yield
    finally:
        trace_manager.flush()


app = FastAPI(
    title="Electrical RAG API",
    version="0.1.0",
    description="Document QA API backed by FAISS retrieval and LM Studio.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(ObservabilityMiddleware, metrics_enabled=settings.enable_metrics)


def _bootstrap_service() -> RAGService | None:
    global embedding_dimension
    global service
    global service_bootstrap_error
    global service_bootstrap_seconds

    if service is not None:
        return service

    with service_bootstrap_lock:
        if service is not None:
            return service

        started_at = time.perf_counter()
        set_rag_service_ready(False)
        try:
            initialization_started_at = time.perf_counter()
            candidate = RAGService(settings, trace_manager=trace_manager)
            initialization_seconds = time.perf_counter() - initialization_started_at
            record_component_initialization(
                "retriever_initialization",
                initialization_seconds,
            )

            warmup_seconds = 0.0
            dimension: int | None = None
            if settings.warmup_embedding_model:
                warmup_started_at = time.perf_counter()
                dimension = candidate.warmup_embeddings()
                warmup_seconds = time.perf_counter() - warmup_started_at
                record_component_initialization(
                    "embedding_warmup",
                    warmup_seconds,
                )

            service = candidate
            embedding_dimension = dimension
            service_bootstrap_error = None
            service_bootstrap_seconds = time.perf_counter() - started_at
            record_component_initialization(
                "rag_service_total",
                service_bootstrap_seconds,
            )
            set_rag_service_ready(True)
            logger.info(
                "rag_service_initialized",
                extra={
                    "embedding_model": settings.embedding_model_name,
                    "embedding_dimension": embedding_dimension,
                    "initialization_seconds": round(initialization_seconds, 6),
                    "warmup_seconds": round(warmup_seconds, 6),
                    "total_seconds": round(service_bootstrap_seconds, 6),
                },
            )
            return service
        except FileNotFoundError as exc:
            service_bootstrap_error = str(exc)
            service_bootstrap_seconds = time.perf_counter() - started_at
            set_rag_service_ready(False)
            logger.warning(
                "rag_service_initialization_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "total_seconds": round(service_bootstrap_seconds, 6),
                },
            )
            return None
        except Exception as exc:
            service_bootstrap_error = str(exc)
            service_bootstrap_seconds = time.perf_counter() - started_at
            set_rag_service_ready(False)
            logger.exception(
                "rag_service_initialization_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "total_seconds": round(service_bootstrap_seconds, 6),
                },
            )
            return None


def _check_lmstudio_ready() -> tuple[bool, str | None]:
    return LMStudioClient(settings).check_health()


def _check_vectorstore_ready() -> bool:
    if settings.vector_backend == "qdrant":
        try:
            return QdrantVectorStore(settings).is_ready()
        except Exception as exc:
            logger.warning("qdrant_health_check_failed error=%s", exc)
            return False

    return settings.vectorstore_dir.exists()


def _to_ingestion_job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        status=job.status,
        document_id=job.document_id,
        source_path=job.source_path,
        task_id=job.task_id,
        pdf_files=job.pdf_files,
        pages_loaded=job.pages_loaded,
        pages_ocr=job.pages_ocr,
        chunks_created=job.chunks_created,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _chat_cache_key(question: str, document_id: int | None) -> str:
    scope = document_id if document_id is not None else "all"
    return f"scope:{scope}:question:{question}"


def _safe_upload_filename(filename: str) -> str:
    raw_name = Path(filename).name
    stem = Path(raw_name).stem
    suffix = Path(raw_name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "document"
    return f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"


def _enqueue_ingestion_job(repo: IngestionJobRepository, job: IngestionJob) -> IngestionJob:
    try:
        async_result = ingest_documents.delay(job.id)
    except Exception as exc:
        repo.mark_failed(job.id, f"Failed to enqueue ingestion task: {exc}")
        logger.exception("ingestion_job_enqueue_failed job_id=%s error=%s", job.id, exc)
        raise HTTPException(status_code=503, detail="Failed to enqueue ingestion job") from exc

    return repo.set_task_id(job.id, async_result.id) or job


def _source_path_for_saved_pdf(saved_path: Path) -> str:
    data_dir = settings.data_dir.resolve()
    resolved_saved_path = saved_path.resolve()
    try:
        return resolved_saved_path.relative_to(data_dir).as_posix()
    except ValueError:
        return saved_path.as_posix()


def _resolve_upload_dir() -> Path:
    upload_dir = settings.upload_dir
    if upload_dir.is_absolute():
        return upload_dir

    data_dir = settings.data_dir.resolve()
    if upload_dir.parts and upload_dir.parts[0] == settings.data_dir.name:
        return (Path.cwd() / upload_dir).resolve()

    return (data_dir / upload_dir).resolve()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vectorstore_ready = _check_vectorstore_ready()
    lmstudio_ready, lmstudio_error = _check_lmstudio_ready()
    rag_service_ready = service is not None
    service_requirement_met = (
        rag_service_ready if settings.preload_rag_service else True
    )
    status = (
        "ready"
        if vectorstore_ready and lmstudio_ready and service_requirement_met
        else "degraded"
    )
    return HealthResponse(
        status=status,
        vectorstore_ready=vectorstore_ready,
        lmstudio_ready=lmstudio_ready,
        lmstudio_error=lmstudio_error,
        rag_service_ready=rag_service_ready,
        rag_service_error=service_bootstrap_error,
        rag_service_startup_seconds=service_bootstrap_seconds,
        embedding_dimension=embedding_dimension,
    )


@app.get("/meta")
def meta() -> dict[str, object]:
    return {
        "service": "electrical-rag-api",
        "model": settings.lmstudio_model,
        "embedding_model": settings.embedding_model_name,
        "embedding_dimension": embedding_dimension,
        "preload_rag_service": settings.preload_rag_service,
        "warmup_embedding_model": settings.warmup_embedding_model,
        "rag_service_ready": service is not None,
        "rag_service_startup_seconds": service_bootstrap_seconds,
        "vector_backend": settings.vector_backend,
        "qdrant_collection": settings.qdrant_collection,
        "retrieval_top_k": settings.retrieval_top_k,
        "rag_min_retrieval_score": settings.rag_min_retrieval_score,
        "enable_reranker": settings.enable_reranker,
        "reranker_model": settings.reranker_model_name if settings.enable_reranker else None,
        "metrics_enabled": settings.enable_metrics,
        "langfuse_enabled": trace_manager.enabled,
        "langfuse_capture_content": settings.langfuse_capture_content,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics are disabled")
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/ingestion/jobs", response_model=IngestionJobResponse, status_code=202)
def create_ingestion_job(
    payload: IngestionJobCreate,
    db: Session = Depends(get_db_session),
) -> IngestionJobResponse:
    source_path = payload.source_path.strip() if payload.source_path else None
    repo = IngestionJobRepository(db)
    job = repo.create_job(source_path=source_path)
    job = _enqueue_ingestion_job(repo, job)
    return _to_ingestion_job_response(job)


@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=202)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = _resolve_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = _safe_upload_filename(file.filename)
    saved_path = upload_dir / safe_filename
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    bytes_written = 0
    try:
        with saved_path.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is larger than {settings.max_upload_size_mb} MB",
                    )
                output.write(chunk)
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF") from exc
    finally:
        file.file.close()

    source_path = _source_path_for_saved_pdf(saved_path)
    repo = IngestionJobRepository(db)
    document = repo.create_uploaded_document(
        source_path=source_path,
        filename=safe_filename,
        storage_path=saved_path.as_posix(),
    )
    job = repo.create_job(document_id=document.id)
    job = _enqueue_ingestion_job(repo, job)

    return DocumentUploadResponse(
        filename=safe_filename,
        saved_path=saved_path.as_posix(),
        document_id=document.id,
        job_id=job.id,
        job_status=job.status,
    )


@app.get("/ingestion/jobs", response_model=list[IngestionJobResponse])
def list_ingestion_jobs(
    db: Session = Depends(get_db_session),
) -> list[IngestionJobResponse]:
    repo = IngestionJobRepository(db)
    return [_to_ingestion_job_response(job) for job in repo.list_jobs()]


@app.get("/ingestion/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(
    job_id: int,
    db: Session = Depends(get_db_session),
) -> IngestionJobResponse:
    repo = IngestionJobRepository(db)
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return _to_ingestion_job_response(job)


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    db: Session = Depends(get_db_session),
) -> list[DocumentSummary]:
    repo = IngestionJobRepository(db)
    return [
        DocumentSummary(
            id=document.id,
            filename=document.filename,
            source_path=document.source_path,
            status=document.status,
            page_count=document.page_count,
            created_at=document.created_at,
            indexed_at=document.indexed_at,
        )
        for document in repo.list_documents(status="indexed")
    ]


@app.get("/chat/sessions", response_model=list[ChatSessionSummary])
def list_chat_sessions(
    db: Session = Depends(get_db_session),
) -> list[ChatSessionSummary]:
    repo = ChatRepository(db)
    demo_user = repo.get_or_create_demo_user()
    sessions = repo.list_chat_sessions(demo_user.id)
    return [
        ChatSessionSummary(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
        )
        for session in sessions
    ]


@app.get(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatSessionMessagesResponse,
)
def get_chat_session_messages(
    session_id: int,
    db: Session = Depends(get_db_session),
) -> ChatSessionMessagesResponse:
    repo = ChatRepository(db)
    demo_user = repo.get_or_create_demo_user()
    chat_session = repo.get_chat_session(session_id)

    if chat_session is None or chat_session.user_id != demo_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = repo.list_messages(session_id)
    return ChatSessionMessagesResponse(
        session_id=session_id,
        messages=[
            ChatMessageItem(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db_session),
) -> ChatResponse:
    request_id = request.state.request_id
    client_ip = request.client.host if request.client else "unknown"
    rate_limit = rate_limiter.check(client_ip)
    response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
    response.headers["X-RateLimit-Remaining"] = str(rate_limit.remaining)

    if not rate_limit.allowed:
        if rate_limit.retry_after_seconds is not None:
            response.headers["Retry-After"] = str(rate_limit.retry_after_seconds)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
        )

    repo = ChatRepository(db)

    rag_service = _bootstrap_service()
    if rag_service is None:
        logger.warning("request_id=%s chat_unavailable vectorstore_missing", request_id)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Vectorstore not found at '{settings.vectorstore_dir}'. "
                "Run: python -m electrical_rag.rag.ingestion"
            ),
        )

    try:
        demo_user = repo.get_or_create_demo_user()
        chat_session = repo.create_chat_session(
            user_id=demo_user.id,
            title=payload.question[:80],
        )
        repo.create_message(
            session_id=chat_session.id,
            role="user",
            content=payload.question,
        )

        with trace_manager.trace(
            name="electrical-rag-chat",
            request_id=request_id,
            mode="sync",
            input_data=trace_manager.content(payload.question, "question"),
            user_id=str(demo_user.id),
            session_id=str(chat_session.id),
            metadata={"document_id": payload.document_id},
        ) as trace:
            cache_key = _chat_cache_key(payload.question, payload.document_id)
            with trace_manager.span(
                name="chat-cache-lookup",
                metadata={"document_id": payload.document_id},
            ) as cache_span:
                cached_payload = chat_cache.get(cache_key)
                cache_result = "hit" if cached_payload is not None else "miss"
                cache_span.update(output={"result": cache_result})

            if cached_payload is not None:
                record_cache_event("hit")
                answer = str(cached_payload.get("answer", ""))
                raw_citations = cached_payload.get("citations", [])
                if not isinstance(raw_citations, list):
                    raw_citations = []
                logger.info("request_id=%s chat_cache_hit", request_id)
            else:
                record_cache_event("miss")
                logger.info("request_id=%s chat_cache_miss", request_id)
                answer, raw_citations = rag_service.ask(
                    payload.question,
                    request_id=request_id,
                    document_id=payload.document_id,
                )
                chat_cache.set(
                    cache_key,
                    {
                        "answer": answer,
                        "citations": raw_citations,
                    },
                )

            trace.update(
                output=trace_manager.content(answer, "answer"),
                metadata={
                    "cache_result": cache_result,
                    "citation_count": len(raw_citations),
                },
            )
            if trace.trace_id:
                logger.info(
                    "langfuse_trace_created",
                    extra={"langfuse_trace_id": trace.trace_id},
                )
    except LMStudioUnavailableError as exc:
        logger.warning("request_id=%s lmstudio_unavailable error=%s", request_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("request_id=%s chat_request_failed error=%s", request_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    citations = [Citation(**item) for item in raw_citations]
    assistant_message = repo.create_message(
        session_id=chat_session.id,
        role="assistant",
        content=answer,
    )
    repo.save_retrieved_sources(
        message_id=assistant_message.id,
        citations=raw_citations,
    )
    return ChatResponse(answer=answer, citations=citations, session_id=chat_session.id)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    request_id = request.state.request_id

    rag_service = _bootstrap_service()
    if rag_service is None:
        logger.warning("request_id=%s chat_stream_unavailable vectorstore_missing", request_id)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Vectorstore not found at '{settings.vectorstore_dir}'. "
                "Run: python -m electrical_rag.rag.ingestion"
            ),
        )

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        finished = object()
        stopped = threading.Event()

        def publish(item: str | object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        # One producer thread owns every next() call so OpenTelemetry context
        # managers are entered and exited in the same execution context.
        def produce() -> None:
            with trace_manager.trace(
                name="electrical-rag-chat",
                request_id=request_id,
                mode="stream",
                input_data=trace_manager.content(payload.question, "question"),
                user_id="demo",
                metadata={"document_id": payload.document_id},
            ) as trace:
                try:
                    for event in rag_service.ask_stream(
                        payload.question,
                        request_id=request_id,
                        document_id=payload.document_id,
                    ):
                        if stopped.is_set():
                            break
                        publish(json.dumps(event, ensure_ascii=False) + "\n")
                    trace.update(output={"stream_completed": not stopped.is_set()})
                    if trace.trace_id:
                        logger.info(
                            "langfuse_trace_created",
                            extra={"langfuse_trace_id": trace.trace_id},
                        )
                except LMStudioUnavailableError as exc:
                    trace.update(level="ERROR", status_message=str(exc))
                    logger.warning(
                        "request_id=%s streaming_lmstudio_unavailable error=%s",
                        request_id,
                        exc,
                    )
                    error_event = {"type": "error", "detail": str(exc)}
                    publish(json.dumps(error_event, ensure_ascii=False) + "\n")
                except Exception as exc:
                    trace.update(level="ERROR", status_message=str(exc))
                    logger.exception(
                        "request_id=%s streaming_chat_request_failed error=%s",
                        request_id,
                        exc,
                    )
                    error_event = {"type": "error", "detail": str(exc)}
                    publish(json.dumps(error_event, ensure_ascii=False) + "\n")
                finally:
                    publish(finished)

        producer_task = asyncio.create_task(asyncio.to_thread(produce))
        completed = False
        try:
            while True:
                item = await queue.get()
                if item is finished:
                    completed = True
                    break
                yield str(item)
        finally:
            stopped.set()
            if completed:
                await producer_task
            else:
                producer_task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )
