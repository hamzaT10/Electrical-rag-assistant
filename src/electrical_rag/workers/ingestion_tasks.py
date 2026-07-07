from __future__ import annotations

import logging
from pathlib import Path

from electrical_rag.core.settings import Settings
from electrical_rag.db.repositories import IngestionJobRepository
from electrical_rag.db.session import SessionLocal
from electrical_rag.rag.ingestion import discover_pdfs, run_document_ingestion, run_ingestion
from electrical_rag.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _settings_for_job(base_settings: Settings, source_path: str | None) -> Settings:
    if not source_path:
        return base_settings

    base_data_dir = base_settings.data_dir.resolve()
    requested_path = Path(source_path)

    if requested_path.is_absolute():
        job_data_dir = requested_path.resolve()
    elif requested_path.parts and requested_path.parts[0] == base_settings.data_dir.name:
        job_data_dir = (Path.cwd() / requested_path).resolve()
    else:
        job_data_dir = (base_data_dir / requested_path).resolve()

    if not job_data_dir.is_relative_to(base_data_dir):
        raise ValueError(f"Ingestion source must stay under DATA_DIR: {source_path}")

    return base_settings.model_copy(update={"data_dir": job_data_dir})


def _resolve_document_path(settings: Settings, storage_path: str) -> Path:
    requested_path = Path(storage_path)
    if requested_path.is_absolute():
        document_path = requested_path.resolve()
    elif requested_path.parts and requested_path.parts[0] == settings.data_dir.name:
        document_path = (Path.cwd() / requested_path).resolve()
    else:
        document_path = (settings.data_dir.resolve() / requested_path).resolve()

    if not document_path.is_relative_to(settings.data_dir.resolve()):
        raise ValueError(f"Document path must stay under DATA_DIR: {storage_path}")

    return document_path


def _register_full_ingestion_documents(settings: Settings) -> dict[str, int]:
    pdf_files = discover_pdfs(settings.data_dir)
    documents_by_source: dict[str, int] = {}

    with SessionLocal() as db:
        repo = IngestionJobRepository(db)
        for pdf_path in pdf_files:
            source_path = pdf_path.relative_to(settings.data_dir).as_posix()
            document = repo.get_or_create_ingested_document(
                source_path=source_path,
                filename=pdf_path.name,
                storage_path=pdf_path.as_posix(),
            )
            documents_by_source[source_path] = document.id

    return documents_by_source


@celery_app.task(name="electrical_rag.ingest_documents")
def ingest_documents(job_id: int) -> dict[str, int]:
    settings = Settings()

    with SessionLocal() as db:
        repo = IngestionJobRepository(db)
        job = repo.mark_running(job_id)
        if job is None:
            raise ValueError(f"Ingestion job not found: {job_id}")
        document_id = job.document_id
        document_storage_path = job.document.storage_path if job.document is not None else None
        source_path = job.source_path
        if document_id is not None:
            repo.mark_document_indexing(document_id)

    full_ingestion_document_ids: list[int] = []
    try:
        if document_id is not None:
            if document_storage_path is None:
                raise ValueError(f"Ingestion job {job_id} has no document storage path")
            document_path = _resolve_document_path(settings, document_storage_path)
            stats = run_document_ingestion(
                settings=settings,
                pdf_path=document_path,
                document_id=document_id,
            )
        else:
            job_settings = _settings_for_job(settings, source_path)
            document_ids_by_source = None
            if job_settings.vector_backend == "qdrant":
                document_ids_by_source = _register_full_ingestion_documents(job_settings)
                full_ingestion_document_ids = list(document_ids_by_source.values())
            stats = run_ingestion(
                job_settings,
                document_ids_by_source=document_ids_by_source,
            )
    except Exception as exc:
        with SessionLocal() as db:
            repo = IngestionJobRepository(db)
            repo.mark_failed(job_id, str(exc))
            if document_id is not None:
                repo.mark_document_failed(document_id)
            else:
                for full_document_id in full_ingestion_document_ids:
                    repo.mark_document_failed(full_document_id)
        logger.exception("ingestion_job_failed job_id=%s error=%s", job_id, exc)
        raise

    with SessionLocal() as db:
        repo = IngestionJobRepository(db)
        repo.mark_completed(job_id, stats)
        if document_id is not None:
            repo.mark_document_indexed(document_id, page_count=stats.pages_loaded)
        else:
            for full_document_id in full_ingestion_document_ids:
                repo.mark_document_indexed(full_document_id)

    logger.info("ingestion_job_completed job_id=%s stats=%s", job_id, stats)
    return {
        "pdf_files": stats.pdf_files,
        "pages_loaded": stats.pages_loaded,
        "pages_ocr": stats.pages_ocr,
        "chunks_created": stats.chunks_created,
    }
