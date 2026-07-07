from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import electrical_rag.workers.ingestion_tasks as task_module
from electrical_rag.core.settings import Settings
from electrical_rag.db.models import Document
from electrical_rag.db.repositories import IngestionJobRepository
from electrical_rag.db.session import Base


@dataclass
class FakeStats:
    pdf_files: int = 2
    pages_loaded: int = 10
    pages_ocr: int = 1
    chunks_created: int = 25


def test_ingestion_task_marks_job_completed(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    db = session_local()
    repo = IngestionJobRepository(db)
    job = repo.create_job(source_path="Data/Standards")
    db.close()

    def fake_run_ingestion(settings, document_ids_by_source=None):
        assert document_ids_by_source is None
        data_dir = str(settings.data_dir)
        assert data_dir.endswith("Data\\Standards") or data_dir.endswith("Data/Standards")
        return FakeStats()

    monkeypatch.setattr(task_module, "SessionLocal", session_local)
    monkeypatch.setattr(task_module, "run_ingestion", fake_run_ingestion)

    result = task_module.ingest_documents.run(job.id)

    db = session_local()
    repo = IngestionJobRepository(db)
    saved_job = repo.get_job(job.id)
    assert result["chunks_created"] == 25
    assert saved_job is not None
    assert saved_job.status == "completed"
    assert saved_job.pdf_files == 2
    assert saved_job.pages_loaded == 10
    assert saved_job.pages_ocr == 1
    assert saved_job.chunks_created == 25
    assert saved_job.started_at is not None
    assert saved_job.finished_at is not None
    db.close()


def test_ingestion_task_marks_job_failed(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    db = session_local()
    repo = IngestionJobRepository(db)
    job = repo.create_job()
    db.close()

    def fake_run_ingestion(settings, document_ids_by_source=None):
        raise RuntimeError("PDF extraction failed")

    monkeypatch.setattr(task_module, "SessionLocal", session_local)
    monkeypatch.setattr(task_module, "run_ingestion", fake_run_ingestion)

    try:
        task_module.ingest_documents.run(job.id)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected ingestion task to raise RuntimeError")

    db = session_local()
    repo = IngestionJobRepository(db)
    saved_job = repo.get_job(job.id)
    assert saved_job is not None
    assert saved_job.status == "failed"
    assert saved_job.started_at is not None
    assert saved_job.finished_at is not None
    assert saved_job.error_message == "PDF extraction failed"
    db.close()


def test_ingestion_task_processes_single_document_job(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    uploaded_pdf_path = tmp_path / "uploaded.pdf"
    uploaded_pdf_path.write_bytes(b"%PDF-1.4\ncontent")

    db = session_local()
    repo = IngestionJobRepository(db)
    document = repo.create_uploaded_document(
        source_path="uploads/uploaded.pdf",
        filename="uploaded.pdf",
        storage_path=str(uploaded_pdf_path),
    )
    job = repo.create_job(document_id=document.id)
    expected_document_id = document.id
    db.close()

    def fake_run_document_ingestion(settings, pdf_path, document_id):
        assert pdf_path == uploaded_pdf_path
        assert document_id == expected_document_id
        return FakeStats(pdf_files=1, pages_loaded=3, pages_ocr=1, chunks_created=9)

    def unexpected_full_ingestion(settings, document_ids_by_source=None):
        raise AssertionError("Full ingestion should not run for document jobs")

    def fake_settings():
        return Settings(data_dir=tmp_path, vector_backend="qdrant")

    monkeypatch.setattr(task_module, "Settings", fake_settings)
    monkeypatch.setattr(task_module, "SessionLocal", session_local)
    monkeypatch.setattr(task_module, "run_document_ingestion", fake_run_document_ingestion)
    monkeypatch.setattr(task_module, "run_ingestion", unexpected_full_ingestion)

    result = task_module.ingest_documents.run(job.id)

    db = session_local()
    repo = IngestionJobRepository(db)
    saved_job = repo.get_job(job.id)
    saved_document = db.get(Document, document.id)
    assert result["chunks_created"] == 9
    assert saved_job is not None
    assert saved_job.status == "completed"
    assert saved_job.document_id == expected_document_id
    assert saved_job.pdf_files == 1
    assert saved_job.pages_loaded == 3
    assert saved_job.chunks_created == 9
    assert saved_document is not None
    assert saved_document.status == "indexed"
    assert saved_document.page_count == 3
    assert saved_document.indexed_at is not None
    db.close()


def test_full_qdrant_ingestion_registers_documents_and_passes_ids(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    data_dir = tmp_path / "Data"
    standards_dir = data_dir / "standards"
    standards_dir.mkdir(parents=True)
    pdf_path = standards_dir / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ncontent")

    db = session_local()
    repo = IngestionJobRepository(db)
    job = repo.create_job()
    db.close()

    def fake_settings():
        return Settings(data_dir=data_dir, vector_backend="qdrant")

    def fake_run_ingestion(settings, document_ids_by_source=None):
        assert document_ids_by_source == {"standards/manual.pdf": 1}
        return FakeStats(pdf_files=1, pages_loaded=7, pages_ocr=0, chunks_created=14)

    monkeypatch.setattr(task_module, "Settings", fake_settings)
    monkeypatch.setattr(task_module, "SessionLocal", session_local)
    monkeypatch.setattr(task_module, "run_ingestion", fake_run_ingestion)

    result = task_module.ingest_documents.run(job.id)

    db = session_local()
    repo = IngestionJobRepository(db)
    saved_job = repo.get_job(job.id)
    documents = repo.list_documents()

    assert result["chunks_created"] == 14
    assert saved_job is not None
    assert saved_job.status == "completed"
    assert len(documents) == 1
    assert documents[0].source_path == "standards/manual.pdf"
    assert documents[0].filename == "manual.pdf"
    assert documents[0].status == "indexed"
    assert documents[0].indexed_at is not None
    db.close()


def test_ingestion_task_resolves_relative_document_storage_path(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    data_dir = tmp_path / "Data"
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True)
    uploaded_pdf_path = uploads_dir / "uploaded.pdf"
    uploaded_pdf_path.write_bytes(b"%PDF-1.4\ncontent")

    db = session_local()
    repo = IngestionJobRepository(db)
    document = repo.create_uploaded_document(
        source_path="uploads/uploaded.pdf",
        filename="uploaded.pdf",
        storage_path="Data/uploads/uploaded.pdf",
    )
    job = repo.create_job(document_id=document.id)
    expected_document_id = document.id
    db.close()

    def fake_settings():
        return Settings(data_dir=data_dir, vector_backend="qdrant")

    def fake_run_document_ingestion(settings, pdf_path, document_id):
        assert pdf_path == uploaded_pdf_path.resolve()
        assert document_id == expected_document_id
        return FakeStats(pdf_files=1, pages_loaded=2, pages_ocr=0, chunks_created=4)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(task_module, "Settings", fake_settings)
    monkeypatch.setattr(task_module, "SessionLocal", session_local)
    monkeypatch.setattr(task_module, "run_document_ingestion", fake_run_document_ingestion)

    result = task_module.ingest_documents.run(job.id)

    assert result["chunks_created"] == 4
