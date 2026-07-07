from __future__ import annotations

import importlib
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from electrical_rag.db.models import Document
from electrical_rag.db.session import Base, get_db_session

app_module = importlib.import_module("electrical_rag.api.app")


def _override_db(session_local):
    def override_db() -> Generator[Session, None, None]:
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    return override_db


def test_upload_pdf_saves_file_and_enqueues_ingestion_job(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    class FakeAsyncResult:
        id = "upload-task-123"

    def fake_delay(job_id: int) -> FakeAsyncResult:
        assert job_id == 1
        return FakeAsyncResult()

    monkeypatch.setattr(app_module.ingest_documents, "delay", fake_delay)
    monkeypatch.setattr(app_module.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(app_module.settings, "max_upload_size_mb", 1)
    app_module.app.dependency_overrides[get_db_session] = _override_db(session_local)
    try:
        client = TestClient(app_module.app)
        response = client.post(
            "/documents/upload",
            files={"file": ("panel manual.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
        )

        assert response.status_code == 202
        payload = response.json()
        assert payload["filename"].startswith("panel_manual-")
        assert payload["filename"].endswith(".pdf")
        assert payload["document_id"] == 1
        assert payload["job_id"] == 1
        assert payload["job_status"] == "pending"
        assert (tmp_path / payload["filename"]).exists()

        job_response = client.get("/ingestion/jobs/1")
        assert job_response.status_code == 200
        job_payload = job_response.json()
        assert job_payload["document_id"] == 1
        assert job_payload["task_id"] == "upload-task-123"
    finally:
        app_module.app.dependency_overrides.clear()


def test_upload_rejects_non_pdf(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    monkeypatch.setattr(app_module.settings, "upload_dir", tmp_path)
    app_module.app.dependency_overrides[get_db_session] = _override_db(session_local)
    try:
        client = TestClient(app_module.app)
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )

        assert response.status_code == 400
        assert not list(tmp_path.iterdir())
    finally:
        app_module.app.dependency_overrides.clear()


def test_documents_endpoint_returns_indexed_documents_only():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    db = session_local()
    db.add_all(
        [
            Document(
                source_path="standards/indexed.pdf",
                filename="indexed.pdf",
                status="indexed",
                page_count=12,
            ),
            Document(
                source_path="uploads/pending.pdf",
                filename="pending.pdf",
                status="uploaded",
            ),
        ]
    )
    db.commit()
    db.close()

    app_module.app.dependency_overrides[get_db_session] = _override_db(session_local)
    try:
        client = TestClient(app_module.app)
        response = client.get("/documents")

        assert response.status_code == 200
        payload = response.json()
        assert [document["filename"] for document in payload] == ["indexed.pdf"]
        assert payload[0]["page_count"] == 12
    finally:
        app_module.app.dependency_overrides.clear()
