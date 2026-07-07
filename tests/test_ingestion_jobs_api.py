from __future__ import annotations

import importlib
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from electrical_rag.db.session import Base, get_db_session

app_module = importlib.import_module("electrical_rag.api.app")


def test_create_and_fetch_ingestion_job(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    def override_db() -> Generator[Session, None, None]:
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    class FakeAsyncResult:
        id = "celery-task-123"

    def fake_delay(job_id: int) -> FakeAsyncResult:
        assert job_id == 1
        return FakeAsyncResult()

    monkeypatch.setattr(app_module.ingest_documents, "delay", fake_delay)
    app_module.app.dependency_overrides[get_db_session] = override_db
    try:
        client = TestClient(app_module.app)

        create_response = client.post(
            "/ingestion/jobs",
            json={"source_path": "Data/Standards"},
        )
        assert create_response.status_code == 202
        created = create_response.json()
        assert created["id"] == 1
        assert created["status"] == "pending"
        assert created["source_path"] == "Data/Standards"
        assert created["task_id"] == "celery-task-123"

        get_response = client.get("/ingestion/jobs/1")
        assert get_response.status_code == 200
        assert get_response.json()["task_id"] == "celery-task-123"

        list_response = client.get("/ingestion/jobs")
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1

        missing_response = client.get("/ingestion/jobs/999")
        assert missing_response.status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()


def test_create_ingestion_job_marks_failed_when_enqueue_fails(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    def override_db() -> Generator[Session, None, None]:
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    def fake_delay(job_id: int):
        raise RuntimeError(f"broker unavailable for job {job_id}")

    monkeypatch.setattr(app_module.ingest_documents, "delay", fake_delay)
    app_module.app.dependency_overrides[get_db_session] = override_db
    try:
        client = TestClient(app_module.app)

        create_response = client.post("/ingestion/jobs", json={})
        assert create_response.status_code == 503

        list_response = client.get("/ingestion/jobs")
        assert list_response.status_code == 200
        jobs = list_response.json()
        assert jobs[0]["status"] == "failed"
        assert "Failed to enqueue ingestion task" in jobs[0]["error_message"]
    finally:
        app_module.app.dependency_overrides.clear()
