import importlib
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse, propagate_attributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from electrical_rag.core.settings import Settings
from electrical_rag.db.session import Base, get_db_session
from electrical_rag.observability.tracing import TraceManager

pytest.importorskip("langchain_huggingface")

app_module = importlib.import_module("electrical_rag.api.app")


class FakeService:
    def __init__(self) -> None:
        self.request_ids: list[str] = []

    def ask(
        self,
        question: str,
        request_id: str = "unknown",
        document_id: int | None = None,
    ):
        self.request_ids.append(request_id)
        return f"Answer for {question}", []

    def ask_stream(
        self,
        question: str,
        request_id: str = "unknown",
        document_id: int | None = None,
    ):
        self.request_ids.append(request_id)
        yield {"type": "token", "content": f"Answer for {question}"}
        yield {"type": "citations", "citations": []}


class FakeCache:
    def get(self, question: str):
        return None

    def set(self, question: str, payload: dict[str, object]) -> None:
        return None


def override_db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def test_chat_returns_request_id_header_and_passes_it_to_service(monkeypatch) -> None:
    fake_service = FakeService()
    monkeypatch.setattr(app_module, "_bootstrap_service", lambda: fake_service)
    monkeypatch.setattr(app_module, "chat_cache", FakeCache())
    app_module.app.dependency_overrides[get_db_session] = override_db_session
    client = TestClient(app_module.app)
    try:
        response = client.post("/chat", json={"question": "What is power meter?"})
    finally:
        app_module.app.dependency_overrides.clear()

    request_id = response.headers["X-Request-ID"]
    assert response.status_code == 200
    assert len(request_id) == 32
    assert fake_service.request_ids == [request_id]


def test_health_reports_llm_readiness(monkeypatch) -> None:
    monkeypatch.setattr(app_module.settings, "vectorstore_dir", Path("."))
    monkeypatch.setattr(app_module, "_check_llm_ready", lambda: (True, None))
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "vectorstore_ready": True,
        "llm_ready": True,
        "llm_error": None,
        "rag_service_ready": False,
        "rag_service_error": None,
        "rag_service_startup_seconds": None,
        "embedding_dimension": None,
    }


def test_bootstrap_warms_embedding_model_and_records_readiness(monkeypatch) -> None:
    class FakeWarmService:
        def __init__(self, settings, trace_manager=None):
            self.settings = settings

        def warmup_embeddings(self) -> int:
            return 384

    monkeypatch.setattr(app_module, "service", None)
    monkeypatch.setattr(app_module, "service_bootstrap_error", None)
    monkeypatch.setattr(app_module, "service_bootstrap_seconds", None)
    monkeypatch.setattr(app_module, "embedding_dimension", None)
    monkeypatch.setattr(app_module, "RAGService", FakeWarmService)
    monkeypatch.setattr(app_module.settings, "warmup_embedding_model", True)

    initialized = app_module._bootstrap_service()

    assert isinstance(initialized, FakeWarmService)
    assert app_module.embedding_dimension == 384
    assert app_module.service_bootstrap_error is None
    assert app_module.service_bootstrap_seconds is not None


def test_chat_stream_returns_request_id_header_and_passes_it_to_service(monkeypatch) -> None:
    fake_service = FakeService()
    monkeypatch.setattr(app_module, "_bootstrap_service", lambda: fake_service)
    client = TestClient(app_module.app)

    response = client.post("/chat/stream", json={"question": "What is power meter?"})

    request_id = response.headers["X-Request-ID"]
    assert response.status_code == 200
    assert len(request_id) == 32
    assert fake_service.request_ids == [request_id]
    assert '"type": "token"' in response.text


def test_request_id_is_added_to_non_chat_endpoints() -> None:
    client = TestClient(app_module.app)

    response = client.get("/meta")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32


def test_valid_incoming_request_id_is_preserved() -> None:
    client = TestClient(app_module.app)

    response = client.get("/meta", headers={"X-Request-ID": "frontend-request-42"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "frontend-request-42"


def test_invalid_incoming_request_id_is_replaced() -> None:
    client = TestClient(app_module.app)

    response = client.get("/meta", headers={"X-Request-ID": "invalid request id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "invalid request id"
    assert len(response.headers["X-Request-ID"]) == 32


def test_metrics_endpoint_exposes_http_and_rag_metrics() -> None:
    client = TestClient(app_module.app)
    client.get("/meta")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "electrical_rag_http_requests_total" in response.text
    assert 'method="GET",route="/meta",status="200"' in response.text
    assert "electrical_rag_rag_stage_duration_seconds" in response.text


def test_streaming_langfuse_trace_closes_in_same_context(monkeypatch, caplog) -> None:
    exporter = InMemorySpanExporter()
    langfuse = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        span_exporter=exporter,
    )
    tracing = TraceManager(Settings(enable_langfuse=False))
    tracing.enabled = True
    tracing.client = langfuse
    tracing._propagate_attributes = propagate_attributes

    monkeypatch.setattr(app_module, "_bootstrap_service", FakeService)
    monkeypatch.setattr(app_module, "trace_manager", tracing)
    client = TestClient(app_module.app)

    response = client.post("/chat/stream", json={"question": "What is power meter?"})
    tracing.flush()

    assert response.status_code == 200
    assert any(span.name == "electrical-rag-chat" for span in exporter.get_finished_spans())
    assert "Failed to detach context" not in caplog.text
