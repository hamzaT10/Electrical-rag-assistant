from __future__ import annotations

from contextlib import contextmanager

from electrical_rag.core.settings import Settings
from electrical_rag.observability.tracing import TraceManager


class FakeObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **fields) -> None:
        self.updates.append(fields)


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.observations: list[FakeObservation] = []
        self.flushed = False

    def create_trace_id(self, seed: str) -> str:
        assert seed == "request-42"
        return "a" * 32

    @contextmanager
    def start_as_current_observation(self, **fields):
        self.started.append(fields)
        observation = FakeObservation()
        self.observations.append(observation)
        yield observation

    def flush(self) -> None:
        self.flushed = True


def _manager(capture_content: bool = False) -> tuple[TraceManager, FakeLangfuseClient]:
    manager = TraceManager(
        Settings(
            enable_langfuse=False,
            langfuse_capture_content=capture_content,
        )
    )
    client = FakeLangfuseClient()
    manager.enabled = True
    manager.client = client

    @contextmanager
    def propagate_attributes(**fields):
        manager.propagated_fields = fields
        yield

    manager._propagate_attributes = propagate_attributes
    return manager, client


def test_trace_uses_request_id_and_propagates_session_attributes() -> None:
    manager, client = _manager()

    with manager.trace(
        name="electrical-rag-chat",
        request_id="request-42",
        mode="sync",
        input_data={"question_chars": 12},
        user_id="7",
        session_id="19",
    ) as trace:
        trace.update(output={"answer_chars": 40})

    assert trace.trace_id == "a" * 32
    assert client.started[0]["trace_context"] == {"trace_id": "a" * 32}
    assert client.started[0]["as_type"] == "chain"
    assert manager.propagated_fields["user_id"] == "7"
    assert manager.propagated_fields["session_id"] == "19"
    assert client.observations[0].updates == [{"output": {"answer_chars": 40}}]


def test_nested_span_records_generation_metadata() -> None:
    manager, client = _manager()

    with manager.span(
        name="generate-answer",
        as_type="generation",
        model="openai-compatible-model",
        model_parameters={"temperature": 0.1},
    ) as span:
        span.update(output={"answer_chars": 22})

    assert client.started[0]["as_type"] == "generation"
    assert client.started[0]["model"] == "openai-compatible-model"
    assert client.observations[0].updates == [{"output": {"answer_chars": 22}}]


def test_content_is_redacted_by_default_and_opt_in_when_enabled() -> None:
    private_manager, _ = _manager(capture_content=False)
    capture_manager, _ = _manager(capture_content=True)

    assert private_manager.content("secret question", "question") == {
        "question_chars": 15
    }
    assert capture_manager.content("secret question", "question") == "secret question"


def test_trace_start_failure_degrades_to_noop() -> None:
    manager, _ = _manager()

    class FailingClient:
        def create_trace_id(self, seed: str) -> str:
            raise RuntimeError("tracing unavailable")

    manager.client = FailingClient()

    with manager.trace(
        name="electrical-rag-chat",
        request_id="request-42",
        mode="sync",
    ) as trace:
        trace.update(output="still works")

    assert trace.trace_id is None


def test_flush_is_forwarded_to_langfuse_client() -> None:
    manager, client = _manager()

    manager.flush()

    assert client.flushed is True
