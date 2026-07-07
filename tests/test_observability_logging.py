import json
import logging

from electrical_rag.observability.context import bind_request_id, reset_request_id
from electrical_rag.observability.logging import JsonFormatter


def test_json_formatter_emits_context_and_structured_fields() -> None:
    token = bind_request_id("request-123")
    try:
        record = logging.LogRecord(
            name="electrical_rag.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=12,
            msg="rag_request_completed",
            args=(),
            exc_info=None,
        )
        record.duration_seconds = 0.125

        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["event"] == "rag_request_completed"
    assert payload["request_id"] == "request-123"
    assert payload["duration_seconds"] == 0.125
    assert payload["level"] == "INFO"
