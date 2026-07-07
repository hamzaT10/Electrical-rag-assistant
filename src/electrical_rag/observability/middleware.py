from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from electrical_rag.observability.context import bind_request_id, reset_request_id
from electrical_rag.observability.metrics import (
    HTTP_REQUESTS_IN_PROGRESS,
    record_http_request,
)

logger = logging.getLogger(__name__)

RequestIdHeader = tuple[bytes, bytes]
Send = Callable[[dict[str, Any]], Awaitable[None]]
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _request_id(headers: list[RequestIdHeader]) -> str:
    for name, value in headers:
        if name.lower() == b"x-request-id":
            candidate = value.decode("ascii", errors="ignore")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
    return uuid.uuid4().hex


def _route_template(scope: dict[str, Any]) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "unmatched"


class ObservabilityMiddleware:
    def __init__(self, app, metrics_enabled: bool = True):
        self.app = app
        self.metrics_enabled = metrics_enabled

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[dict[str, Any]]],
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope.get("headers", []))
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_request_id(request_id)
        method = scope.get("method", "UNKNOWN")
        started_at = time.perf_counter()
        status_code = 500
        completed = False
        metrics_enabled = self.metrics_enabled and scope.get("path") != "/metrics"

        if metrics_enabled:
            HTTP_REQUESTS_IN_PROGRESS.inc()

        async def finish() -> None:
            nonlocal completed
            if completed:
                return
            completed = True
            duration_seconds = time.perf_counter() - started_at
            route = _route_template(scope)
            if metrics_enabled:
                HTTP_REQUESTS_IN_PROGRESS.dec()
                record_http_request(method, route, status_code, duration_seconds)
            logger.info(
                "http_request_completed",
                extra={
                    "method": method,
                    "route": route,
                    "status_code": status_code,
                    "duration_seconds": round(duration_seconds, 6),
                },
            )

        async def send_with_observability(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await finish()

        try:
            await self.app(scope, receive, send_with_observability)
            await finish()
        except Exception:
            await finish()
            logger.exception(
                "http_request_failed",
                extra={
                    "method": method,
                    "route": _route_template(scope),
                    "status_code": status_code,
                },
            )
            raise
        finally:
            reset_request_id(token)
