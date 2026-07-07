from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

from electrical_rag.core.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TraceHandle:
    observation: Any = None
    trace_id: str | None = None

    def update(self, **fields: Any) -> None:
        if self.observation is None:
            return
        try:
            self.observation.update(**fields)
        except Exception as exc:
            logger.warning(
                "langfuse_observation_update_failed",
                extra={"error_type": type(exc).__name__},
            )


class TraceManager:
    def __init__(self, settings: Settings):
        self.enabled = (
            settings.enable_langfuse
            and bool(settings.langfuse_public_key)
            and bool(settings.langfuse_secret_key)
        )
        self.capture_content = settings.langfuse_capture_content
        self.client: Any = None
        self._propagate_attributes: Any = None

        if not self.enabled:
            return

        try:
            from langfuse import Langfuse, propagate_attributes

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                environment=settings.langfuse_environment,
                release=settings.app_release,
                sample_rate=settings.langfuse_sample_rate,
            )
            self._propagate_attributes = propagate_attributes
        except Exception as exc:
            self.enabled = False
            logger.warning(
                "langfuse_initialization_failed",
                extra={"error_type": type(exc).__name__},
            )

    def content(self, value: str, field_name: str) -> str | dict[str, int]:
        if self.capture_content:
            return value
        return {f"{field_name}_chars": len(value)}

    @contextmanager
    def trace(
        self,
        *,
        name: str,
        request_id: str,
        mode: str,
        input_data: Any = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceHandle]:
        if self.client is None or self._propagate_attributes is None:
            yield TraceHandle()
            return

        stack = ExitStack()
        trace_id: str | None = None
        try:
            trace_id = self.client.create_trace_id(seed=request_id)
            observation = stack.enter_context(
                self.client.start_as_current_observation(
                    trace_context={"trace_id": trace_id},
                    name=name,
                    as_type="chain",
                    input=input_data,
                    metadata={"request_id": request_id, "mode": mode, **(metadata or {})},
                )
            )
            stack.enter_context(
                self._propagate_attributes(
                    user_id=user_id,
                    session_id=session_id,
                    trace_name=name,
                    tags=["electrical_rag", "rag", mode],
                )
            )
        except Exception as exc:
            stack.close()
            logger.warning(
                "langfuse_trace_start_failed",
                extra={"error_type": type(exc).__name__},
            )
            yield TraceHandle()
            return

        try:
            yield TraceHandle(observation=observation, trace_id=trace_id)
        finally:
            try:
                stack.close()
            except Exception as exc:
                logger.warning(
                    "langfuse_trace_close_failed",
                    extra={"error_type": type(exc).__name__},
                )

    @contextmanager
    def span(
        self,
        *,
        name: str,
        as_type: str = "span",
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[TraceHandle]:
        if self.client is None:
            yield TraceHandle()
            return

        stack = ExitStack()
        try:
            observation = stack.enter_context(
                self.client.start_as_current_observation(
                    name=name,
                    as_type=as_type,
                    input=input_data,
                    metadata=metadata,
                    model=model,
                    model_parameters=model_parameters,
                )
            )
        except Exception as exc:
            stack.close()
            logger.warning(
                "langfuse_span_start_failed",
                extra={"span_name": name, "error_type": type(exc).__name__},
            )
            yield TraceHandle()
            return

        try:
            yield TraceHandle(observation=observation)
        finally:
            try:
                stack.close()
            except Exception as exc:
                logger.warning(
                    "langfuse_span_close_failed",
                    extra={"span_name": name, "error_type": type(exc).__name__},
                )

    def flush(self) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:
            logger.warning(
                "langfuse_flush_failed",
                extra={"error_type": type(exc).__name__},
            )
