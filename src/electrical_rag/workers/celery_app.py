from __future__ import annotations

from celery import Celery

from electrical_rag.core.settings import Settings
from electrical_rag.observability.logging import configure_logging

settings = Settings()
configure_logging(settings.app_log_level, settings.app_log_format)

celery_app = Celery(
    "electrical_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["electrical_rag.workers.ingestion_tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_hijack_root_logger=False,
)
