from __future__ import annotations

import hashlib
import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from electrical_rag.core.settings import Settings


class ChatCache:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.enable_chat_cache
        self.client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )

    @staticmethod
    def normalize_question(question: str) -> str:
        return " ".join(question.strip().lower().split())

    @classmethod
    def cache_key(cls, question: str) -> str:
        normalized = cls.normalize_question(question)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"electrical_rag:chat:{digest}"

    def get(self, question: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        try:
            raw_value = self.client.get(self.cache_key(question))
        except RedisError:
            return None

        if not raw_value:
            return None

        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            return None

        if not isinstance(value, dict):
            return None

        return value

    def set(self, question: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            self.client.set(
                self.cache_key(question),
                json.dumps(payload, ensure_ascii=False),
                ex=self.settings.chat_cache_ttl_seconds,
            )
        except RedisError:
            return
