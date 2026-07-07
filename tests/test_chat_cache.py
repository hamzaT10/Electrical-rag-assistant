from __future__ import annotations

import json

from redis.exceptions import RedisError

from electrical_rag.cache.redis_cache import ChatCache


class DummySettings:
    redis_url = "redis://localhost:6379/0"
    enable_chat_cache = True
    chat_cache_ttl_seconds = 3600


class FailingRedis:
    def get(self, key: str):
        raise RedisError("redis unavailable")

    def set(self, key: str, value: str, ex: int):
        raise RedisError("redis unavailable")


class MemoryRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int):
        self.values[key] = value
        self.ttls[key] = ex


def test_chat_cache_normalizes_questions():
    assert ChatCache.normalize_question("  What   IS Voltage? ") == "what is voltage?"


def test_chat_cache_key_is_stable_for_same_question():
    key_a = ChatCache.cache_key("What is voltage unbalance?")
    key_b = ChatCache.cache_key(" what   is VOLTAGE unbalance? ")

    assert key_a == key_b
    assert key_a.startswith("electrical_rag:chat:")


def test_chat_cache_returns_none_when_redis_fails():
    cache = ChatCache(DummySettings())  # type: ignore[arg-type]
    cache.client = FailingRedis()  # type: ignore[assignment]

    assert cache.get("What is voltage unbalance?") is None
    cache.set("What is voltage unbalance?", {"answer": "ignored"})


def test_chat_cache_stores_payload_with_ttl():
    cache = ChatCache(DummySettings())  # type: ignore[arg-type]
    memory_redis = MemoryRedis()
    cache.client = memory_redis  # type: ignore[assignment]
    payload = {"answer": "Voltage unbalance is...", "citations": []}

    cache.set("What is voltage unbalance?", payload)

    key = ChatCache.cache_key("what is voltage unbalance?")
    assert json.loads(memory_redis.values[key]) == payload
    assert memory_redis.ttls[key] == 3600
    assert cache.get(" what   is VOLTAGE unbalance? ") == payload
