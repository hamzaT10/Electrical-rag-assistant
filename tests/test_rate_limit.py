from __future__ import annotations

from redis.exceptions import RedisError

from electrical_rag.security.rate_limit import RedisRateLimiter


class DummySettings:
    redis_url = "redis://localhost:6379/0"
    enable_rate_limit = True
    rate_limit_requests = 2
    rate_limit_window_seconds = 60


class DisabledSettings(DummySettings):
    enable_rate_limit = False


class MemoryRedis:
    def __init__(self):
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)


class FailingRedis:
    def incr(self, key: str):
        raise RedisError("redis unavailable")


def test_rate_limiter_allows_until_limit():
    limiter = RedisRateLimiter(DummySettings())  # type: ignore[arg-type]
    limiter.client = MemoryRedis()  # type: ignore[assignment]

    first = limiter.check("127.0.0.1")
    second = limiter.check("127.0.0.1")

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0


def test_rate_limiter_blocks_after_limit():
    limiter = RedisRateLimiter(DummySettings())  # type: ignore[arg-type]
    limiter.client = MemoryRedis()  # type: ignore[assignment]

    limiter.check("127.0.0.1")
    limiter.check("127.0.0.1")
    third = limiter.check("127.0.0.1")

    assert third.allowed is False
    assert third.remaining == 0
    assert third.retry_after_seconds == 60


def test_rate_limiter_can_be_disabled():
    limiter = RedisRateLimiter(DisabledSettings())  # type: ignore[arg-type]
    limiter.client = MemoryRedis()  # type: ignore[assignment]

    result = limiter.check("127.0.0.1")

    assert result.allowed is True
    assert result.remaining == 2


def test_rate_limiter_fails_open_when_redis_fails():
    limiter = RedisRateLimiter(DummySettings())  # type: ignore[arg-type]
    limiter.client = FailingRedis()  # type: ignore[assignment]

    result = limiter.check("127.0.0.1")

    assert result.allowed is True
    assert result.remaining == 2
