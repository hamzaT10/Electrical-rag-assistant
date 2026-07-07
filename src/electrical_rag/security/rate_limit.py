from __future__ import annotations

from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError

from electrical_rag.core.settings import Settings


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int | None = None


class RedisRateLimiter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.enable_rate_limit
        self.client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )

    @staticmethod
    def client_key(client_id: str) -> str:
        safe_client_id = client_id.replace(":", "_")
        return f"electrical_rag:rate_limit:ip:{safe_client_id}"

    def check(self, client_id: str) -> RateLimitResult:
        limit = self.settings.rate_limit_requests
        window = self.settings.rate_limit_window_seconds

        if not self.enabled:
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
            )

        key = self.client_key(client_id)

        try:
            count = int(self.client.incr(key))
            if count == 1:
                self.client.expire(key, window)
            ttl = int(self.client.ttl(key))
        except RedisError:
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
            )

        remaining = max(limit - count, 0)

        if count > limit:
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                retry_after_seconds=ttl if ttl > 0 else window,
            )

        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=remaining,
        )
