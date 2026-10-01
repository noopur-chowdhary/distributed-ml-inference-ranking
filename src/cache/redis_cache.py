from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import redis
from redis.exceptions import RedisError


class RedisRecommendationCache:
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.ttl_seconds = int(
            os.getenv("REDIS_TTL_SECONDS", "300")
        )

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

    @staticmethod
    def _key(
        user_id: int,
        top_k: int,
        candidate_k: int,
    ) -> str:
        return (
            f"dmlir:v1:"
            f"user:{user_id}:"
            f"top:{top_k}:"
            f"cand:{candidate_k}"
        )

    def get(
        self,
        user_id: int,
        top_k: int,
        candidate_k: int,
    ) -> tuple[list[dict[str, Any]] | None, float]:

        start = perf_counter()

        try:
            raw = self.client.get(
                self._key(
                    user_id,
                    top_k,
                    candidate_k,
                )
            )

            latency_ms = (
                perf_counter() - start
            ) * 1000

            if raw is None:
                return None, latency_ms

            return json.loads(raw), latency_ms

        except RedisError:
            latency_ms = (
                perf_counter() - start
            ) * 1000

            return None, latency_ms

    def set(
        self,
        user_id: int,
        top_k: int,
        candidate_k: int,
        recommendations: list[dict[str, Any]],
    ) -> float:

        start = perf_counter()

        try:
            self.client.setex(
                self._key(
                    user_id,
                    top_k,
                    candidate_k,
                ),
                self.ttl_seconds,
                json.dumps(recommendations),
            )

        except RedisError:
            pass

        return (
            perf_counter() - start
        ) * 1000

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False
