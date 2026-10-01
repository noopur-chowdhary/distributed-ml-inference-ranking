from __future__ import annotations
from pathlib import Path

import json
import os
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from ray import serve

from src.cache.redis_cache import RedisRecommendationCache
from src.inference.engine import RankingInferenceEngine


NUM_REPLICAS = int(
    os.getenv("RANKING_REPLICAS", "2")
)


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": 1,
    },
)
class RankingReplica:

    def __init__(self):
        print(
            f"Loading RankingInferenceEngine "
            f"in replica PID={os.getpid()}"
        )

        self.engine = (
            RankingInferenceEngine
            .from_project_root(
                project_root=Path(__file__).resolve().parents[2]
            )
        )

        print(
            f"Replica ready PID={os.getpid()}"
        )

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        candidate_k: int = 200,
    ):
        start = perf_counter()

        recommendations = self.engine.recommend(
            user_id=user_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        inference_latency_ms = (
            perf_counter() - start
        ) * 1000

        records = json.loads(
            recommendations.to_json(
                orient="records"
            )
        )

        return {
            "user_id": user_id,
            "candidate_count": candidate_k,
            "recommendation_count": len(records),
            "inference_latency_ms": round(
                inference_latency_ms,
                3,
            ),
            "replica_pid": os.getpid(),
            "recommendations": records,
        }


fastapi_app = FastAPI(
    title="Distributed ML Inference & Ranking",
    version="3.0.0",
)


@serve.deployment
@serve.ingress(fastapi_app)
class APIGateway:

    def __init__(
        self,
        ranking_handle,
    ):
        self.ranking = ranking_handle
        self.cache = RedisRecommendationCache()

    @fastapi_app.get("/health")
    async def health(self):
        return {
            "status": "ok",
            "service": "distributed-ml-ranking",
            "serving": "ray-serve",
            "ranking_replicas": NUM_REPLICAS,
            "ranker": "HistGradientBoosting",
            "feature_count": 36,
            "redis_cache": self.cache.ping(),
            "cache_ttl_seconds": (
                self.cache.ttl_seconds
            ),
        }

    @fastapi_app.get(
        "/recommend/{user_id}"
    )
    async def recommend(
        self,
        user_id: int,
        top_k: int = Query(
            default=10,
            ge=1,
            le=50,
        ),
        candidate_k: int = Query(
            default=200,
            ge=10,
            le=500,
        ),
    ):
        if top_k > candidate_k:
            raise HTTPException(
                status_code=400,
                detail=(
                    "top_k cannot be greater "
                    "than candidate_k."
                ),
            )

        request_start = perf_counter()

        cached, cache_lookup_ms = (
            self.cache.get(
                user_id=user_id,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        )

        if cached is not None:
            gateway_latency_ms = (
                perf_counter() - request_start
            ) * 1000

            return {
                "user_id": user_id,
                "candidate_count": candidate_k,
                "recommendation_count": len(cached),
                "cache_hit": True,
                "cache_lookup_ms": round(
                    cache_lookup_ms,
                    3,
                ),
                "cache_store_ms": 0.0,
                "inference_latency_ms": 0.0,
                "gateway_latency_ms": round(
                    gateway_latency_ms,
                    3,
                ),
                "replica_pid": None,
                "recommendations": cached,
            }

        result = await (
            self.ranking
            .recommend
            .remote(
                user_id=user_id,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        )

        cache_store_ms = self.cache.set(
            user_id=user_id,
            top_k=top_k,
            candidate_k=candidate_k,
            recommendations=result[
                "recommendations"
            ],
        )

        gateway_latency_ms = (
            perf_counter() - request_start
        ) * 1000

        result["cache_hit"] = False

        result["cache_lookup_ms"] = round(
            cache_lookup_ms,
            3,
        )

        result["cache_store_ms"] = round(
            cache_store_ms,
            3,
        )

        result["gateway_latency_ms"] = round(
            gateway_latency_ms,
            3,
        )

        return result


ranking = RankingReplica.bind()

serve_app = APIGateway.bind(
    ranking
)
