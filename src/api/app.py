from __future__ import annotations

import json
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query

from src.inference.engine import RankingInferenceEngine


engine: RankingInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    print("Loading ranking inference engine...")
    engine = RankingInferenceEngine.from_project_root()

    print("Inference engine ready.")

    yield

    engine = None


app = FastAPI(
    title="Distributed ML Inference & Ranking",
    version="1.0.0",
    description=(
        "Two-stage recommendation service using hybrid candidate retrieval, "
        "36 dynamic ranking features, and HistGradientBoosting ranking."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not ready.",
        )

    return {
        "status": "ok",
        "service": "distributed-ml-ranking",
        "ranker": "HistGradientBoosting",
        "feature_count": 36,
    }


@app.get("/recommend/{user_id}")
def recommend(
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
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not ready.",
        )

    if top_k > candidate_k:
        raise HTTPException(
            status_code=400,
            detail="top_k cannot be greater than candidate_k.",
        )

    start = perf_counter()

    try:
        recommendations = engine.recommend(
            user_id=user_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    latency_ms = (perf_counter() - start) * 1000

    # DataFrame -> JSON-safe native Python values.
    records = json.loads(
        recommendations.to_json(
            orient="records",
        )
    )

    return {
        "user_id": user_id,
        "candidate_count": candidate_k,
        "recommendation_count": len(records),
        "latency_ms": round(latency_ms, 3),
        "recommendations": records,
    }
