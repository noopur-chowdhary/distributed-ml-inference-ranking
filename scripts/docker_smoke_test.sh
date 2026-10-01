#!/usr/bin/env bash

set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
USER_ID="1"
TOP_K="10"
CANDIDATE_K="200"

CACHE_KEY="dmlir:v1:user:${USER_ID}:top:${TOP_K}:cand:${CANDIDATE_K}"

FIRST_RESPONSE=$(mktemp)
SECOND_RESPONSE=$(mktemp)

cleanup() {
    rm -f "$FIRST_RESPONSE" "$SECOND_RESPONSE"
}
trap cleanup EXIT


echo "========================================"
echo "Distributed ML Docker Smoke Test"
echo "========================================"


echo
echo "[1/5] Starting Docker Compose services..."

docker compose up -d


echo
echo "[2/5] Waiting for API health..."

MAX_ATTEMPTS=30

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do

    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        echo "API is healthy."
        break
    fi

    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
        echo "ERROR: API did not become healthy."
        echo
        docker compose ps
        echo
        docker compose logs inference --tail=100
        exit 1
    fi

    echo "Waiting... attempt ${attempt}/${MAX_ATTEMPTS}"
    sleep 3

done


echo
echo "[3/5] Checking health response..."

curl -fsS "${BASE_URL}/health"

echo

python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:8000/health"
) as response:
    data = json.load(response)

assert data["status"] == "ok"
assert data["serving"] == "ray-serve"
assert data["ranking_replicas"] == 2
assert data["ranker"] == "HistGradientBoosting"
assert data["feature_count"] == 36
assert data["redis_cache"] is True

print("PASS: health checks")
PY


echo
echo "[4/5] Testing cache MISS..."

# Remove this exact recommendation from Redis so the first
# request must execute the inference pipeline.
docker compose exec -T redis \
    redis-cli DEL "$CACHE_KEY" >/dev/null

curl -fsS \
    "${BASE_URL}/recommend/${USER_ID}?top_k=${TOP_K}&candidate_k=${CANDIDATE_K}" \
    > "$FIRST_RESPONSE"

python3 - "$FIRST_RESPONSE" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    data = json.load(f)

assert data["cache_hit"] is False
assert data["candidate_count"] == 200
assert data["recommendation_count"] == 10
assert data["replica_pid"] is not None
assert data["inference_latency_ms"] > 0

print(
    "PASS: cache miss -> inference executed"
)
print(
    f"      inference latency: "
    f"{data['inference_latency_ms']:.3f} ms"
)
print(
    f"      gateway latency: "
    f"{data['gateway_latency_ms']:.3f} ms"
)
print(
    f"      replica PID: "
    f"{data['replica_pid']}"
)
PY


echo
echo "[5/5] Testing cache HIT..."

curl -fsS \
    "${BASE_URL}/recommend/${USER_ID}?top_k=${TOP_K}&candidate_k=${CANDIDATE_K}" \
    > "$SECOND_RESPONSE"

python3 - "$SECOND_RESPONSE" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    data = json.load(f)

assert data["cache_hit"] is True
assert data["candidate_count"] == 200
assert data["recommendation_count"] == 10
assert data["replica_pid"] is None
assert data["inference_latency_ms"] == 0
assert data["cache_store_ms"] == 0

print(
    "PASS: cache hit -> inference bypassed"
)
print(
    f"      cache lookup: "
    f"{data['cache_lookup_ms']:.3f} ms"
)
print(
    f"      gateway latency: "
    f"{data['gateway_latency_ms']:.3f} ms"
)
PY


echo
echo "========================================"
echo "PASS: Docker serving stack is healthy"
echo "========================================"
echo
echo "Verified:"
echo "  Docker Compose"
echo "  Redis"
echo "  Ray Serve"
echo "  FastAPI gateway"
echo "  2 ranking replicas"
echo "  36-feature HGB inference"
echo "  Redis miss -> inference"
echo "  Redis hit  -> bypass inference"
