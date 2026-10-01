from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_RRF_K,
    DEFAULT_SOURCE_WEIGHTS,
    HISTORY_SOURCE_K,
    MAX_CANDIDATES,
    POPULARITY_SOURCE_K,
    SVD_SOURCE_K,
)


class HybridCandidateGenerator:
    """
    Notebook-05 v2 retriever:
      history + SVD collaborative retrieval + popularity
      -> weighted reciprocal-rank fusion
      -> ~200 candidates
    """

    def __init__(
        self,
        retriever_bundle: dict,
        pair_history: pd.DataFrame,
        streamer_stats: pd.DataFrame,
    ):
        self.user_factors = np.asarray(
            retriever_bundle["user_factors"],
            dtype=np.float32,
        )
        self.item_factors = np.asarray(
            retriever_bundle["item_factors"],
            dtype=np.float32,
        )
        self.user_to_idx = retriever_bundle["user_to_idx"]
        self.streamers = list(retriever_bundle["streamers"])

        self.source_weights = dict(
            retriever_bundle.get(
                "source_weights",
                DEFAULT_SOURCE_WEIGHTS,
            )
        )
        self.rrf_k = int(
            retriever_bundle.get("rrf_k", DEFAULT_RRF_K)
        )

        hist = pair_history.copy()
        hist["history_strength"] = (
            np.log1p(hist["interactions"].astype(float))
            + 0.25 * np.log1p(hist["watch_minutes"].astype(float))
        )

        hist = hist.sort_values(
            ["user_id", "history_strength"],
            ascending=[True, False],
        )

        self.history_lookup = {
            user_id: list(
                zip(
                    group["streamer"].tolist(),
                    group["history_strength"].astype(float).tolist(),
                )
            )
            for user_id, group in hist.groupby("user_id", sort=False)
        }

        if "streamer_history_interactions" in streamer_stats.columns:
            popularity = streamer_stats.sort_values(
                "streamer_history_interactions",
                ascending=False,
            )
        else:
            popularity = streamer_stats.copy()

        self.popular_streamers = popularity["streamer"].tolist()

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
    ) -> "HybridCandidateGenerator":
        artifact_dir = Path(artifact_dir)

        retriever_path = artifact_dir / "hybrid_candidate_retriever.joblib"
        pair_path = artifact_dir / "retrieval_pair_history.csv.gz"
        streamer_path = artifact_dir / "retrieval_streamer_stats.csv.gz"

        for path in (retriever_path, pair_path, streamer_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing retrieval artifact: {path}")

        bundle = joblib.load(retriever_path)
        pair_history = pd.read_csv(pair_path)
        streamer_stats = pd.read_csv(streamer_path)

        return cls(
            retriever_bundle=bundle,
            pair_history=pair_history,
            streamer_stats=streamer_stats,
        )

    def retrieve_history(
        self,
        user_id: int,
        top_k: int = HISTORY_SOURCE_K,
    ) -> list[tuple[str, float]]:
        return [
            (streamer, float(score))
            for streamer, score
            in self.history_lookup.get(user_id, [])[:top_k]
        ]

    def retrieve_svd(
        self,
        user_id: int,
        top_k: int = SVD_SOURCE_K,
    ) -> list[tuple[str, float]]:
        if user_id not in self.user_to_idx:
            return []

        uidx = self.user_to_idx[user_id]

        scores = self.user_factors[uidx] @ self.item_factors.T

        k = min(int(top_k), len(scores))
        if k <= 0:
            return []

        if k == len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[
                np.argsort(scores[top_idx])[::-1]
            ]

        return [
            (
                self.streamers[int(idx)],
                float(scores[int(idx)]),
            )
            for idx in top_idx[:k]
        ]

    def retrieve_popularity(
        self,
        top_k: int = POPULARITY_SOURCE_K,
    ) -> list[tuple[str, float]]:
        return [
            (
                streamer,
                float(1.0 / math.log2(rank + 2)),
            )
            for rank, streamer in enumerate(
                self.popular_streamers[:top_k],
                start=1,
            )
        ]

    def generate(
        self,
        user_id: int,
        max_candidates: int = MAX_CANDIDATES,
    ) -> pd.DataFrame:
        sources = {
            "history": self.retrieve_history(
                user_id, HISTORY_SOURCE_K
            ),
            "svd": self.retrieve_svd(
                user_id, SVD_SOURCE_K
            ),
            "popularity": self.retrieve_popularity(
                POPULARITY_SOURCE_K
            ),
        }

        candidate_info: dict[str, dict] = {}

        for source_name, items in sources.items():
            source_weight = self.source_weights[source_name]

            for rank, (streamer, raw_score) in enumerate(items, start=1):
                row = candidate_info.setdefault(
                    streamer,
                    {
                        "fusion_score": 0.0,
                        "sources": set(),
                        "best_source_rank": {},
                        "raw_scores": {},
                    },
                )

                row["fusion_score"] += (
                    source_weight / (self.rrf_k + rank)
                )
                row["sources"].add(source_name)
                row["best_source_rank"][source_name] = rank
                row["raw_scores"][source_name] = float(raw_score)

        ranked = sorted(
            candidate_info.items(),
            key=lambda x: x[1]["fusion_score"],
            reverse=True,
        )[:max_candidates]

        rows = []
        for final_rank, (streamer, meta) in enumerate(ranked, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "streamer": streamer,
                    "retrieval_rank": final_rank,
                    "retrieval_score": float(meta["fusion_score"]),
                    "candidate_sources": "|".join(
                        sorted(meta["sources"])
                    ),
                    "history_rank": meta["best_source_rank"].get(
                        "history", np.nan
                    ),
                    "svd_rank": meta["best_source_rank"].get(
                        "svd", np.nan
                    ),
                    "popularity_rank": meta["best_source_rank"].get(
                        "popularity", np.nan
                    ),
                }
            )

        return pd.DataFrame(rows)
