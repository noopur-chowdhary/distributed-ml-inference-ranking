from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ProjectPaths, discover_project_root
from src.data.twitch_loader import (
    load_clean_twitch_history,
    read_cutoff_from_metadata,
)
from src.features.dynamic_features import DynamicFeatureEngine
from src.features.feature_store import (
    FeatureStore,
    infer_unseen_pair_recency_from_reference,
)
from src.ranking.ranker import HGBRanker
from src.retrieval.candidate_generator import HybridCandidateGenerator


class RankingInferenceEngine:
    """
    End-to-end local inference path:

      user_id
        -> hybrid candidate generator
        -> ~200 candidates
        -> exact dynamic 36-feature join
        -> HGB scoring
        -> Top-K
    """

    def __init__(
        self,
        retriever: HybridCandidateGenerator,
        feature_engine: DynamicFeatureEngine,
        ranker: HGBRanker,
    ):
        self.retriever = retriever
        self.feature_engine = feature_engine
        self.ranker = ranker

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path | None = None,
        rebuild_feature_store: bool = False,
    ) -> "RankingInferenceEngine":
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else discover_project_root()
        )

        paths = ProjectPaths.from_root(root)
        artifact_dir = paths.artifacts_twitch

        metadata_path = (
            artifact_dir / "candidate_generation_v2_metadata.json"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Candidate-generation metadata missing: {metadata_path}"
            )

        cutoff = read_cutoff_from_metadata(metadata_path)

        store_path = artifact_dir / "feature_store_36.joblib"

        if store_path.exists() and not rebuild_feature_store:
            store = FeatureStore.load(store_path)
            if store.cutoff != cutoff:
                raise ValueError(
                    "Saved feature store cutoff does not match candidate "
                    "retriever cutoff. Rebuild the feature store."
                )
        else:
            reference_path = (
                paths.processed_twitch / "twitch_test_ranking.csv.gz"
            )

            unseen_recency = (
                infer_unseen_pair_recency_from_reference(reference_path)
            )

            history = load_clean_twitch_history(
                paths.raw_twitch,
                cutoff=cutoff,
            )

            store = FeatureStore.from_history(
                history=history,
                cutoff=cutoff,
                unseen_pair_recency=unseen_recency,
            )
            store.save(store_path)

        retriever = HybridCandidateGenerator.from_artifacts(
            artifact_dir
        )

        feature_engine = DynamicFeatureEngine(store)

        ranker = HGBRanker.from_artifact(
            artifact_dir
            / "baseline_hist_gradient_boosting.joblib"
        )

        return cls(
            retriever=retriever,
            feature_engine=feature_engine,
            ranker=ranker,
        )

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        candidate_k: int = 200,
    ) -> pd.DataFrame:
        candidates = self.retriever.generate(
            user_id=user_id,
            max_candidates=candidate_k,
        )

        if candidates.empty:
            return candidates

        feature_frame = self.feature_engine.transform(candidates)

        ranked = self.ranker.rank(
            feature_frame,
            top_k=top_k,
        )

        preferred_columns = [
            "user_id",
            "streamer",
            "rank_position",
            "rank_score",
            "retrieval_rank",
            "retrieval_score",
            "candidate_sources",
            "seen_before",
            "pair_interactions",
            "pair_watch_minutes",
            "streamer_interactions",
            "streamer_trend_ratio",
        ]

        return ranked[
            [
                c for c in preferred_columns
                if c in ranked.columns
            ]
        ].copy()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local Twitch ranking inference."
    )
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--rebuild-feature-store",
        action="store_true",
    )

    args = parser.parse_args()

    engine = RankingInferenceEngine.from_project_root(
        project_root=args.project_root,
        rebuild_feature_store=args.rebuild_feature_store,
    )

    result = engine.recommend(
        user_id=args.user_id,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    _main()
