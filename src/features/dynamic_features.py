from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.constants import FEATURE_COLUMNS
from src.features.feature_store import FeatureStore


PAIR_ZERO_COLUMNS = [
    "pair_interactions",
    "pair_watch_minutes",
    "pair_avg_watch_minutes",
    "pair_active_span_intervals",
    "pair_repeat_strength",
    "pair_interactions_last_6",
    "pair_interactions_last_18",
    "pair_interactions_last_42",
]

USER_ZERO_COLUMNS = [
    "user_interactions",
    "user_unique_streamers",
    "user_total_watch_minutes",
    "user_avg_watch_minutes",
    "user_median_watch_minutes",
    "user_active_span_intervals",
    "user_last_activity_recency",
    "user_repeat_rate",
    "user_interactions_last_6_intervals",
    "user_interactions_last_18_intervals",
    "user_interactions_last_42_intervals",
]


class DynamicFeatureEngine:
    """
    Fast serving-time feature construction.

    Instead of merging candidates against full historical tables,
    indexed lookups retrieve only the rows required for this request.
    """

    def __init__(
        self,
        store: FeatureStore,
    ):
        self.store = store

    def transform(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "user_id",
            "streamer",
        }

        missing = required.difference(
            candidates.columns
        )

        if missing:
            raise ValueError(
                "Candidate frame missing required "
                f"columns: {sorted(missing)}"
            )

        result = (
            candidates
            .copy()
            .reset_index(drop=True)
        )

        n = len(result)

        if n == 0:
            return result

        # --------------------------------------------------
        # User lookup
        # --------------------------------------------------

        user_keys = pd.Index(
            result["user_id"].to_numpy(),
            name="user_id",
        )

        user_block = (
            self.store.user_index
            .reindex(user_keys)
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # Streamer lookup
        # --------------------------------------------------

        streamer_keys = pd.Index(
            result["streamer"].to_numpy(),
            name="streamer",
        )

        streamer_block = (
            self.store.streamer_index
            .reindex(streamer_keys)
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # User-streamer pair lookup
        # --------------------------------------------------

        pair_keys = pd.MultiIndex.from_arrays(
            [
                result["user_id"].to_numpy(),
                result["streamer"].to_numpy(),
            ],
            names=[
                "user_id",
                "streamer",
            ],
        )

        pair_block = (
            self.store.pair_index
            .reindex(pair_keys)
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # Combine blocks positionally
        # --------------------------------------------------

        result = pd.concat(
            [
                result,
                user_block,
                streamer_block,
                pair_block,
            ],
            axis=1,
        )

        # Warm-user model, but keep safe defaults.
        for col in USER_ZERO_COLUMNS:
            if col in result.columns:
                result[col] = (
                    result[col].fillna(0)
                )

        # Preserve streamer NaN behavior from Notebook 02.
        # HGB's imputer handles unseen streamers.

        for col in PAIR_ZERO_COLUMNS:
            result[col] = (
                result[col].fillna(0)
            )

        result["seen_before"] = (
            result["pair_interactions"] > 0
        ).astype("int8")

        result["pair_recency_intervals"] = (
            result["pair_recency_intervals"]
            .fillna(
                self.store.unseen_pair_recency
            )
        )

        result["pair_interaction_share"] = (
            result["pair_interactions"]
            / result[
                "user_interactions"
            ].clip(lower=1)
        )

        result["pair_watch_share"] = (
            result["pair_watch_minutes"]
            / result[
                "user_total_watch_minutes"
            ].clip(lower=1)
        )

        missing_features = [
            col
            for col in FEATURE_COLUMNS
            if col not in result.columns
        ]

        if missing_features:
            raise RuntimeError(
                "Feature engine failed to construct: "
                f"{missing_features}"
            )

        return result

    def transform_user(
        self,
        user_id: int,
        streamers: Iterable[str],
    ) -> pd.DataFrame:

        candidates = pd.DataFrame(
            {
                "user_id": user_id,
                "streamer": list(streamers),
            }
        )

        return self.transform(
            candidates
        )

    @staticmethod
    def model_matrix(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        return frame[
            FEATURE_COLUMNS
        ].copy()
