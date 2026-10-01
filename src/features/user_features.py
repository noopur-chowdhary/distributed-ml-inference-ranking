from __future__ import annotations

import pandas as pd

from src.constants import RECENT_WINDOWS


def build_user_features(
    history: pd.DataFrame,
    cutoff: int,
) -> pd.DataFrame:
    """Exact Notebook-02 user feature formulas."""
    base = (
        history.groupby("user_id")
        .agg(
            user_interactions=("streamer", "size"),
            user_unique_streamers=("streamer", "nunique"),
            user_total_watch_minutes=("watch_minutes", "sum"),
            user_avg_watch_minutes=("watch_minutes", "mean"),
            user_median_watch_minutes=("watch_minutes", "median"),
            user_first_start=("start_time", "min"),
            user_last_start=("start_time", "max"),
        )
        .reset_index()
    )

    base["user_active_span_intervals"] = (
        base["user_last_start"] - base["user_first_start"]
    )

    base["user_last_activity_recency"] = (
        cutoff - base["user_last_start"]
    ).clip(lower=0)

    base["user_repeat_rate"] = (
        1
        - (
            base["user_unique_streamers"]
            / base["user_interactions"].clip(lower=1)
        )
    )

    for window in RECENT_WINDOWS:
        recent = history[
            history["start_time"] >= cutoff - window
        ]

        recent_counts = (
            recent.groupby("user_id")
            .size()
            .rename(f"user_interactions_last_{window}_intervals")
            .reset_index()
        )

        base = base.merge(
            recent_counts,
            on="user_id",
            how="left",
        )

        col = f"user_interactions_last_{window}_intervals"
        base[col] = base[col].fillna(0)

    return base.drop(
        columns=["user_first_start", "user_last_start"],
        errors="ignore",
    )
