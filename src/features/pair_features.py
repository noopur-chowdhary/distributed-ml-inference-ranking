from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import RECENT_WINDOWS


def build_pair_features(
    history: pd.DataFrame,
    cutoff: int,
) -> pd.DataFrame:
    """Exact Notebook-02 user-streamer affinity formulas."""
    pair = (
        history.groupby(["user_id", "streamer"])
        .agg(
            pair_interactions=("stream_id", "size"),
            pair_watch_minutes=("watch_minutes", "sum"),
            pair_avg_watch_minutes=("watch_minutes", "mean"),
            pair_last_start=("start_time", "max"),
            pair_first_start=("start_time", "min"),
        )
        .reset_index()
    )

    pair["pair_recency_intervals"] = (
        cutoff - pair["pair_last_start"]
    ).clip(lower=0)

    pair["pair_active_span_intervals"] = (
        pair["pair_last_start"] - pair["pair_first_start"]
    )

    pair["pair_repeat_strength"] = np.log1p(
        pair["pair_interactions"]
    )

    for window in RECENT_WINDOWS:
        recent = history[
            history["start_time"] >= cutoff - window
        ]

        recent_counts = (
            recent.groupby(["user_id", "streamer"])
            .size()
            .rename(f"pair_interactions_last_{window}")
            .reset_index()
        )

        pair = pair.merge(
            recent_counts,
            on=["user_id", "streamer"],
            how="left",
        )

        col = f"pair_interactions_last_{window}"
        pair[col] = pair[col].fillna(0)

    return pair.drop(
        columns=["pair_last_start", "pair_first_start"],
        errors="ignore",
    )
