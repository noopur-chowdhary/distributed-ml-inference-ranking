from __future__ import annotations

import numpy as np
import pandas as pd


def build_streamer_features(
    history: pd.DataFrame,
    cutoff: int,
) -> pd.DataFrame:
    """Exact Notebook-02 streamer popularity/trend formulas."""
    base = (
        history.groupby("streamer")
        .agg(
            streamer_interactions=("user_id", "size"),
            streamer_unique_viewers=("user_id", "nunique"),
            streamer_total_watch_minutes=("watch_minutes", "sum"),
            streamer_avg_watch_minutes=("watch_minutes", "mean"),
        )
        .reset_index()
    )

    total_interactions = max(len(history), 1)

    base["streamer_popularity_share"] = (
        base["streamer_interactions"] / total_interactions
    )

    base["streamer_log_interactions"] = np.log1p(
        base["streamer_interactions"]
    )

    base["streamer_log_viewers"] = np.log1p(
        base["streamer_unique_viewers"]
    )

    pair_counts = (
        history.groupby(["streamer", "user_id"])
        .size()
        .reset_index(name="pair_count")
    )

    repeat_viewer_rate = (
        pair_counts.assign(
            repeated=(pair_counts["pair_count"] > 1).astype(int)
        )
        .groupby("streamer")["repeated"]
        .mean()
        .rename("streamer_repeat_viewer_rate")
        .reset_index()
    )

    base = base.merge(
        repeat_viewer_rate,
        on="streamer",
        how="left",
    )

    windows = {
        "streamer_interactions_last_6": history[
            history["start_time"] >= cutoff - 6
        ],
        "streamer_interactions_last_18": history[
            history["start_time"] >= cutoff - 18
        ],
        "streamer_interactions_last_42": history[
            history["start_time"] >= cutoff - 42
        ],
        "streamer_interactions_prev_42": history[
            (history["start_time"] >= cutoff - 84)
            & (history["start_time"] < cutoff - 42)
        ],
    }

    for name, frame in windows.items():
        counts = (
            frame.groupby("streamer")
            .size()
            .rename(name)
            .reset_index()
        )

        base = base.merge(
            counts,
            on="streamer",
            how="left",
        )

    count_cols = list(windows)
    base[count_cols] = base[count_cols].fillna(0)

    base["streamer_trend_ratio"] = (
        (base["streamer_interactions_last_42"] + 1)
        / (base["streamer_interactions_prev_42"] + 1)
    )

    return base
