from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.constants import INTERVAL_MINUTES, RAW_COLUMNS


def read_cutoff_from_metadata(metadata_path: str | Path) -> int:
    metadata_path = Path(metadata_path)
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    if "cutoff_time" not in metadata:
        raise KeyError(f"'cutoff_time' missing from {metadata_path}")

    return int(metadata["cutoff_time"])


def load_clean_twitch_history(
    raw_path: str | Path,
    cutoff: int,
) -> pd.DataFrame:
    """
    Load the Twitch raw interactions and reproduce Notebook 02 cleaning.

    Only rows strictly before `cutoff` are returned.
    """
    raw_path = Path(raw_path)

    df = pd.read_csv(
        raw_path,
        header=None,
        names=RAW_COLUMNS,
        usecols=RAW_COLUMNS,
    )

    df["duration_intervals"] = df["stop_time"] - df["start_time"]
    df["watch_minutes"] = df["duration_intervals"] * INTERVAL_MINUTES

    df = df[
        df["user_id"].notna()
        & df["streamer"].notna()
        & df["start_time"].notna()
        & df["stop_time"].notna()
        & (df["duration_intervals"] > 0)
        & (df["start_time"] < cutoff)
    ].copy()

    df = df.sort_values(
        ["start_time", "user_id", "streamer"]
    ).reset_index(drop=True)

    return df
