from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src.features.pair_features import build_pair_features
from src.features.streamer_features import build_streamer_features
from src.features.user_features import build_user_features


@dataclass
class FeatureStore:
    """
    Precomputed historical features used at request time.

    Expensive groupbys happen once.

    Indexed lookup tables are also built once so request-time inference
    does not perform large DataFrame merges.
    """

    cutoff: int
    user_features: pd.DataFrame
    streamer_features: pd.DataFrame
    pair_features: pd.DataFrame
    unseen_pair_recency: float

    user_index: pd.DataFrame = field(init=False, repr=False)
    streamer_index: pd.DataFrame = field(init=False, repr=False)
    pair_index: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._build_indexes()

    def _build_indexes(self):
        self.user_index = (
            self.user_features
            .set_index("user_id")
            .sort_index()
        )

        self.streamer_index = (
            self.streamer_features
            .set_index("streamer")
            .sort_index()
        )

        self.pair_index = (
            self.pair_features
            .set_index(["user_id", "streamer"])
            .sort_index()
        )

    @classmethod
    def from_history(
        cls,
        history: pd.DataFrame,
        cutoff: int,
        unseen_pair_recency: float | None = None,
    ) -> "FeatureStore":

        users = build_user_features(
            history,
            cutoff,
        )

        streamers = build_streamer_features(
            history,
            cutoff,
        )

        pairs = build_pair_features(
            history,
            cutoff,
        )

        if unseen_pair_recency is None:
            max_recency = pairs[
                "pair_recency_intervals"
            ].max()

            unseen_pair_recency = (
                0.0
                if pd.isna(max_recency)
                else float(max_recency) + 1.0
            )

        return cls(
            cutoff=int(cutoff),
            user_features=users,
            streamer_features=streamers,
            pair_features=pairs,
            unseen_pair_recency=float(
                unseen_pair_recency
            ),
        )

    def save(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save raw feature tables.
        # Indexes are rebuilt automatically on load.
        joblib.dump(
            {
                "cutoff": self.cutoff,
                "user_features": self.user_features,
                "streamer_features": self.streamer_features,
                "pair_features": self.pair_features,
                "unseen_pair_recency": self.unseen_pair_recency,
            },
            path,
            compress=3,
        )

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "FeatureStore":

        obj = joblib.load(path)

        return cls(
            cutoff=int(obj["cutoff"]),
            user_features=obj["user_features"],
            streamer_features=obj["streamer_features"],
            pair_features=obj["pair_features"],
            unseen_pair_recency=float(
                obj["unseen_pair_recency"]
            ),
        )


def infer_unseen_pair_recency_from_reference(
    prepared_ranking_path: str | Path,
) -> float | None:

    path = Path(prepared_ranking_path)

    if not path.exists():
        return None

    frame = pd.read_csv(
        path,
        usecols=[
            "seen_before",
            "pair_recency_intervals",
        ],
    )

    unseen = frame.loc[
        frame["seen_before"].eq(0),
        "pair_recency_intervals",
    ].dropna()

    if unseen.empty:
        return None

    return float(
        unseen.mode().iloc[0]
    )
