from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class HGBRanker:
    """Thin serving wrapper around the Notebook-03 HGB artifact."""

    def __init__(
        self,
        model,
        imputer,
        feature_columns: list[str],
    ):
        self.model = model
        self.imputer = imputer
        self.feature_columns = list(feature_columns)

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
    ) -> "HGBRanker":
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"HGB artifact not found: {artifact_path}"
            )

        bundle = joblib.load(artifact_path)

        if not isinstance(bundle, dict):
            raise TypeError(
                "Expected HGB artifact dict containing model, imputer, "
                "and feature_columns."
            )

        return cls(
            model=bundle["model"],
            imputer=bundle["imputer"],
            feature_columns=list(bundle["feature_columns"]),
        )

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [
            col for col in self.feature_columns
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Ranker input missing model features: {missing}"
            )

        X = self.imputer.transform(
            frame[self.feature_columns]
        )

        return self.model.predict_proba(X)[:, 1]

    def rank(
        self,
        frame: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        result = frame.copy()
        result["rank_score"] = self.score(result)

        result = (
            result.sort_values(
                ["rank_score", "retrieval_score"],
                ascending=[False, False],
                na_position="last",
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        result["rank_position"] = (
            np.arange(len(result), dtype=int) + 1
        )

        return result
