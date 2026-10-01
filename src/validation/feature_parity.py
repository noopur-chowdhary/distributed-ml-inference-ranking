from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectPaths, discover_project_root
from src.constants import FEATURE_COLUMNS
from src.inference.engine import RankingInferenceEngine


def validate_feature_parity(
    engine: RankingInferenceEngine,
    reference_path: str | Path,
    sample_size: int = 500,
    random_state: int = 42,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Compare serving-time feature construction against Notebook-02 rows.

    Returns a per-feature report and raises AssertionError if parity fails.
    """
    reference_path = Path(reference_path)

    reference = pd.read_csv(reference_path)

    sample_n = min(sample_size, len(reference))
    expected = reference.sample(
        n=sample_n,
        random_state=random_state,
    ).reset_index(drop=True)

    candidates = expected[["user_id", "streamer"]].copy()

    generated = engine.feature_engine.transform(candidates)

    rows = []
    failures = []

    for col in FEATURE_COLUMNS:
        exp = expected[col].to_numpy(dtype=float)
        got = generated[col].to_numpy(dtype=float)

        close = np.isclose(
            exp,
            got,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

        max_abs_error = float(
            np.nanmax(np.abs(exp - got))
        ) if len(exp) else 0.0

        rows.append(
            {
                "feature": col,
                "rows": len(exp),
                "matches": int(close.sum()),
                "match_rate": float(close.mean()),
                "max_abs_error": max_abs_error,
            }
        )

        if not close.all():
            failures.append(col)

    report = pd.DataFrame(rows)

    if failures:
        bad = report[
            report["feature"].isin(failures)
        ]
        raise AssertionError(
            "Feature parity failed for: "
            f"{failures}\n\n{bad.to_string(index=False)}"
        )

    return report


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate 36 serving features against Notebook 02."
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=500)

    args = parser.parse_args()

    root = (
        Path(args.project_root).expanduser().resolve()
        if args.project_root
        else discover_project_root()
    )

    paths = ProjectPaths.from_root(root)

    reference_path = (
        paths.processed_twitch
        / "twitch_test_ranking.csv.gz"
    )

    engine = RankingInferenceEngine.from_project_root(root)

    report = validate_feature_parity(
        engine=engine,
        reference_path=reference_path,
        sample_size=args.sample_size,
    )

    print(report.to_string(index=False))
    print("\nPASS: all 36 features match the sampled Notebook-02 rows.")


if __name__ == "__main__":
    _main()
