from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_twitch: Path
    processed_twitch: Path
    artifacts_twitch: Path
    results: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            raw_twitch=root / "data/raw/twitch/100k_a.csv",
            processed_twitch=root / "data/processed/twitch",
            artifacts_twitch=root / "artifacts/twitch",
            results=root / "results",
        )


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by locating data/raw/twitch/100k_a.csv."""
    start_path = Path(start).expanduser().resolve() if start else Path.cwd().resolve()

    candidates = [
        Path.home() / "Desktop/resume_projects/Distributed ML Inference & Ranking",
        start_path,
        *start_path.parents,
    ]

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if (candidate / "data/raw/twitch/100k_a.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root containing data/raw/twitch/100k_a.csv. "
        "Pass project_root explicitly."
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path
