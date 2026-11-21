"""Ingest package providing adapters for MuJoCo sim, RigSim, and Phone captures."""

from __future__ import annotations

from pathlib import Path

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.ingest.generated import ingest_generated_episode
from episodeforge.ingest.phone import ingest_phone_session
from episodeforge.ingest.rigsim import ingest_rigsim_package


def ingest_episode(path: Path | str) -> CanonicalEpisode:
    """Auto-detect format and ingest directory into CanonicalEpisode."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Episode path must be a directory: {p}")

    # Check for phone session
    if (p / "session_meta.json").exists():
        return ingest_phone_session(p)

    # Check for RigSim package (has robot/ or qa_verdict.json)
    if (p / "robot").is_dir() or (p / "qa_verdict.json").exists():
        return ingest_rigsim_package(p)

    # Check for generated MuJoCo demonstration
    if (p / "episode_meta.json").exists() and (p / "trajectory.parquet").exists():
        return ingest_generated_episode(p)

    raise ValueError(f"Unrecognized episode directory format at {p}")


__all__ = [
    "ingest_episode",
    "ingest_generated_episode",
    "ingest_phone_session",
    "ingest_rigsim_package",
]
