"""Dead-letter queue (DLQ) quarantine handler for rejected episodes."""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.validate_in.models import InputQAResult


class QuarantineManager:
    """Manages rejected and quarantined episodes in a dead-letter queue."""

    def __init__(self, rejected_dir: Path | str = "data/rejected") -> None:
        self.rejected_dir = Path(rejected_dir).resolve()
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

    def quarantine(
        self,
        episode: CanonicalEpisode,
        qa_result: InputQAResult,
        source_dir: Path | str | None = None,
    ) -> Path:
        """Isolate a failed episode into the dead-letter queue with a structured manifest."""
        quarantine_dir = self.rejected_dir / episode.episode_id
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Write rejection manifest
        manifest = {
            "episode_id": episode.episode_id,
            "source_ref": episode.source_ref,
            "source": episode.source,
            "task": episode.task,
            "quarantined_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "verdict": qa_result.verdict.value,
            "passed": qa_result.passed,
            "reasons": qa_result.reasons,
            "failing_checks": [
                {
                    "check_name": c.check_name,
                    "severity": c.severity.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in qa_result.checks
                if not c.passed
            ],
            "metrics": qa_result.metrics,
            "injected_imperfection": episode.imperfection.model_dump(),
        }

        manifest_path = quarantine_dir / "rejection_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Copy original raw metadata if source_dir is provided
        if source_dir:
            src = Path(source_dir)
            if src.is_dir():
                for json_file in src.glob("*.json"):
                    shutil.copy2(json_file, quarantine_dir / json_file.name)

        return quarantine_dir
