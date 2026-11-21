"""DuckDB-backed lineage, provenance, and operational metrics catalog."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.transform.pipeline import TransformedEpisode

if TYPE_CHECKING:
    from episodeforge.orchestrate.pipeline_runner import PipelineRunReport


class DuckDBLineageCatalog:
    """Manages persistent audit trails, lineage graphs, and operational metrics in DuckDB."""

    def __init__(self, db_path: Path | str = "data/lineage_catalog.duckdb") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id VARCHAR PRIMARY KEY,
                source_ref VARCHAR,
                source VARCHAR,
                task VARCHAR,
                success BOOLEAN,
                duration_sec DOUBLE,
                control_steps INTEGER,
                content_hash VARCHAR,
                seed INTEGER,
                upstream_verdict VARCHAR,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                timestamp_utc TIMESTAMP,
                client VARCHAR,
                dataset_name VARCHAR,
                format VARCHAR,
                episodes_discovered INTEGER,
                episodes_quarantined INTEGER,
                episodes_released INTEGER,
                total_frames INTEGER,
                elapsed_sec DOUBLE,
                throughput_eps_hr DOUBLE,
                cache_hits INTEGER,
                cache_misses INTEGER,
                output_dir VARCHAR,
                validation_status VARCHAR
            );

            CREATE TABLE IF NOT EXISTS deliveries (
                receipt_id VARCHAR PRIMARY KEY,
                manifest_id VARCHAR,
                manifest_sha256 VARCHAR,
                client VARCHAR,
                dataset_name VARCHAR,
                target_s3_uri VARCHAR,
                uploaded_files_count INTEGER,
                total_bytes BIGINT,
                status VARCHAR,
                completed_at_utc TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recalls (
                recall_id VARCHAR PRIMARY KEY,
                original_release VARCHAR,
                patch_release VARCHAR,
                dataset_name VARCHAR,
                client VARCHAR,
                recalled_count INTEGER,
                recalled_episode_ids VARCHAR,
                reason VARCHAR,
                timestamp_utc TIMESTAMP
            );
        """)

    def log_episode(self, episode: CanonicalEpisode | TransformedEpisode) -> None:
        """Upsert episode metadata into lineage catalog."""
        timestamps: list[int] = []
        if hasattr(episode, "timestamps_ns"):
            timestamps = episode.timestamps_ns
        elif hasattr(episode, "state") and hasattr(episode.state, "timestamps_ns"):
            timestamps = episode.state.timestamps_ns

        duration_sec = 0.0
        if timestamps:
            duration_sec = (timestamps[-1] - timestamps[0]) / 1e9

        source = getattr(episode, "source", "sim")
        seed = getattr(episode, "seed", 0)
        verdict = getattr(episode, "upstream_qa_verdict", None)
        verdict_str = verdict.value if (verdict is not None and hasattr(verdict, "value")) else "PASS"

        content_hash = ""
        if hasattr(episode, "provenance") and hasattr(episode.provenance, "input_content_hash"):
            content_hash = episode.provenance.input_content_hash

        self.conn.execute(
            """
            INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                episode.episode_id,
                episode.source_ref,
                source,
                episode.task,
                episode.success,
                duration_sec,
                len(timestamps),
                content_hash,
                seed,
                verdict_str,
                datetime.datetime.now(datetime.UTC),
            ],
        )

    def log_run(self, report: PipelineRunReport) -> None:
        """Record pipeline execution run report."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                report.run_id,
                datetime.datetime.now(datetime.UTC),
                report.client,
                report.dataset_name,
                report.format,
                report.total_episodes_discovered,
                report.episodes_quarantined,
                report.episodes_transformed,
                report.total_frames,
                report.elapsed_time_sec,
                report.throughput_eps_per_hour,
                report.cache_hits,
                report.cache_misses,
                report.output_dir,
                report.validation_status,
            ],
        )

    def log_delivery(self, receipt: dict[str, Any]) -> None:
        """Record cloud transfer receipt."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO deliveries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                receipt.get("receipt_id"),
                receipt.get("manifest_id"),
                receipt.get("manifest_sha256"),
                receipt.get("client"),
                receipt.get("dataset_name"),
                receipt.get("target_s3_uri"),
                receipt.get("uploaded_files_count", 0),
                receipt.get("total_transferred_bytes", 0),
                receipt.get("verification_status", "UNKNOWN"),
                datetime.datetime.now(datetime.UTC),
            ],
        )

    def log_recall(self, notice: dict[str, Any]) -> None:
        """Record data recall action."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO recalls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                notice.get("recall_id"),
                notice.get("original_release"),
                notice.get("patch_release"),
                notice.get("dataset_name"),
                notice.get("client"),
                notice.get("recalled_episodes_count", 0),
                ",".join(notice.get("recalled_episode_ids", [])),
                notice.get("reason"),
                datetime.datetime.now(datetime.UTC),
            ],
        )

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute arbitrary SQL query and return rows as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(sql)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def close(self) -> None:
        self.conn.close()
