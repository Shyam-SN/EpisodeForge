"""Pipeline Runner managing end-to-end execution, parallel workers, and delivery validation."""

from __future__ import annotations

import concurrent.futures
import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from episodeforge.config.client_spec import ClientSpec
from episodeforge.config.pipeline_config import PipelineConfig
from episodeforge.lineage.catalog import DuckDBLineageCatalog
from episodeforge.orchestrate.cache import ContentAddressableCache
from episodeforge.orchestrate.stages import process_single_episode
from episodeforge.transform.pipeline import TransformedEpisode
from episodeforge.validate_in.quarantine import QuarantineManager
from episodeforge.validate_out.engine import validate_output_dataset
from episodeforge.writers import write_dataset_for_spec


@dataclass
class PipelineRunReport:
    run_id: str
    client: str
    dataset_name: str
    format: str
    total_episodes_discovered: int
    episodes_processed: int
    episodes_quarantined: int
    episodes_transformed: int
    total_frames: int
    elapsed_time_sec: float
    throughput_eps_per_sec: float
    throughput_eps_per_hour: float
    cache_hits: int
    cache_misses: int
    output_dir: str
    validation_status: str
    validation_details: dict[str, Any] = field(default_factory=dict)
    stage_breakdown: dict[str, float] = field(default_factory=dict)


def _worker_wrapper(
    ep_dir_str: str,
    spec_dict: dict[str, Any],
    cache_enabled: bool,
    cache_dir_str: str,
    rejected_dir_str: str,
) -> tuple[TransformedEpisode | None, dict[str, Any]]:
    """Worker function for ProcessPoolExecutor execution."""
    spec = ClientSpec.model_validate(spec_dict)
    cache = ContentAddressableCache(cache_dir=cache_dir_str, enabled=cache_enabled)
    quarantine_mgr = QuarantineManager(rejected_dir=rejected_dir_str)
    return process_single_episode(
        ep_dir=ep_dir_str,
        spec=spec,
        cache=cache,
        quarantine_mgr=quarantine_mgr,
    )


class PipelineRunner:
    """Orchestrates end-to-end execution from raw recordings to validated client release."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        catalog: DuckDBLineageCatalog | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.cache = ContentAddressableCache(
            cache_dir=self.config.cache.cache_dir,
            enabled=self.config.cache.enabled,
        )
        self.quarantine_mgr = QuarantineManager(rejected_dir=self.config.dead_letter_dir)
        self.catalog = catalog or DuckDBLineageCatalog()

    def run(
        self,
        raw_dataset_dir: Path | str,
        spec: ClientSpec,
        limit: int | None = None,
        workers: int | None = None,
    ) -> PipelineRunReport:
        t_global_start = time.time()
        run_id = f"run_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}"

        raw_path = Path(raw_dataset_dir).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Input dataset path does not exist: {raw_path}")

        # Find episode directories
        ep_dirs = sorted([d for d in raw_path.glob("ep_*") if d.is_dir()])
        if limit:
            ep_dirs = ep_dirs[:limit]

        total_discovered = len(ep_dirs)
        num_workers = workers or self.config.execution.workers

        transformed_episodes: list[TransformedEpisode] = []
        quarantined_count = 0
        cache_hits = 0
        cache_misses = 0

        spec_dict = spec.model_dump()
        cache_dir_str = str(self.cache.cache_dir)
        rejected_dir_str = str(self.quarantine_mgr.rejected_dir)
        cache_enabled = self.cache.enabled

        # Process episodes in parallel
        if num_workers > 1 and len(ep_dirs) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _worker_wrapper,
                        str(ep_p),
                        spec_dict,
                        cache_enabled,
                        cache_dir_str,
                        rejected_dir_str,
                    ): ep_p
                    for ep_p in ep_dirs
                }
                for fut in concurrent.futures.as_completed(futures):
                    t_ep, metrics = fut.result()
                    if metrics.get("cached"):
                        cache_hits += 1
                    else:
                        cache_misses += 1

                    if metrics.get("rejected"):
                        quarantined_count += 1
                    elif t_ep is not None:
                        transformed_episodes.append(t_ep)
        else:
            # Sequential execution
            for ep_p in ep_dirs:
                t_ep, metrics = process_single_episode(
                    ep_p,
                    spec,
                    cache=self.cache,
                    quarantine_mgr=self.quarantine_mgr,
                )
                if metrics.get("cached"):
                    cache_hits += 1
                else:
                    cache_misses += 1

                if metrics.get("rejected"):
                    quarantined_count += 1
                elif t_ep is not None:
                    transformed_episodes.append(t_ep)

        # Sort transformed episodes deterministically
        transformed_episodes.sort(key=lambda e: e.episode_id)

        # Write dataset release
        release_out_dir = (
            Path(self.config.releases_dir)
            / spec.client
            / f"{spec.dataset_name}_{spec.spec_version}"
        )
        release_out_dir.mkdir(parents=True, exist_ok=True)

        if transformed_episodes:
            write_dataset_for_spec(transformed_episodes, release_out_dir, spec)

        # Run output validation
        val_res = validate_output_dataset(release_out_dir, spec)

        elapsed = time.time() - t_global_start
        total_frames = sum(ep.length for ep in transformed_episodes)
        rate = total_discovered / elapsed if elapsed > 0 else 0.0

        report = PipelineRunReport(
            run_id=run_id,
            client=spec.client,
            dataset_name=spec.dataset_name,
            format=spec.format,
            total_episodes_discovered=total_discovered,
            episodes_processed=total_discovered,
            episodes_quarantined=quarantined_count,
            episodes_transformed=len(transformed_episodes),
            total_frames=total_frames,
            elapsed_time_sec=elapsed,
            throughput_eps_per_sec=rate,
            throughput_eps_per_hour=rate * 3600.0,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            output_dir=str(release_out_dir),
            validation_status=val_res.get("status", "UNKNOWN"),
            validation_details=val_res,
        )

        if self.catalog:
            try:
                for ep in transformed_episodes:
                    self.catalog.log_episode(ep)
                self.catalog.log_run(report)
            except Exception:
                pass

        return report
