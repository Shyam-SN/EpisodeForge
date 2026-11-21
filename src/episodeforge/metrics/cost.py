"""Cost modeling, throughput metrics, and financial forecasting for robotics pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from episodeforge.config.client_spec import ClientSpec
from episodeforge.config.cost_config import CostRates


@dataclass
class CostForecast:
    """Forecasted resource usage and dollar quote for an episode delivery."""

    episodes_requested: int
    client: str
    dataset_name: str
    format: str

    # Performance & Timing Estimates
    estimated_duration_sec_per_ep: float
    total_robot_hours: float
    pipeline_throughput_eps_per_sec: float
    worker_count: int
    projected_wall_clock_hours: float
    total_compute_core_hours: float

    # Storage & Data Transfer Estimates
    estimated_mb_per_episode: float
    total_dataset_gb: float
    s3_put_requests: int
    s3_get_requests: int

    # Financial Cost Breakdown ($ USD)
    compute_cost_usd: float
    storage_monthly_cost_usd: float
    egress_cost_usd: float
    api_requests_cost_usd: float
    total_cost_usd: float
    cost_per_episode_usd: float
    cost_per_delivered_hour_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_requested": self.episodes_requested,
            "client": self.client,
            "dataset_name": self.dataset_name,
            "format": self.format,
            "total_robot_hours": round(self.total_robot_hours, 2),
            "worker_count": self.worker_count,
            "projected_wall_clock_hours": round(self.projected_wall_clock_hours, 3),
            "total_compute_core_hours": round(self.total_compute_core_hours, 2),
            "total_dataset_gb": round(self.total_dataset_gb, 3),
            "financial_breakdown_usd": {
                "compute": round(self.compute_cost_usd, 4),
                "storage_monthly": round(self.storage_monthly_cost_usd, 4),
                "egress": round(self.egress_cost_usd, 4),
                "api_requests": round(self.api_requests_cost_usd, 4),
                "total": round(self.total_cost_usd, 4),
            },
            "unit_metrics": {
                "cost_per_episode_usd": round(self.cost_per_episode_usd, 4),
                "cost_per_delivered_hour_usd": round(self.cost_per_delivered_hour_usd, 2),
            },
        }


def load_cost_rates(rates_path: Path | str = "configs/cost/rates.yaml") -> CostRates:
    """Load cost rates from YAML file or return defaults."""
    path = Path(rates_path)
    if not path.exists():
        return CostRates()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return CostRates.model_validate(data)


def forecast_pipeline_cost(
    episodes: int,
    spec: ClientSpec,
    rates: CostRates | None = None,
    worker_count: int = 4,
    measured_eps_per_sec: float = 1.5,
    avg_episode_sec: float = 15.0,
) -> CostForecast:
    """Forecast compute time, cloud storage, egress, API requests, and financial cost.

    Based on empirical pipeline telemetry:
    - Average robot demo duration: 15s (750 control steps at 50Hz, 450 frames at 30Hz)
    - Throughput: ~1.5 eps/sec cold (ProcessPoolExecutor)
    - Data footprint:
        - AV1 compressed video (2 cams, 224x224): ~0.8 MB/ep
        - H.264 compressed video: ~1.8 MB/ep
        - JPEG compressed HDF5: ~3.5 MB/ep
        - Raw uncompressed arrays: ~15 MB/ep
    """
    if rates is None:
        rates = load_cost_rates()

    # Determine empirical MB/episode footprint based on format and camera config
    num_cams = len(spec.cameras)
    if spec.format == "lerobot":
        # AV1 video + Parquet metadata
        mb_per_ep = 0.4 * num_cams + 0.15
    elif spec.format == "robomimic_hdf5":
        # HDF5 with JPEG or raw arrays
        opt = getattr(spec, "robomimic_options", None)
        img_fmt = getattr(opt, "image_storage_format", "jpeg") if opt else "jpeg"
        mb_per_ep = (1.5 if img_fmt == "jpeg" else 6.0) * num_cams + 0.2
    elif spec.format == "rlds":
        # TFRecords with JPEG encoded images
        mb_per_ep = 1.2 * num_cams + 0.2
    else:
        mb_per_ep = 1.0 * num_cams + 0.5

    total_mb = episodes * mb_per_ep
    total_gb = total_mb / 1024.0

    # Total robot demonstration time
    total_robot_sec = episodes * avg_episode_sec
    total_robot_hours = total_robot_sec / 3600.0

    # Wall-clock processing time
    total_processing_sec = episodes / max(measured_eps_per_sec, 0.01)
    projected_wall_clock_hours = (total_processing_sec / 3600.0) / max(worker_count / 4.0, 1.0)

    # Core hours consumed (worker_count * wall_clock_hours)
    compute_core_hours = projected_wall_clock_hours * worker_count

    # S3 API requests: ~3 PUTs per episode (videos + shards) + 1 manifest
    put_requests = int(episodes * (num_cams + 1) + 10)
    get_requests = int(episodes * 2 + 10)

    # Dollar cost calculations
    compute_cost = compute_core_hours * rates.compute_core_hour_usd
    storage_cost = total_gb * rates.s3_standard_storage_gb_month_usd
    egress_cost = total_gb * rates.s3_egress_gb_usd
    api_cost = (put_requests / 1000.0) * rates.s3_put_request_per_1k_usd + (
        get_requests / 10000.0
    ) * rates.s3_get_request_per_10k_usd

    total_cost = compute_cost + storage_cost + egress_cost + api_cost
    cost_per_ep = total_cost / episodes if episodes > 0 else 0.0
    cost_per_hr = total_cost / total_robot_hours if total_robot_hours > 0 else 0.0

    return CostForecast(
        episodes_requested=episodes,
        client=spec.client,
        dataset_name=spec.dataset_name,
        format=spec.format,
        estimated_duration_sec_per_ep=avg_episode_sec,
        total_robot_hours=total_robot_hours,
        pipeline_throughput_eps_per_sec=measured_eps_per_sec,
        worker_count=worker_count,
        projected_wall_clock_hours=projected_wall_clock_hours,
        total_compute_core_hours=compute_core_hours,
        estimated_mb_per_episode=mb_per_ep,
        total_dataset_gb=total_gb,
        s3_put_requests=put_requests,
        s3_get_requests=get_requests,
        compute_cost_usd=compute_cost,
        storage_monthly_cost_usd=storage_cost,
        egress_cost_usd=egress_cost,
        api_requests_cost_usd=api_cost,
        total_cost_usd=total_cost,
        cost_per_episode_usd=cost_per_ep,
        cost_per_delivered_hour_usd=cost_per_hr,
    )
