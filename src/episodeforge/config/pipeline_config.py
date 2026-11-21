"""Pipeline execution and orchestration configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    enabled: bool = True
    cache_dir: str = "data/cache"


class ExecutionConfig(BaseModel):
    backend: Literal["local", "ray"] = "local"
    workers: int = 4
    batch_size: int = 16
    max_in_flight: int = 32
    max_retries: int = 3
    retry_backoff_sec: float = 0.5


class PipelineConfig(BaseModel):
    pipeline_name: str = "episodeforge_standard"
    version: str = "0.1.0"
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    dead_letter_dir: str = "data/rejected"
    staging_dir: str = "data/staging"
    releases_dir: str = "data/releases"
    catalog_db_path: str = "data/lineage_catalog.duckdb"
