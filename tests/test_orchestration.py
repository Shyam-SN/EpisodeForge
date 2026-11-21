"""Unit and integration tests for Pipeline Orchestration, Stage Caching, and End-to-End Execution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from episodeforge.config.client_spec import ClientSpec
from episodeforge.config.pipeline_config import CacheConfig, PipelineConfig
from episodeforge.orchestrate.cache import (
    ContentAddressableCache,
    compute_stage_cache_key,
)
from episodeforge.orchestrate.pipeline_runner import PipelineRunner


def test_cache_key_determinism():
    key1 = compute_stage_cache_key("hash_123", "transform", {"fps": 30})
    key2 = compute_stage_cache_key("hash_123", "transform", {"fps": 30})
    key3 = compute_stage_cache_key("hash_123", "transform", {"fps": 15})

    assert key1 == key2
    assert key1 != key3


def test_content_addressable_cache_put_get(tmp_path: Path):
    cache = ContentAddressableCache(cache_dir=tmp_path / "cache", enabled=True)
    key = "test_key_001"
    data = {"state": [1, 2, 3], "value": 42.0}

    assert cache.get("transform", key) is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.put("transform", key, data)
    retrieved = cache.get("transform", key)
    assert retrieved == data
    assert cache.hits == 1


def test_pipeline_runner_end_to_end(tmp_path: Path):
    raw_dir = Path("data/raw/m1_episodes")
    if not raw_dir.exists():
        pytest.skip("data/raw/m1_episodes does not exist")

    with open("configs/clients/client_c_robomimic.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec = ClientSpec.model_validate(spec_data)

    p_config = PipelineConfig(
        cache=CacheConfig(enabled=True, cache_dir=str(tmp_path / "cache")),
        dead_letter_dir=str(tmp_path / "rejected"),
        releases_dir=str(tmp_path / "releases"),
    )

    runner = PipelineRunner(config=p_config)

    # First run on 5 episodes
    report = runner.run(raw_dataset_dir=raw_dir, spec=spec, limit=5, workers=2)
    assert report.total_episodes_discovered == 5
    assert report.episodes_processed == 5
    assert report.validation_status == "PASS"
    assert report.cache_misses > 0

    # Second run should hit cache
    report2 = runner.run(raw_dataset_dir=raw_dir, spec=spec, limit=5, workers=2)
    assert report2.cache_hits > 0
