"""Pipeline stages executing Ingest -> InputQA -> Transform on individual episodes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from episodeforge.canonical.schema import CanonicalEpisode, QAVerdict
from episodeforge.config.client_spec import ClientSpec
from episodeforge.ingest import ingest_episode
from episodeforge.orchestrate.cache import ContentAddressableCache, compute_stage_cache_key
from episodeforge.transform.pipeline import TransformedEpisode, apply_client_transforms
from episodeforge.validate_in.engine import InputQAEngine
from episodeforge.validate_in.quarantine import QuarantineManager


def process_single_episode(
    ep_dir: Path | str,
    spec: ClientSpec,
    cache: ContentAddressableCache | None = None,
    quarantine_mgr: QuarantineManager | None = None,
) -> tuple[TransformedEpisode | None, dict[str, Any]]:
    """Execute Ingest -> InputQA -> Transform on an individual episode directory."""
    p = Path(ep_dir).resolve()
    t_start = time.time()
    stage_metrics: dict[str, Any] = {
        "episode_dir": p.name,
        "stages": {},
        "cached": False,
        "error": None,
    }

    # 1. Ingest Stage
    t0 = time.time()
    try:
        episode: CanonicalEpisode = ingest_episode(p)
        stage_metrics["stages"]["ingest"] = {
            "elapsed_sec": time.time() - t0,
            "episode_id": episode.episode_id,
        }
    except Exception as e:
        stage_metrics["error"] = f"Ingest failed: {e}"
        return None, stage_metrics

    # 2. Input QA Stage
    t0 = time.time()
    engine = InputQAEngine(reject_on_failure_flag=spec.selection.require_success)
    qa_res = engine.validate(episode)
    stage_metrics["stages"]["input_qa"] = {
        "elapsed_sec": time.time() - t0,
        "verdict": qa_res.verdict.value,
        "passed": qa_res.passed,
    }

    if not qa_res.passed or qa_res.verdict == QAVerdict.FAIL:
        if quarantine_mgr:
            quarantine_mgr.quarantine(episode, qa_res, source_dir=p)
        stage_metrics["rejected"] = True
        stage_metrics["rejection_reasons"] = qa_res.reasons
        return None, stage_metrics

    # 3. Transform Stage (with caching)
    t0 = time.time()
    cache_key = ""
    if cache and cache.enabled:
        spec_summary = {
            "client": spec.client,
            "format": spec.format,
            "target_fps": spec.timing.target_fps,
            "action": spec.action.model_dump(),
        }
        cache_key = compute_stage_cache_key(
            episode.provenance.input_content_hash,
            "transform",
            spec_summary,
        )
        cached_trans = cache.get("transform", cache_key)
        if cached_trans is not None:
            stage_metrics["stages"]["transform"] = {
                "elapsed_sec": time.time() - t0,
                "cached": True,
            }
            stage_metrics["cached"] = True
            stage_metrics["total_elapsed_sec"] = time.time() - t_start
            return cached_trans, stage_metrics

    transformed = apply_client_transforms(episode, spec)
    stage_metrics["stages"]["transform"] = {
        "elapsed_sec": time.time() - t0,
        "cached": False,
        "filtered": transformed is None,
    }

    if transformed and cache and cache.enabled and cache_key:
        cache.put("transform", cache_key, transformed)

    stage_metrics["total_elapsed_sec"] = time.time() - t_start
    return transformed, stage_metrics
