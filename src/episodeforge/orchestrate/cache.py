"""Content-addressable stage cache for pipeline stages."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def compute_stage_cache_key(
    input_hash: str,
    stage_name: str,
    spec_or_params: dict[str, Any] | str,
    version: str = "0.1.0",
) -> str:
    """Compute deterministic SHA256 cache key based on inputs, stage, params, and version."""
    hasher = hashlib.sha256()
    hasher.update(input_hash.encode("utf-8"))
    hasher.update(stage_name.encode("utf-8"))
    if isinstance(spec_or_params, dict):
        param_str = json.dumps(spec_or_params, sort_keys=True)
    else:
        param_str = str(spec_or_params)
    hasher.update(param_str.encode("utf-8"))
    hasher.update(version.encode("utf-8"))
    return hasher.hexdigest()


class ContentAddressableCache:
    """Stores and retrieves stage outputs using content hashes."""

    def __init__(self, cache_dir: Path | str = "data/cache", enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, stage_name: str, key: str) -> Any | None:
        """Retrieve cached output if present."""
        if not self.enabled:
            return None

        entry_path = self.cache_dir / stage_name / f"{key}.pkl"
        if entry_path.exists():
            try:
                with open(entry_path, "rb") as f:
                    data = pickle.load(f)
                self.hits += 1
                return data
            except Exception:
                return None

        self.misses += 1
        return None

    def put(self, stage_name: str, key: str, data: Any) -> None:
        """Store output in cache."""
        if not self.enabled:
            return

        stage_dir = self.cache_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        entry_path = stage_dir / f"{key}.pkl"
        try:
            with open(entry_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # Non-fatal if cache serialization fails
