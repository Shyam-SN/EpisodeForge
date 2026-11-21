"""Deterministic dataset splitting with zero data leakage verification."""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Sequence

from episodeforge.canonical.schema import CanonicalEpisode


def compute_deterministic_split(
    episode_ids: Sequence[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    salt: str = "episodeforge_split",
) -> dict[str, list[str]]:
    """Partition episode IDs deterministically into train, val, and test splits."""
    total = train_ratio + val_ratio + test_ratio
    p_train = train_ratio / total
    p_val = val_ratio / total

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for ep_id in sorted(episode_ids):
        # Stable hash float in [0.0, 1.0)
        h = hashlib.sha256(f"{salt}_{ep_id}".encode()).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF

        if u < p_train:
            splits["train"].append(ep_id)
        elif u < (p_train + p_val):
            splits["val"].append(ep_id)
        else:
            splits["test"].append(ep_id)

    verify_split_leakage(splits)
    return splits


def split_episodes_stratified(
    episodes: Sequence[CanonicalEpisode],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict[str, list[str]]:
    """Partition episodes stratified by task ensuring proportional task representation in all splits."""
    by_task: dict[str, list[str]] = collections.defaultdict(list)
    for ep in episodes:
        by_task[ep.task].append(ep.episode_id)

    combined_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for task_name, task_ep_ids in sorted(by_task.items()):
        task_split = compute_deterministic_split(
            task_ep_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            salt=f"split_task_{task_name}",
        )
        for s in ("train", "val", "test"):
            combined_splits[s].extend(task_split[s])

    verify_split_leakage(combined_splits)
    return combined_splits


def verify_split_leakage(splits: dict[str, list[str]]) -> None:
    """Verify that there is zero overlap between train, val, and test sets."""
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("val", []))
    test_set = set(splits.get("test", []))

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val:
        raise ValueError(f"Data leakage detected between train and val: {overlap_train_val}")
    if overlap_train_test:
        raise ValueError(f"Data leakage detected between train and test: {overlap_train_test}")
    if overlap_val_test:
        raise ValueError(f"Data leakage detected between val and test: {overlap_val_test}")
