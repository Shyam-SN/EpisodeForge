"""Unit tests for demonstration generator and IK solver."""

from __future__ import annotations

from pathlib import Path

from episodeforge.config.generation_config import GenerationConfig
from episodeforge.generate.generator import generate_single_episode
from episodeforge.generate.scene import sample_randomization


def test_scene_randomization():
    rand = sample_randomization(seed=42, task="pick_place")
    assert len(rand.table_color) == 4
    assert len(rand.obj1_pos) == 3
    assert len(rand.target_pos) == 3


def test_single_episode_generation(tmp_path: Path):
    cfg = GenerationConfig(
        robot_model="franka_emika_panda",
        tasks=["pick_place"],
        total_episodes=1,
    )
    res = generate_single_episode(
        episode_idx=0,
        task_name="pick_place",
        output_dir=tmp_path,
        config=cfg,
        base_seed=10,
        render_cameras=False,  # Headless test without rendering
    )
    assert res["episode_id"] == "ep_000000"
    assert res["task"] == "pick_place"
    assert (tmp_path / "ep_000000" / "episode_meta.json").exists()
    assert (tmp_path / "ep_000000" / "trajectory.parquet").exists()
    assert (tmp_path / "ep_000000" / "ground_truth_qa.json").exists()
