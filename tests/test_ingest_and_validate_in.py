"""Unit and integration tests for Ingest Adapters, Input QA Engine, and DLQ Quarantine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from episodeforge.canonical.schema import (
    ActionStream,
    CameraConvention,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraStream,
    CanonicalEpisode,
    ConventionsMetadata,
    ImperfectionLabel,
    ProvenanceMetadata,
    QAVerdict,
    StateStream,
)
from episodeforge.ingest.phone import ingest_phone_session
from episodeforge.validate_in.checks.kinematics import (
    check_kinematic_limits,
    check_numeric_integrity,
)
from episodeforge.validate_in.checks.timestamps import (
    check_cross_stream_alignment,
    check_monotonic_timestamps,
    check_sampling_jitter,
)
from episodeforge.validate_in.engine import InputQAEngine
from episodeforge.validate_in.quarantine import QuarantineManager


def _make_dummy_episode(
    nan_in_state: bool = False,
    joint_limit_violation: bool = False,
    non_monotonic_ts: bool = False,
    success: bool = True,
    frames_dir: Path | None = None,
) -> CanonicalEpisode:
    """Helper to build test CanonicalEpisode fixtures."""
    from PIL import Image

    timestamps = [int(i * 66.666e6) for i in range(20)]
    if non_monotonic_ts:
        timestamps[10] = timestamps[9] - 1000

    qpos = [[0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.0] for _ in range(20)]
    if nan_in_state:
        qpos[5][0] = float("nan")
    if joint_limit_violation:
        qpos[5][0] = 5.0  # Franka Panda limit is 2.8973

    qvel = [[0.0] * 7 for _ in range(20)]
    ee_pos = [[0.5, 0.0, 0.5] for _ in range(20)]
    ee_quat = [[1.0, 0.0, 0.0, 0.0] for _ in range(20)]
    grippers = [0.04] * 20

    state = StateStream(
        timestamps_ns=timestamps,
        joint_names=[f"panda_joint{i + 1}" for i in range(7)],
        joint_positions=qpos,
        joint_velocities=qvel,
        ee_positions=ee_pos,
        ee_quaternions=ee_quat,
        gripper_widths=grippers,
    )
    action = ActionStream(
        timestamps_ns=timestamps,
        nominal_rate_hz=15.0,
        joint_position_targets=qpos,
        ee_delta_positions=[[0.0, 0.0, 0.0] for _ in range(20)],
        gripper_targets=grippers,
    )

    cameras = {}
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            arr = np.full((48, 64, 3), fill_value=(50 + i * 5, 100, 150), dtype=np.uint8)
            Image.fromarray(arr).save(frames_dir / f"frame_{i:04d}.png")

        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48)
        extrinsics = CameraExtrinsics(
            translation=[0.5, 0.0, 0.5],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            reference_frame="robot_base",
        )
        cam = CameraStream(
            name="front",
            width=64,
            height=48,
            fps_nominal=30.0,
            frame_count=20,
            convention=CameraConvention.OPENCV,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps_ns=timestamps,
            frames_dir=str(frames_dir),
        )
        cameras["front"] = cam

    return CanonicalEpisode(
        episode_id="ep_test12345678",
        source="test_fixture",
        source_ref="ep_000000",
        task="pick_place",
        instruction="pick up the object",
        success=success,
        seed=42,
        conventions=ConventionsMetadata(),
        cameras=cameras,
        state=state,
        action=action,
        upstream_qa_verdict=QAVerdict.PASS,
        imperfection=ImperfectionLabel(),
        provenance=ProvenanceMetadata(
            input_content_hash="test_hash_12345678",
            created_at_utc="2026-11-14T00:00:00Z",
        ),
    )


def test_numeric_integrity_check():
    clean_ep = _make_dummy_episode()
    res_clean = check_numeric_integrity(clean_ep)
    assert res_clean.passed is True

    bad_ep = _make_dummy_episode(nan_in_state=True)
    res_bad = check_numeric_integrity(bad_ep)
    assert res_bad.passed is False
    assert "joint_positions" in res_bad.details


def test_kinematic_limits_check():
    clean_ep = _make_dummy_episode()
    res_clean = check_kinematic_limits(clean_ep)
    assert res_clean.passed is True

    bad_ep = _make_dummy_episode(joint_limit_violation=True)
    res_bad = check_kinematic_limits(bad_ep)
    assert res_bad.passed is False
    assert len(res_bad.details["violations"]) > 0


def test_monotonic_timestamps_check():
    clean_ep = _make_dummy_episode()
    res_clean = check_monotonic_timestamps(clean_ep)
    assert res_clean.passed is True

    bad_ep = _make_dummy_episode(non_monotonic_ts=True)
    res_bad = check_monotonic_timestamps(bad_ep)
    assert res_bad.passed is False


def test_sampling_jitter_check():
    clean_ep = _make_dummy_episode()
    res = check_sampling_jitter(clean_ep)
    assert res.passed is True
    assert "mean_dt_ms" in res.details
    assert pytest.approx(res.details["mean_dt_ms"], 0.1) == 66.666


def test_cross_stream_alignment_check():
    clean_ep = _make_dummy_episode()
    res = check_cross_stream_alignment(clean_ep)
    assert res.passed is True


def test_input_qa_engine_verdict(tmp_path: Path):
    engine = InputQAEngine(reject_on_failure_flag=True)

    clean_ep = _make_dummy_episode(frames_dir=tmp_path / "frames")
    res_clean = engine.validate(clean_ep)
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.passed is True

    fail_ep = _make_dummy_episode(success=False)
    res_fail = engine.validate(fail_ep)
    assert res_fail.verdict == QAVerdict.FAIL
    assert res_fail.passed is False


def test_quarantine_dead_letter_queue(tmp_path: Path):
    mgr = QuarantineManager(rejected_dir=tmp_path / "rejected")
    bad_ep = _make_dummy_episode(nan_in_state=True, success=False)
    engine = InputQAEngine(reject_on_failure_flag=True)
    res = engine.validate(bad_ep)

    quarantine_path = mgr.quarantine(bad_ep, res)
    assert quarantine_path.exists()

    manifest_file = quarantine_path / "rejection_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file) as f:
        manifest = json.load(f)
    assert manifest["verdict"] == "FAIL"
    assert len(manifest["failing_checks"]) >= 1


def test_ingest_phone_session(tmp_path: Path):
    sess_dir = tmp_path / "phone_session_01"
    sess_dir.mkdir()

    meta = {
        "session_id": "phone_session_01",
        "device_id": "nothing_phone_2a",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "frame_count": 10,
        "timestamps_ns": [i * int(1e9 / 30) for i in range(10)],
    }
    with open(sess_dir / "session_meta.json", "w") as f:
        json.dump(meta, f)

    ep = ingest_phone_session(sess_dir)
    assert ep.source == "rigsim_phone"
    assert "phone_rgb" in ep.cameras
    assert ep.cameras["phone_rgb"].width == 1920
    assert ep.cameras["phone_rgb"].frame_count == 10
