"""Unit and property-based tests for Milestone 3 Transform Library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from episodeforge.config.client_spec import ClientSpec
from episodeforge.ingest.generated import ingest_generated_episode
from episodeforge.transform.pipeline import apply_client_transforms
from episodeforge.transform.resample import (
    resample_actions_zoh,
    resample_continuous,
)
from episodeforge.transform.rotations import (
    axis_angle_to_quat_wxyz,
    euler_xyz_to_quat_wxyz,
    matrix_to_quat_wxyz,
    normalize_quaternion,
    quat_wxyz_to_axis_angle,
    quat_wxyz_to_euler_xyz,
    quat_wxyz_to_matrix,
    quat_wxyz_to_rot6d,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
    slerp,
)
from episodeforge.transform.splits import (
    compute_deterministic_split,
    verify_split_leakage,
)
from episodeforge.transform.stats import WelfordAccumulator

# --- Rotations & Math Tests ---


def test_quaternion_conventions_roundtrip():
    q_wxyz = [0.7071068, 0.0, 0.7071068, 0.0]
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    assert q_xyzw == [0.0, 0.7071068, 0.0, 0.7071068]
    q_back = quat_xyzw_to_wxyz(q_xyzw)
    assert pytest.approx(q_back, 1e-6) == q_wxyz


def test_matrix_quaternion_roundtrip():
    # 90 deg rotation around Z
    theta = np.pi / 2.0
    q = [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)]
    R = quat_wxyz_to_matrix(q)
    # Check SO(3) properties: R * R^T = I, det(R) = 1
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert pytest.approx(np.linalg.det(R), 1e-7) == 1.0

    q_recovered = matrix_to_quat_wxyz(R)
    assert pytest.approx(q_recovered, 1e-6) == q


def test_rot6d_roundtrip():
    q = normalize_quaternion(np.array([0.5, 0.5, 0.5, 0.5])).tolist()
    v6 = quat_wxyz_to_rot6d(q)
    assert len(v6) == 6
    R = rot6d_to_matrix(v6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)

    q_recovered = rot6d_to_quat_wxyz(v6)
    # Signs can be antipodal, verify same rotation matrix
    R_recovered = quat_wxyz_to_matrix(q_recovered)
    assert np.allclose(R, R_recovered, atol=1e-6)


def test_axis_angle_roundtrip():
    aa = [0.1, -0.2, 0.3]
    q = axis_angle_to_quat_wxyz(aa)
    aa_recovered = quat_wxyz_to_axis_angle(q)
    assert pytest.approx(aa_recovered, 1e-6) == aa


def test_euler_xyz_roundtrip():
    angles = [0.2, -0.3, 0.4]
    q = euler_xyz_to_quat_wxyz(angles)
    angles_recovered = quat_wxyz_to_euler_xyz(q)
    assert pytest.approx(angles_recovered, 1e-6) == angles


def test_slerp_properties():
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [0.0, 1.0, 0.0, 0.0]

    # Boundaries
    assert pytest.approx(slerp(q0, q1, 0.0), 1e-6) == q0
    assert pytest.approx(slerp(q0, q1, 1.0), 1e-6) == q1

    # Midpoint (45 deg)
    q_mid = slerp(q0, q1, 0.5)
    assert pytest.approx(np.linalg.norm(q_mid), 1e-6) == 1.0
    assert pytest.approx(q_mid[0], 1e-6) == np.cos(np.pi / 4)
    assert pytest.approx(q_mid[1], 1e-6) == np.sin(np.pi / 4)

    # Antipodal shortest path check
    q1_neg = [-x for x in q1]
    q_mid_neg = slerp(q0, q1_neg, 0.5)
    # Must interpolate along same hemisphere
    assert pytest.approx(q_mid_neg[0], 1e-6) == np.cos(np.pi / 4)


@settings(max_examples=50)
@given(
    w=st.floats(min_value=-1.0, max_value=1.0),
    x=st.floats(min_value=-1.0, max_value=1.0),
    y=st.floats(min_value=-1.0, max_value=1.0),
    z=st.floats(min_value=-1.0, max_value=1.0),
)
def test_hypothesis_quaternion_normalization_and_matrix(w, x, y, z):
    raw = np.array([w, x, y, z], dtype=np.float64)
    if np.linalg.norm(raw) < 1e-6:
        return
    q = normalize_quaternion(raw).tolist()
    assert pytest.approx(np.linalg.norm(q), 1e-7) == 1.0
    assert q[0] >= 0.0  # Positive scalar convention

    R = quat_wxyz_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert pytest.approx(np.linalg.det(R), 1e-6) == 1.0


# --- Resampling Tests ---


def test_resample_continuous():
    src_times = [0, 100, 200]
    src_vals = [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]
    tgt_times = [50, 150]

    interp = resample_continuous(src_times, src_vals, tgt_times)
    assert pytest.approx(interp[0], 1e-6) == [0.5, 15.0]
    assert pytest.approx(interp[1], 1e-6) == [1.5, 25.0]


def test_resample_actions_zoh():
    src_times = [0, 100, 200]
    src_actions = [1.0, 2.0, 3.0]
    tgt_times = [0, 49, 99, 100, 150, 200, 250]

    out = resample_actions_zoh(src_times, src_actions, tgt_times)
    assert out == [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


# --- Splits & Leakage Tests ---


def test_deterministic_splits_zero_leakage():
    ids = [f"ep_{i:04d}" for i in range(100)]
    splits = compute_deterministic_split(ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 100

    # Leakage check must pass
    verify_split_leakage(splits)

    # Determinism
    splits2 = compute_deterministic_split(ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert splits == splits2


# --- Streaming Statistics Tests ---


def test_welford_accumulator_streaming():
    np.random.seed(42)
    data = np.random.randn(500, 4)

    acc = WelfordAccumulator(dim=4)
    # Stream in chunks
    for chunk in np.array_split(data, 10):
        acc.update_batch(chunk)

    stats = acc.finalize()
    assert stats.count == 500
    assert pytest.approx(stats.mean, 1e-6) == np.mean(data, axis=0).tolist()
    assert pytest.approx(stats.std, 1e-5) == np.std(data, axis=0, ddof=1).tolist()
    assert pytest.approx(stats.min, 1e-6) == np.min(data, axis=0).tolist()
    assert pytest.approx(stats.max, 1e-6) == np.max(data, axis=0).tolist()


# --- Full Client Pipeline Application Tests ---


@pytest.mark.parametrize(
    "spec_path",
    [
        "configs/clients/client_a_lerobot.v1.yaml",
        "configs/clients/client_a_lerobot.v2.yaml",
        "configs/clients/client_b_rlds.v1.yaml",
        "configs/clients/client_c_robomimic.v1.yaml",
    ],
)
def test_apply_client_transforms_on_real_episode(spec_path: str):
    p = Path(spec_path)
    if not p.exists():
        pytest.skip(f"Spec {spec_path} does not exist")

    with open(p) as f:
        spec_data = yaml.safe_load(f)
    spec = ClientSpec.model_validate(spec_data)

    ep_dir = Path("data/raw/m1_episodes/ep_000000")
    if not ep_dir.exists():
        pytest.skip("m1_episodes/ep_000000 not generated yet")

    ep = ingest_generated_episode(ep_dir)
    # Make sure ep passes selection
    ep.success = True

    transformed = apply_client_transforms(ep, spec)
    assert transformed is not None
    assert transformed.length > 0
    assert transformed.actions.shape[0] == transformed.length
    assert len(transformed.action_dim_names) == transformed.actions.shape[1]
    assert len(transformed.timestamps_ns) == transformed.length
