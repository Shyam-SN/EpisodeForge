"""Ingest adapter for simulated MuJoCo demonstration packages."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

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
    QuaternionOrder,
    StateStream,
)


def ingest_generated_episode(episode_dir: Path | str) -> CanonicalEpisode:
    """Load a generated MuJoCo demonstration package into CanonicalEpisode."""
    ep_path = Path(episode_dir).resolve()
    meta_path = ep_path / "episode_meta.json"
    qa_path = ep_path / "ground_truth_qa.json"
    traj_path = ep_path / "trajectory.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episode_meta.json in {ep_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory.parquet in {ep_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # QA / Imperfection ground truth if present
    upstream_verdict = QAVerdict.PASS
    imperfection = ImperfectionLabel()
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            qa_data = json.load(f)
            upstream_verdict = QAVerdict(qa_data.get("expected_qa_verdict", "PASS"))
            imperfection = ImperfectionLabel(
                injected=bool(qa_data.get("injected_imperfection", False)),
                defect_type=qa_data.get("defect_type"),
                details=qa_data,
            )

    # Read trajectory
    traj_df = pd.read_parquet(traj_path)

    # Compute deterministic content hash based on meta + trajectory
    hasher = hashlib.sha256()
    hasher.update(ep_path.name.encode())
    hasher.update(str(meta.get("seed", 0)).encode())
    hasher.update(meta.get("task", "").encode())
    hasher.update(str(len(traj_df)).encode())
    if len(traj_df) > 0:
        first_q = traj_df["joint_positions"].iloc[0]
        last_q = traj_df["joint_positions"].iloc[-1]
        hasher.update(str(first_q).encode())
        hasher.update(str(last_q).encode())
    content_hash = hasher.hexdigest()
    canonical_id = f"ep_{content_hash[:12]}"

    # Parse camera streams
    camera_streams: dict[str, CameraStream] = {}
    for cam_name, cinfo in meta.get("cameras", {}).items():
        intr_data = cinfo.get("intrinsics", {})
        intrinsics = CameraIntrinsics(
            fx=float(intr_data.get("fx", 500.0)),
            fy=float(intr_data.get("fy", 500.0)),
            cx=float(intr_data.get("cx", 320.0)),
            cy=float(intr_data.get("cy", 240.0)),
            width=int(intr_data.get("width", 640)),
            height=int(intr_data.get("height", 480)),
            distortion_model=intr_data.get("distortion_model", "pinhole"),
            distortion_coeffs=intr_data.get("distortion_coeffs", []),
        )

        # Extrinsics relative to robot_base
        if cam_name == "front":
            extr_trans = [0.75, 0.0, 0.65]
            extr_quat = [0.2706, 0.6533, -0.6533, 0.2706]
        elif cam_name == "wrist":
            extr_trans = [0.0, 0.0, 0.05]
            extr_quat = [1.0, 0.0, 0.0, 0.0]
        else:  # side
            extr_trans = [0.0, 0.75, 0.60]
            extr_quat = [0.3827, 0.0, 0.0, 0.9239]

        extrinsics = CameraExtrinsics(
            translation=extr_trans,
            quaternion=extr_quat,
            reference_frame="robot_base" if cam_name != "wrist" else "panda_hand",
        )

        frames_dir_rel = cinfo.get("frames_dir")
        frames_dir_abs = str(ep_path / "cameras" / cam_name) if frames_dir_rel else None

        camera_streams[cam_name] = CameraStream(
            name=cam_name,
            width=intrinsics.width,
            height=intrinsics.height,
            fps_nominal=30.0,
            frame_count=int(cinfo.get("frame_count", len(cinfo.get("timestamps_ns", [])))),
            convention=CameraConvention.OPENCV,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps_ns=cinfo.get("timestamps_ns", []),
            frames_dir=frames_dir_abs,
        )

    # Parse state stream
    joint_names = [f"panda_joint{i + 1}" for i in range(7)]
    state_stream = StateStream(
        timestamps_ns=traj_df["timestamp_ns"].tolist(),
        joint_names=joint_names,
        joint_positions=[list(q) for q in traj_df["joint_positions"]],
        joint_velocities=[list(qd) for qd in traj_df["joint_velocities"]],
        ee_positions=[list(p) for p in traj_df["ee_positions"]],
        ee_quaternions=[list(quat) for quat in traj_df["ee_quaternions"]],
        gripper_widths=traj_df["gripper_widths"].astype(float).tolist(),
    )

    # Parse action stream
    action_stream = ActionStream(
        timestamps_ns=traj_df["timestamp_ns"].tolist(),
        nominal_rate_hz=15.0,
        joint_position_targets=[list(q) for q in traj_df["action_joint_targets"]],
        ee_delta_positions=[list(dp) for dp in traj_df["action_ee_delta"]],
        gripper_targets=traj_df["action_gripper"].astype(float).tolist(),
    )

    # Build CanonicalEpisode
    return CanonicalEpisode(
        episode_id=canonical_id,
        source="generated_mujoco",
        source_ref=meta.get("episode_id", ep_path.name),
        task=meta.get("task", "unknown"),
        instruction=meta.get("instruction", ""),
        success=bool(meta.get("success", False)),
        seed=int(meta.get("seed", 0)),
        randomization=meta.get("randomization", {}),
        conventions=ConventionsMetadata(
            spatial_units="meters",
            angular_units="radians",
            quaternion_order=QuaternionOrder.WXYZ,
            base_frame="robot_base",
            world_frame="world",
            gripper_range_m=(0.0, 0.08),
        ),
        cameras=camera_streams,
        state=state_stream,
        action=action_stream,
        imu=None,
        upstream_qa_verdict=upstream_verdict,
        upstream_qa_details=qa_data if qa_path.exists() else {},
        imperfection=imperfection,
        provenance=ProvenanceMetadata(
            input_content_hash=content_hash,
            generator_version="0.1.0",
            git_commit="head",
            created_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            host_platform="darwin-arm64",
        ),
    )
