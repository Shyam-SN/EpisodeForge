"""Ingest adapter for RigSim Episode packages (Project 1)."""

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
    IMUStream,
    ProvenanceMetadata,
    QAVerdict,
    QuaternionOrder,
    StateStream,
)


def ingest_rigsim_package(package_dir: Path | str) -> CanonicalEpisode:
    """Ingest a RigSim package directory into CanonicalEpisode."""
    pkg_path = Path(package_dir).resolve()
    meta_path = pkg_path / "episode_meta.json"
    qa_path = pkg_path / "qa_verdict.json"
    traj_path = pkg_path / "robot" / "trajectory.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episode_meta.json in {pkg_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # QA verdict if available
    upstream_verdict = QAVerdict.PASS
    qa_details = {}
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            qa_details = json.load(f)
            upstream_verdict = QAVerdict(qa_details.get("verdict", "PASS"))

    # Trajectory dataframe
    if traj_path.exists():
        traj_df = pd.read_parquet(traj_path)
    else:
        traj_df = pd.DataFrame()

    # Content hashing
    hasher = hashlib.sha256()
    hasher.update(pkg_path.name.encode())
    hasher.update(str(meta.get("session_id", "")).encode())
    hasher.update(str(meta.get("seed", 0)).encode())
    content_hash = hasher.hexdigest()
    canonical_id = f"ep_{content_hash[:12]}"

    # Camera streams
    cams_dir = pkg_path / "cameras"
    camera_streams: dict[str, CameraStream] = {}
    if cams_dir.exists():
        for cam_sub in cams_dir.iterdir():
            if cam_sub.is_dir():
                cam_name = cam_sub.name.replace("cam_", "")
                intr_path = cam_sub / "intrinsics.json"
                extr_path = cam_sub / "extrinsics.json"

                intr_data = {}
                if intr_path.exists():
                    with open(intr_path, encoding="utf-8") as f:
                        intr_data = json.load(f)

                intrinsics = CameraIntrinsics(
                    fx=float(intr_data.get("fx", 600.0)),
                    fy=float(intr_data.get("fy", 600.0)),
                    cx=float(intr_data.get("cx", 320.0)),
                    cy=float(intr_data.get("cy", 240.0)),
                    width=int(intr_data.get("width", 640)),
                    height=int(intr_data.get("height", 480)),
                    distortion_model=intr_data.get("distortion_model", "pinhole"),
                    distortion_coeffs=intr_data.get("distortion_coeffs", []),
                )

                extr_data = {}
                if extr_path.exists():
                    with open(extr_path, encoding="utf-8") as f:
                        extr_data = json.load(f)

                extrinsics = CameraExtrinsics(
                    translation=extr_data.get("translation", [0.0, 0.0, 0.0]),
                    quaternion=extr_data.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
                    reference_frame=extr_data.get("reference_frame", "robot_base"),
                )

                # Check video or frames
                video_file = cam_sub / "video.mp4"
                video_path = str(video_file) if video_file.exists() else None
                frames_dir = (
                    str(cam_sub / "frames") if (cam_sub / "frames").is_dir() else str(cam_sub)
                )

                # Frame timestamps
                ts_path = cam_sub / "timestamps.json"
                ts_list: list[int] = []
                if ts_path.exists():
                    with open(ts_path, encoding="utf-8") as f:
                        ts_list = json.load(f)

                camera_streams[cam_name] = CameraStream(
                    name=cam_name,
                    width=intrinsics.width,
                    height=intrinsics.height,
                    fps_nominal=float(meta.get("camera_fps", 30.0)),
                    frame_count=len(ts_list) if ts_list else int(meta.get("frame_count", 0)),
                    convention=CameraConvention.OPENCV,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    timestamps_ns=ts_list,
                    video_path=video_path,
                    frames_dir=frames_dir if not video_path else None,
                )

    # Robot state and action streams
    if not traj_df.empty:
        joint_cols = [c for c in traj_df.columns if "joint" in c or c.startswith("qpos")]
        num_joints = len(joint_cols) if joint_cols else 7
        joint_names = [f"panda_joint{i + 1}" for i in range(num_joints)]

        state_stream = StateStream(
            timestamps_ns=traj_df["timestamp_ns"].tolist()
            if "timestamp_ns" in traj_df
            else [int(t * 1e9) for t in traj_df.get("timestamp_s", [])],
            joint_names=joint_names,
            joint_positions=[list(q) for q in traj_df.get("joint_positions", [])],
            joint_velocities=[list(qd) for qd in traj_df.get("joint_velocities", [])],
            ee_positions=[list(p) for p in traj_df.get("ee_positions", [])],
            ee_quaternions=[list(quat) for quat in traj_df.get("ee_quaternions", [])],
            gripper_widths=traj_df.get("gripper_widths", pd.Series(dtype=float))
            .astype(float)
            .tolist(),
        )

        action_stream = ActionStream(
            timestamps_ns=state_stream.timestamps_ns,
            nominal_rate_hz=float(meta.get("control_rate_hz", 15.0)),
            joint_position_targets=[
                list(q) for q in traj_df.get("action_joint_targets", traj_df.get("target_qpos", []))
            ],
            ee_delta_positions=[
                list(dp)
                for dp in traj_df.get("action_ee_delta", traj_df.get("target_ee_delta", []))
            ],
            gripper_targets=traj_df.get(
                "action_gripper", traj_df.get("target_gripper", pd.Series(dtype=float))
            )
            .astype(float)
            .tolist(),
        )
    else:
        state_stream = StateStream()
        action_stream = ActionStream()

    # IMU stream if available
    imu_stream = None
    imu_file = pkg_path / "imu" / "imu_data.parquet"
    if not imu_file.exists():
        imu_file = pkg_path / "imu" / "imu_data.csv"
    if imu_file.exists():
        idf = pd.read_parquet(imu_file) if imu_file.suffix == ".parquet" else pd.read_csv(imu_file)
        imu_stream = IMUStream(
            timestamps_ns=idf["timestamp_ns"].tolist(),
            accelerometer=[list(a) for a in idf[["accel_x", "accel_y", "accel_z"]].values],
            gyroscope=[list(g) for g in idf[["gyro_x", "gyro_y", "gyro_z"]].values],
        )

    return CanonicalEpisode(
        episode_id=canonical_id,
        source="rigsim_package",
        source_ref=meta.get("session_id", pkg_path.name),
        task=meta.get("task_name", meta.get("task", "unknown")),
        instruction=meta.get("instruction", ""),
        success=bool(meta.get("success", False)),
        seed=int(meta.get("seed", 0)),
        randomization=meta.get("rig_config", {}),
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
        imu=imu_stream,
        upstream_qa_verdict=upstream_verdict,
        upstream_qa_details=qa_details,
        imperfection=ImperfectionLabel(),
        provenance=ProvenanceMetadata(
            input_content_hash=content_hash,
            generator_version="rigsim-1.0.0",
            git_commit=meta.get("git_commit", "unknown"),
            created_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            host_platform="rigsim",
        ),
    )
