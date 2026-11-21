"""Ingest adapter for Nothing Phone egocentric captures (Client D)."""

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


def ingest_phone_session(session_dir: Path | str) -> CanonicalEpisode:
    """Ingest a real Nothing Phone egocentric capture into CanonicalEpisode."""
    sess_path = Path(session_dir).resolve()
    meta_path = sess_path / "session_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing session_meta.json in {sess_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Content hash
    hasher = hashlib.sha256()
    hasher.update(sess_path.name.encode())
    hasher.update(str(meta.get("session_id", "")).encode())
    hasher.update(str(meta.get("device_id", "nothing_phone")).encode())
    content_hash = hasher.hexdigest()
    canonical_id = f"ep_{content_hash[:12]}"

    # Camera video / frames
    video_path = str(sess_path / "video.mp4") if (sess_path / "video.mp4").exists() else None
    frames_dir = (
        str(sess_path / "frames")
        if (sess_path / "frames").is_dir()
        else (None if video_path else str(sess_path))
    )

    width = int(meta.get("width", 1920))
    height = int(meta.get("height", 1080))
    fps = float(meta.get("fps", 30.0))

    intrinsics = CameraIntrinsics(
        fx=float(meta.get("fx", 1450.0)),
        fy=float(meta.get("fy", 1450.0)),
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
        distortion_model="radtan",
        distortion_coeffs=meta.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0]),
    )

    extrinsics = CameraExtrinsics(
        translation=[0.0, 0.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        reference_frame="phone_body",
    )

    # Timestamps
    ts_list: list[int] = meta.get("timestamps_ns", [])
    if not ts_list:
        ts_file = sess_path / "timestamps.json"
        if ts_file.exists():
            with open(ts_file, encoding="utf-8") as f:
                ts_list = json.load(f)
        elif "frame_count" in meta:
            frame_count = int(meta["frame_count"])
            dt_ns = int(round(1e9 / fps))
            ts_list = [i * dt_ns for i in range(frame_count)]

    phone_cam = CameraStream(
        name="phone_rgb",
        width=width,
        height=height,
        fps_nominal=fps,
        frame_count=len(ts_list),
        convention=CameraConvention.OPENCV,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        timestamps_ns=ts_list,
        video_path=video_path,
        frames_dir=frames_dir,
    )

    # IMU stream
    imu_stream = None
    imu_file = sess_path / "imu.csv"
    if not imu_file.exists():
        imu_file = sess_path / "imu.parquet"
    if imu_file.exists():
        idf = pd.read_csv(imu_file) if imu_file.suffix == ".csv" else pd.read_parquet(imu_file)
        imu_stream = IMUStream(
            timestamps_ns=idf["timestamp_ns"].tolist()
            if "timestamp_ns" in idf
            else [int(t * 1e9) for t in idf.get("timestamp_s", [])],
            accelerometer=[list(a) for a in idf[["accel_x", "accel_y", "accel_z"]].values],
            gyroscope=[list(g) for g in idf[["gyro_x", "gyro_y", "gyro_z"]].values],
        )

    return CanonicalEpisode(
        episode_id=canonical_id,
        source="rigsim_phone",
        source_ref=meta.get("session_id", sess_path.name),
        task=meta.get("task", "egocentric_manipulation"),
        instruction=meta.get("instruction", "Human egocentric manipulation demonstration"),
        success=bool(meta.get("success", True)),
        seed=int(meta.get("seed", 0)),
        randomization=meta.get("device_info", {}),
        conventions=ConventionsMetadata(
            spatial_units="meters",
            angular_units="radians",
            quaternion_order=QuaternionOrder.WXYZ,
            base_frame="phone_body",
            world_frame="world",
            gripper_range_m=(0.0, 0.0),
        ),
        cameras={"phone_rgb": phone_cam},
        state=StateStream(),
        action=ActionStream(),
        imu=imu_stream,
        upstream_qa_verdict=QAVerdict.PASS,
        upstream_qa_details={},
        imperfection=ImperfectionLabel(),
        provenance=ProvenanceMetadata(
            input_content_hash=content_hash,
            generator_version="nothing_capture-1.0",
            git_commit="unknown",
            created_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            host_platform="android-arm64",
        ),
    )
