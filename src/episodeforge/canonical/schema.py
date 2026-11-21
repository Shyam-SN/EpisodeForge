"""Canonical internal episode representation for EpisodeForge.

Every input (simulated MuJoCo demonstrations, RigSim multi-camera packages,
real phone egocentric sessions) is ingested and normalized into this schema
before any client transformation.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QAVerdict(enum.StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CameraConvention(enum.StrEnum):
    OPENCV = "opencv"  # +X right, +Y down, +Z forward
    OPENGL = "opengl"  # +X right, +Y up, -Z forward


class QuaternionOrder(enum.StrEnum):
    WXYZ = "wxyz"  # [w, x, y, z] - MuJoCo & SciPy default convention
    XYZW = "xyzw"  # [x, y, z, w] - ROS / pybullet convention


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length in x (pixels)")
    fy: float = Field(..., description="Focal length in y (pixels)")
    cx: float = Field(..., description="Principal point x (pixels)")
    cy: float = Field(..., description="Principal point y (pixels)")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    distortion_model: str = Field(
        default="pinhole", description="Distortion model (pinhole, radtan, fisheye)"
    )
    distortion_coeffs: list[float] = Field(
        default_factory=list, description="Distortion coefficients [k1, k2, p1, p2, k3]"
    )


class CameraExtrinsics(BaseModel):
    translation: list[float] = Field(
        ..., min_length=3, max_length=3, description="Translation [x, y, z] in meters"
    )
    quaternion: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Rotation quaternion [w, x, y, z] in canonical wxyz",
    )
    reference_frame: str = Field(default="robot_base", description="Reference frame of extrinsics")


class CameraStream(BaseModel):
    name: str = Field(..., description="Camera identifier, e.g. 'front', 'wrist', 'side'")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps_nominal: float = Field(..., gt=0)
    frame_count: int = Field(..., ge=0)
    convention: CameraConvention = CameraConvention.OPENCV
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    timestamps_ns: list[int] = Field(
        default_factory=list, description="Monotonic frame timestamps in nanoseconds"
    )
    video_path: str | None = Field(
        default=None, description="Path to lossless archive video (e.g. FFV1/lossless MP4)"
    )
    frames_dir: str | None = Field(
        default=None, description="Path to lossless PNG frames directory"
    )


class StateStream(BaseModel):
    timestamps_ns: list[int] = Field(
        default_factory=list, description="Monotonic timestamps in nanoseconds"
    )
    joint_names: list[str] = Field(default_factory=list)
    joint_positions: list[list[float]] = Field(
        default_factory=list, description="(T, N) joint positions in radians"
    )
    joint_velocities: list[list[float]] = Field(
        default_factory=list, description="(T, N) joint velocities in rad/s"
    )
    ee_positions: list[list[float]] = Field(
        default_factory=list, description="(T, 3) end-effector [x, y, z] in meters"
    )
    ee_quaternions: list[list[float]] = Field(
        default_factory=list, description="(T, 4) end-effector [w, x, y, z] in canonical wxyz"
    )
    gripper_widths: list[float] = Field(
        default_factory=list, description="(T,) gripper opening width in meters"
    )


class ActionStream(BaseModel):
    timestamps_ns: list[int] = Field(
        default_factory=list, description="Monotonic timestamps in nanoseconds"
    )
    nominal_rate_hz: float = Field(default=15.0, gt=0)
    joint_position_targets: list[list[float]] = Field(
        default_factory=list, description="(T, N) target joint positions in radians"
    )
    ee_delta_positions: list[list[float]] = Field(
        default_factory=list, description="(T, 3) target delta [dx, dy, dz] in meters"
    )
    ee_delta_quaternions: list[list[float]] = Field(
        default_factory=list, description="(T, 4) target delta [dw, dx, dy, dz] in wxyz"
    )
    gripper_targets: list[float] = Field(
        default_factory=list, description="(T,) target gripper width in meters"
    )


class IMUStream(BaseModel):
    timestamps_ns: list[int] = Field(default_factory=list)
    accelerometer: list[list[float]] = Field(
        default_factory=list, description="(T, 3) accel in m/s^2"
    )
    gyroscope: list[list[float]] = Field(
        default_factory=list, description="(T, 3) angular velocity in rad/s"
    )


class ConventionsMetadata(BaseModel):
    spatial_units: str = Field(default="meters", description="Length unit, e.g. meters")
    angular_units: str = Field(default="radians", description="Angle unit, e.g. radians")
    quaternion_order: QuaternionOrder = Field(
        default=QuaternionOrder.WXYZ, description="Internal canonical quaternion order"
    )
    base_frame: str = Field(default="robot_base")
    world_frame: str = Field(default="world")
    gripper_range_m: tuple[float, float] = Field(
        default=(0.0, 0.08), description="Min and max gripper width in meters"
    )


class ImperfectionLabel(BaseModel):
    injected: bool = False
    defect_type: str | None = (
        None  # e.g. "slipped_grasp", "timeout", "nan_glitch", "duplicate", "idle_padding"
    )
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceMetadata(BaseModel):
    input_content_hash: str
    generator_version: str = "0.1.0"
    git_commit: str = "unknown"
    created_at_utc: str
    host_platform: str = "darwin-arm64"


class CanonicalEpisode(BaseModel):
    """The central internal representation of a robot demonstration or sensory episode."""

    episode_id: str = Field(..., description="Deterministic content-hash id, e.g. sha256[:16]")
    source: str = Field(..., description="'generated_mujoco', 'rigsim_sim', 'rigsim_phone', etc.")
    source_ref: str = Field(..., description="Original raw identifier, filename or session id")
    task: str = Field(..., description="Task identifier, e.g. 'pick_place', 'push', 'stack'")
    instruction: str = Field(..., description="Natural language task instruction")
    success: bool = Field(
        ..., description="Whether task succeeded according to termination criteria"
    )
    seed: int = Field(default=0, description="Random seed used during generation")
    randomization: dict[str, Any] = Field(
        default_factory=dict, description="Domain randomization parameters applied"
    )

    # Conventions
    conventions: ConventionsMetadata = Field(default_factory=ConventionsMetadata)

    # Sensor and control streams
    cameras: dict[str, CameraStream] = Field(default_factory=dict)
    state: StateStream = Field(default_factory=StateStream)
    action: ActionStream = Field(default_factory=ActionStream)
    imu: IMUStream | None = None

    # Quality and provenance
    upstream_qa_verdict: QAVerdict = QAVerdict.PASS
    upstream_qa_details: dict[str, Any] = Field(default_factory=dict)
    imperfection: ImperfectionLabel = Field(default_factory=ImperfectionLabel)
    provenance: ProvenanceMetadata

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, v: str) -> str:
        if not v or len(v) < 8:
            raise ValueError(f"Invalid episode_id '{v}': must be non-empty and at least 8 chars")
        return v
