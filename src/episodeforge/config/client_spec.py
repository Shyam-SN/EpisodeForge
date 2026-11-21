"""Client specification schema and validation logic for EpisodeForge."""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FormatType(enum.StrEnum):
    LEROBOT = "lerobot"
    RLDS = "rlds"
    ROBOMIMIC_HDF5 = "robomimic_hdf5"
    WEBDATASET = "webdataset"


class ResizeMode(enum.StrEnum):
    CENTER_CROP = "center_crop"
    LETTERBOX = "letterbox"
    STRETCH = "stretch"


class ImageCodec(enum.StrEnum):
    AV1 = "av1"
    H264 = "h264"
    HEVC = "hevc"
    RAW = "raw"
    JPEG = "jpeg"
    PNG = "png"


class ActionRepresentation(enum.StrEnum):
    JOINT_POS = "joint_pos"
    EE_DELTA_POSE = "ee_delta_pose"
    EE_ABS_POSE = "ee_abs_pose"


class RotationFormat(enum.StrEnum):
    QUAT_XYZW = "quat_xyzw"
    QUAT_WXYZ = "quat_wxyz"
    EULER_XYZ = "euler_xyz"
    AXIS_ANGLE = "axis_angle"
    ROT6D = "rot6d"


class GripperConvention(enum.StrEnum):
    ZERO_TO_ONE = "zero_to_one"
    MINUS_ONE_TO_ONE = "minus_one_to_one"
    RAW_WIDTH_M = "raw_width_m"


class ResampleMethod(enum.StrEnum):
    NEAREST = "nearest"
    LINEAR = "linear"
    SLERP = "slerp"
    ZOH = "zoh"


class SelectionSpec(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["generated_mujoco"])
    tasks: list[str] | None = None
    require_success: bool = True
    min_qa_verdict: Literal["PASS", "WARN"] = "PASS"
    episode_length_bounds_sec: tuple[float, float] | None = None
    episode_length_bounds_steps: tuple[int, int] | None = None
    max_episodes: int | None = None
    balance_per_task: bool = True


class TimingSpec(BaseModel):
    target_fps: float = Field(..., gt=0)
    resampling_method_state: ResampleMethod = ResampleMethod.LINEAR
    resampling_method_rotation: ResampleMethod = ResampleMethod.SLERP
    resampling_method_action: ResampleMethod = ResampleMethod.ZOH
    resampling_method_camera: ResampleMethod = ResampleMethod.NEAREST
    max_timestamp_error_ms: float = Field(default=20.0, ge=0.0)


class CameraTargetSpec(BaseModel):
    target_key: str
    resolution: tuple[int, int] = Field(..., description="(width, height)")
    resize_mode: ResizeMode = ResizeMode.CENTER_CROP
    color_space: Literal["rgb", "bgr", "gray"] = "rgb"
    dtype: Literal["uint8", "float32"] = "uint8"
    codec: ImageCodec = ImageCodec.AV1
    crf: int | None = Field(
        default=28, ge=0, le=63, description="Constant Rate Factor for video codecs (0-63)"
    )
    quality: int | None = Field(
        default=None, ge=1, le=100, description="Quality setting for image codecs like JPEG (1-100)"
    )
    bitrate_kbps: int | None = None
    pixel_format: str = "yuv420p"


class StateSpec(BaseModel):
    include_fields: list[str] = Field(default_factory=lambda: ["joint_positions", "gripper_width"])
    order: list[str] = Field(default_factory=list)
    units_length: Literal["m", "mm"] = "m"
    units_angle: Literal["rad", "deg"] = "rad"
    target_key: str = "observation.state"


class ActionSpec(BaseModel):
    representation: ActionRepresentation = ActionRepresentation.JOINT_POS
    rotation_format: RotationFormat = RotationFormat.QUAT_XYZW
    units_length: Literal["m", "mm"] = "m"
    units_angle: Literal["rad", "deg"] = "rad"
    gripper_convention: GripperConvention = GripperConvention.ZERO_TO_ONE
    gripper_inverted: bool = False  # If True: 1 is closed, 0 is open (or vice-versa)
    chunking_horizon: int | None = None
    clip_range: tuple[float, float] | None = None
    target_key: str = "action"


class FramesSpec(BaseModel):
    target_base_frame: str = "robot_base"
    camera_convention: Literal["opencv", "opengl"] = "opencv"
    handedness: Literal["right", "left"] = "right"


class TrimmingSpec(BaseModel):
    trim_idle_start: bool = True
    trim_idle_end: bool = True
    velocity_threshold: float = Field(default=0.01, ge=0.0)
    padding_steps: int = Field(default=2, ge=0)


class LanguageSpec(BaseModel):
    target_key: str = "task"
    paraphrase_augmentation: bool = False


class SplitsSpec(BaseModel):
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    split_by: Literal["episode"] = "episode"
    stratify_by_task: bool = True
    seed: int = 42
    leakage_check: bool = True

    @model_validator(mode="after")
    def validate_ratios(self) -> SplitsSpec:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return self


class StatisticsSpec(BaseModel):
    compute_stats: bool = True
    target_keys: list[str] = Field(default_factory=lambda: ["observation.state", "action"])
    metrics: list[str] = Field(default_factory=lambda: ["mean", "std", "min", "max", "q01", "q99"])


class PackagingSpec(BaseModel):
    sharding_max_episodes_per_shard: int | None = 1000
    sharding_max_size_mb: int | None = 2000
    compression: Literal["gzip", "lzf", "none"] = "none"
    generate_datacard: bool = True


class DeliverySpec(BaseModel):
    bucket: str = "robotics-data-lake"
    prefix: str = "deliveries"
    release_naming: str = "v{n}"
    checksum_algorithm: Literal["sha256", "md5"] = "sha256"
    manifest_format: Literal["json", "parquet", "both"] = "both"
    compression_for_upload: Literal["tar.gz", "zip", "none"] = "none"


class AcceptanceCriteria(BaseModel):
    max_loader_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_timestamp_error_ms: float = Field(default=20.0, ge=0.0)
    min_episodes_delivered_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    verify_stats_roundtrip: bool = True


class ClientSpec(BaseModel):
    """Declarative specification defining exact client requirements for delivery."""

    client: str = Field(..., description="Unique client identifier, e.g. 'client_a'")
    spec_version: str = Field(..., description="Specification version string, e.g. 'v1'")
    dataset_name: str = Field(..., description="Dataset name, e.g. 'panda_manipulation_lerobot'")
    format: FormatType = Field(..., description="Target schema format")
    format_version: str = Field(..., description="Target format specification version")

    selection: SelectionSpec = Field(default_factory=SelectionSpec)
    timing: TimingSpec
    cameras: dict[str, CameraTargetSpec] = Field(default_factory=dict)
    state: StateSpec = Field(default_factory=StateSpec)
    action: ActionSpec = Field(default_factory=ActionSpec)
    frames: FramesSpec = Field(default_factory=FramesSpec)
    trimming: TrimmingSpec = Field(default_factory=TrimmingSpec)
    language: LanguageSpec = Field(default_factory=LanguageSpec)
    splits: SplitsSpec = Field(default_factory=SplitsSpec)
    statistics: StatisticsSpec = Field(default_factory=StatisticsSpec)
    packaging: PackagingSpec = Field(default_factory=PackagingSpec)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)

    @field_validator("spec_version")
    @classmethod
    def validate_spec_version(cls, v: str) -> str:
        if not v.startswith("v"):
            raise ValueError(f"spec_version must start with 'v', got '{v}'")
        return v
