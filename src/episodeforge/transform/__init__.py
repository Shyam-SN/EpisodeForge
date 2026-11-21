"""Transform library for kinematics, resampling, frame conversions, and client schemas."""

from __future__ import annotations

from episodeforge.transform.actions import scale_gripper, transform_actions
from episodeforge.transform.frames import (
    compute_relative_ee_delta,
    transform_camera_convention,
    transform_point,
)
from episodeforge.transform.pipeline import TransformedEpisode, apply_client_transforms
from episodeforge.transform.resample import (
    compute_target_timestamps,
    resample_actions_zoh,
    resample_continuous,
    resample_nearest_frames,
    resample_quaternions,
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
    split_episodes_stratified,
    verify_split_leakage,
)
from episodeforge.transform.stats import StreamStatistics, WelfordAccumulator
from episodeforge.transform.trim import trim_idle_padding

__all__ = [
    "StreamStatistics",
    "TransformedEpisode",
    "WelfordAccumulator",
    "apply_client_transforms",
    "axis_angle_to_quat_wxyz",
    "compute_deterministic_split",
    "compute_relative_ee_delta",
    "compute_target_timestamps",
    "euler_xyz_to_quat_wxyz",
    "matrix_to_quat_wxyz",
    "normalize_quaternion",
    "quat_wxyz_to_axis_angle",
    "quat_wxyz_to_euler_xyz",
    "quat_wxyz_to_matrix",
    "quat_wxyz_to_rot6d",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "resample_actions_zoh",
    "resample_continuous",
    "resample_nearest_frames",
    "resample_quaternions",
    "rot6d_to_matrix",
    "rot6d_to_quat_wxyz",
    "scale_gripper",
    "slerp",
    "split_episodes_stratified",
    "transform_actions",
    "transform_camera_convention",
    "transform_point",
    "trim_idle_padding",
    "verify_split_leakage",
]
