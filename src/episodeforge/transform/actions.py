"""Action representation transformations for client specifications."""

from __future__ import annotations

import numpy as np

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.config.client_spec import (
    ActionRepresentation,
    ActionSpec,
    GripperConvention,
    RotationFormat,
)
from episodeforge.transform.rotations import (
    quat_wxyz_to_axis_angle,
    quat_wxyz_to_euler_xyz,
    quat_wxyz_to_rot6d,
    quat_wxyz_to_xyzw,
)


def scale_gripper(
    width_m: float,
    gripper_convention: GripperConvention,
    max_width_m: float = 0.08,
    inverted: bool = False,
) -> float:
    """Scale gripper width from meters to requested client convention."""
    norm = float(np.clip(width_m / max_width_m, 0.0, 1.0))
    if inverted:
        norm = 1.0 - norm

    if gripper_convention == GripperConvention.RAW_WIDTH_M:
        return float(width_m)
    elif gripper_convention == GripperConvention.ZERO_TO_ONE:
        return norm
    elif gripper_convention == GripperConvention.MINUS_ONE_TO_ONE:
        return float(2.0 * norm - 1.0)
    else:
        return norm


def transform_actions(
    episode: CanonicalEpisode,
    action_spec: ActionSpec,
) -> tuple[np.ndarray, list[str]]:
    """Transform canonical action stream to target client action vector and dimension names."""
    T = len(episode.action.timestamps_ns)
    if T == 0:
        return np.empty((0, 0), dtype=np.float32), []

    components: list[np.ndarray] = []
    dim_names: list[str] = []

    # 1. Base motion components
    if action_spec.representation == ActionRepresentation.JOINT_POS:
        targets = np.asarray(episode.action.joint_position_targets, dtype=np.float32)
        # Exclude gripper from targets if it was attached at end
        if targets.shape[1] > len(episode.state.joint_names):
            targets = targets[:, : len(episode.state.joint_names)]
        components.append(targets)
        dim_names.extend([f"target_{name}" for name in episode.state.joint_names])

    elif action_spec.representation in (
        ActionRepresentation.EE_DELTA_POSE,
        ActionRepresentation.EE_ABS_POSE,
    ):
        # Translation
        dp = np.asarray(episode.action.ee_delta_positions, dtype=np.float32)
        if action_spec.units_length == "mm":
            dp = dp * 1000.0
        components.append(dp)
        dim_names.extend(["delta_x", "delta_y", "delta_z"])

        # Rotation component
        rots = []
        rot_names = []

        for t in range(T):
            if episode.action.ee_delta_quaternions:
                q = episode.action.ee_delta_quaternions[t]
            else:
                q = [1.0, 0.0, 0.0, 0.0]

            if action_spec.rotation_format == RotationFormat.ROT6D:
                r_val = quat_wxyz_to_rot6d(q)
                rot_names = ["rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"]
            elif action_spec.rotation_format == RotationFormat.EULER_XYZ:
                r_val = quat_wxyz_to_euler_xyz(q)
                if action_spec.units_angle == "deg":
                    r_val = [float(np.rad2deg(a)) for a in r_val]
                rot_names = ["delta_roll", "delta_pitch", "delta_yaw"]
            elif action_spec.rotation_format == RotationFormat.AXIS_ANGLE:
                r_val = quat_wxyz_to_axis_angle(q)
                rot_names = ["axis_angle_x", "axis_angle_y", "axis_angle_z"]
            elif action_spec.rotation_format == RotationFormat.QUAT_XYZW:
                r_val = quat_wxyz_to_xyzw(q)
                rot_names = ["delta_qx", "delta_qy", "delta_qz", "delta_qw"]
            else:  # canonical wxyz
                r_val = q
                rot_names = ["delta_qw", "delta_qx", "delta_qy", "delta_qz"]

            rots.append(r_val)

        components.append(np.asarray(rots, dtype=np.float32))
        dim_names.extend(rot_names)

    # 2. Gripper component
    if episode.action.gripper_targets:
        max_w = episode.conventions.gripper_range_m[1]
        gripper_scaled = [
            scale_gripper(
                w,
                action_spec.gripper_convention,
                max_w,
                inverted=action_spec.gripper_inverted,
            )
            for w in episode.action.gripper_targets
        ]
        components.append(np.asarray(gripper_scaled, dtype=np.float32)[:, np.newaxis])
        dim_names.append("gripper")

    action_matrix = (
        np.concatenate(components, axis=1) if components else np.empty((T, 0), dtype=np.float32)
    )
    return action_matrix, dim_names
