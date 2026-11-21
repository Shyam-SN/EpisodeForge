"""Spatial coordinate frame transforms between robot_base, end-effector, and camera frames."""

from __future__ import annotations

import numpy as np

from episodeforge.transform.rotations import (
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
)

# OpenCV (+X right, +Y down, +Z forward) to OpenGL (+X right, +Y up, -Z forward)
OPENCV_TO_OPENGL = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def transform_point(
    point: list[float] | np.ndarray,
    translation: list[float] | np.ndarray,
    quaternion_wxyz: list[float] | np.ndarray,
) -> list[float]:
    """Transform 3D point by applying rotation then translation: p_new = R * p + t."""
    p = np.asarray(point, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    R = quat_wxyz_to_matrix(quaternion_wxyz)

    p_new = R @ p + t
    return [float(x) for x in p_new]


def transform_camera_convention(
    intrinsics_or_points: np.ndarray,
    from_convention: str,
    to_convention: str,
) -> np.ndarray:
    """Transform camera coordinates between OpenCV and OpenGL standards."""
    if from_convention == to_convention:
        return intrinsics_or_points

    if from_convention == "opencv" and to_convention == "opengl":
        return intrinsics_or_points @ OPENCV_TO_OPENGL.T
    elif from_convention == "opengl" and to_convention == "opencv":
        return intrinsics_or_points @ OPENCV_TO_OPENGL.T

    return intrinsics_or_points


def compute_relative_ee_delta(
    p_prev: list[float] | np.ndarray,
    q_prev: list[float] | np.ndarray,
    p_curr: list[float] | np.ndarray,
    q_curr: list[float] | np.ndarray,
) -> tuple[list[float], list[float]]:
    """Compute translation delta in previous end-effector frame and relative delta quaternion.

    dp_ee = R_prev^T * (p_curr - p_prev)
    R_delta = R_prev^T * R_curr
    """
    p0 = np.asarray(p_prev, dtype=np.float64)
    p1 = np.asarray(p_curr, dtype=np.float64)

    R0 = quat_wxyz_to_matrix(q_prev)
    R1 = quat_wxyz_to_matrix(q_curr)

    dp_world = p1 - p0
    dp_ee = R0.T @ dp_world

    R_rel = R0.T @ R1
    q_rel = matrix_to_quat_wxyz(R_rel)

    return [float(x) for x in dp_ee], q_rel
