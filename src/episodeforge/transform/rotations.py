"""Rigorous rotation representation transforms and SLERP interpolation."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length. Enforces positive scalar w."""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_norm = q / norm
    if q_norm[0] < 0:
        q_norm = -q_norm
    return np.asarray(q_norm, dtype=np.float64)


def quat_wxyz_to_xyzw(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion from canonical [w, x, y, z] to ROS/LeRobot [x, y, z, w]."""
    q_arr = np.asarray(q, dtype=np.float64)
    return [float(q_arr[1]), float(q_arr[2]), float(q_arr[3]), float(q_arr[0])]


def quat_xyzw_to_wxyz(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion from ROS/LeRobot [x, y, z, w] to canonical [w, x, y, z]."""
    q_arr = np.asarray(q, dtype=np.float64)
    return [float(q_arr[3]), float(q_arr[0]), float(q_arr[1]), float(q_arr[2])]


def quat_wxyz_to_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """Convert canonical [w, x, y, z] quaternion to 3x3 rotation matrix."""
    q_norm = normalize_quaternion(np.asarray(q, dtype=np.float64))
    w, x, y, z = q_norm[0], q_norm[1], q_norm[2], q_norm[3]

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to canonical [w, x, y, z] quaternion via Shepperd's algorithm."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return [float(x) for x in normalize_quaternion(np.array([w, x, y, z]))]


def quat_wxyz_to_rot6d(q: list[float] | np.ndarray) -> list[float]:
    """Convert canonical quaternion to continuous 6D rotation representation (Zhou et al., CVPR 2019).

    Returns [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z].
    """
    R = quat_wxyz_to_matrix(q)
    col0 = R[:, 0]
    col1 = R[:, 1]
    return [
        float(col0[0]),
        float(col0[1]),
        float(col0[2]),
        float(col1[0]),
        float(col1[1]),
        float(col1[2]),
    ]


def rot6d_to_matrix(v6: list[float] | np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthogonalization from 6D vector to SO(3) 3x3 rotation matrix."""
    v = np.asarray(v6, dtype=np.float64)
    a1 = v[:3]
    a2 = v[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    dot = np.dot(b1, a2)
    b2_raw = a2 - dot * b1
    b2 = b2_raw / (np.linalg.norm(b2_raw) + 1e-12)
    b3 = np.cross(b1, b2)

    return np.column_stack([b1, b2, b3])


def rot6d_to_quat_wxyz(v6: list[float] | np.ndarray) -> list[float]:
    """Convert 6D continuous rotation vector to canonical [w, x, y, z] quaternion."""
    R = rot6d_to_matrix(v6)
    return matrix_to_quat_wxyz(R)


def quat_wxyz_to_euler_xyz(q: list[float] | np.ndarray) -> list[float]:
    """Convert canonical [w, x, y, z] quaternion to extrinsic Euler XYZ angles (in radians)."""
    R = quat_wxyz_to_matrix(q)
    # R = Rx(roll) * Ry(pitch) * Rz(yaw)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return [float(x), float(y), float(z)]


def euler_xyz_to_quat_wxyz(angles: list[float] | np.ndarray) -> list[float]:
    """Convert extrinsic Euler XYZ angles (radians) to canonical [w, x, y, z] quaternion."""
    ai, aj, ak = (
        float(angles[0]) / 2.0,
        float(angles[1]) / 2.0,
        float(angles[2]) / 2.0,
    )
    ci, si = np.cos(ai), np.sin(ai)
    cj, sj = np.cos(aj), np.sin(aj)
    ck, sk = np.cos(ak), np.sin(ak)

    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    w = cj * cc + sj * ss
    x = cj * sc - sj * cs
    y = cj * ss + sj * cc
    z = cj * cs - sj * sc

    return [float(x) for x in normalize_quaternion(np.array([w, x, y, z]))]


def quat_wxyz_to_axis_angle(q: list[float] | np.ndarray) -> list[float]:
    """Convert canonical [w, x, y, z] quaternion to axis-angle vector [ax, ay, az].

    Magnitude of the vector is angle in radians.
    """
    q_norm = normalize_quaternion(np.asarray(q, dtype=np.float64))
    w = np.clip(q_norm[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)

    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8 or angle < 1e-8:
        return [0.0, 0.0, 0.0]

    axis = q_norm[1:4] / sin_half
    axis_angle = axis * angle
    return [float(axis_angle[0]), float(axis_angle[1]), float(axis_angle[2])]


def axis_angle_to_quat_wxyz(aa: list[float] | np.ndarray) -> list[float]:
    """Convert axis-angle vector [ax, ay, az] (norm is radians) to canonical [w, x, y, z]."""
    v = np.asarray(aa, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]

    axis = v / angle
    sin_half = np.sin(angle / 2.0)
    w = np.cos(angle / 2.0)
    xyz = axis * sin_half

    return [float(x) for x in normalize_quaternion(np.array([w, xyz[0], xyz[1], xyz[2]]))]


def slerp(
    q0: list[float] | np.ndarray,
    q1: list[float] | np.ndarray,
    alpha: float,
) -> list[float]:
    """Spherical linear interpolation (SLERP) between two canonical [w, x, y, z] quaternions.

    Handles shortest antipodal path (alpha in [0, 1]).
    """
    v0 = normalize_quaternion(np.asarray(q0, dtype=np.float64))
    v1 = normalize_quaternion(np.asarray(q1, dtype=np.float64))

    dot = np.dot(v0, v1)

    # Antipodal shortest path
    if dot < 0.0:
        v1 = -v1
        dot = -dot

    # Close quaternions: linear interpolation to prevent division by zero
    if dot > 0.9995:
        result = v0 + alpha * (v1 - v0)
        return [float(x) for x in normalize_quaternion(result)]

    dot = np.clip(dot, -1.0, 1.0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * v0 + s1 * v1
    return [float(x) for x in normalize_quaternion(result)]
