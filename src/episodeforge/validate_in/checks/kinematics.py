"""Kinematic limit, velocity, acceleration, and numeric integrity QA checks."""

from __future__ import annotations

import numpy as np

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.validate_in.models import CheckResult, CheckSeverity

# Franka Emika Panda standard kinematic limits
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

PANDA_VELOCITY_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
MAX_ACCEL_LIMIT = 75.0  # rad/s^2 (discrete 15 Hz control threshold)


def check_numeric_integrity(episode: CanonicalEpisode) -> CheckResult:
    """Verify that no NaN or infinite values exist in state or action trajectories."""
    has_nan_or_inf = False
    details: dict[str, int] = {}

    def scan_array(name: str, arr: list) -> None:
        nonlocal has_nan_or_inf
        if not arr:
            return
        np_arr = np.asarray(arr, dtype=float)
        invalid = np.isnan(np_arr) | np.isinf(np_arr)
        count = int(np.sum(invalid))
        if count > 0:
            has_nan_or_inf = True
            details[name] = count

    scan_array("joint_positions", episode.state.joint_positions)
    scan_array("joint_velocities", episode.state.joint_velocities)
    scan_array("ee_positions", episode.state.ee_positions)
    scan_array("ee_quaternions", episode.state.ee_quaternions)
    scan_array("gripper_widths", episode.state.gripper_widths)
    scan_array("action_joint_targets", episode.action.joint_position_targets)
    scan_array("action_ee_delta", episode.action.ee_delta_positions)
    scan_array("action_gripper", episode.action.gripper_targets)

    if has_nan_or_inf:
        return CheckResult(
            check_name="numeric_integrity",
            passed=False,
            severity=CheckSeverity.ERROR,
            message=f"Detected NaN or Inf values in streams: {details}",
            details=details,
        )

    return CheckResult(
        check_name="numeric_integrity",
        passed=True,
        severity=CheckSeverity.INFO,
        message="All numerical streams are finite and valid.",
    )


def check_kinematic_limits(episode: CanonicalEpisode) -> CheckResult:
    """Check whether joint positions stay within defined physical limits."""
    if not episode.state.joint_positions:
        return CheckResult(
            check_name="kinematic_limits",
            passed=True,
            severity=CheckSeverity.INFO,
            message="No joint positions present (human/phone session).",
        )

    qpos = np.asarray(episode.state.joint_positions, dtype=float)
    if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
        return CheckResult(
            check_name="kinematic_limits",
            passed=False,
            severity=CheckSeverity.ERROR,
            message="Cannot evaluate kinematic limits: NaN/Inf detected in joint positions.",
        )

    num_joints = min(qpos.shape[1], len(PANDA_JOINT_LIMITS))
    violations = []
    margin = 0.05  # 0.05 rad buffer

    for j in range(num_joints):
        q_min, q_max = PANDA_JOINT_LIMITS[j]
        actual_min = float(np.min(qpos[:, j]))
        actual_max = float(np.max(qpos[:, j]))

        if actual_min < (q_min - margin) or actual_max > (q_max + margin):
            violations.append(
                {
                    "joint": j,
                    "limit": (q_min, q_max),
                    "actual_range": (actual_min, actual_max),
                }
            )

    if violations:
        return CheckResult(
            check_name="kinematic_limits",
            passed=False,
            severity=CheckSeverity.ERROR,
            message=f"Joint limit violations detected on {len(violations)} joints",
            details={"violations": violations},
        )

    return CheckResult(
        check_name="kinematic_limits",
        passed=True,
        severity=CheckSeverity.INFO,
        message="All joint positions are within physical workspace limits.",
    )


def check_velocity_and_acceleration(episode: CanonicalEpisode) -> CheckResult:
    """Check for excessive joint velocities or sudden acceleration spikes."""
    if not episode.state.joint_positions or len(episode.state.joint_positions) < 3:
        return CheckResult(
            check_name="velocity_and_acceleration",
            passed=True,
            severity=CheckSeverity.INFO,
            message="Insufficient trajectory steps for velocity/acceleration check.",
        )

    qpos = np.asarray(episode.state.joint_positions, dtype=float)
    ts = np.asarray(episode.state.timestamps_ns, dtype=float) * 1e-9  # seconds

    if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
        return CheckResult(
            check_name="velocity_and_acceleration",
            passed=False,
            severity=CheckSeverity.ERROR,
            message="Cannot evaluate dynamics: NaN/Inf values in trajectory.",
        )

    dt = np.diff(ts)
    dt[dt <= 0] = 1e-6

    # Velocities
    qd = np.diff(qpos, axis=0) / dt[:, np.newaxis]
    max_vel = float(np.max(np.abs(qd)))

    # Accelerations
    dt_accel = dt[1:]
    qdd = np.diff(qd, axis=0) / dt_accel[:, np.newaxis]
    max_accel = float(np.max(np.abs(qdd)))

    details = {"max_velocity_rad_s": max_vel, "max_acceleration_rad_s2": max_accel}

    if max_accel > MAX_ACCEL_LIMIT:
        return CheckResult(
            check_name="velocity_and_acceleration",
            passed=False,
            severity=CheckSeverity.WARN,
            message=f"Acceleration spike detected: {max_accel:.2f} rad/s^2 > {MAX_ACCEL_LIMIT:.2f} limit",
            details=details,
        )

    return CheckResult(
        check_name="velocity_and_acceleration",
        passed=True,
        severity=CheckSeverity.INFO,
        message="Joint velocities and accelerations are within dynamic tolerances.",
        details=details,
    )
