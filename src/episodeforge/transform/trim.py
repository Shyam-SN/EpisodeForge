"""Automated trimming of idle start padding and trailing inactivity."""

from __future__ import annotations

import copy

import numpy as np

from episodeforge.canonical.schema import CanonicalEpisode


def trim_idle_padding(
    episode: CanonicalEpisode,
    velocity_threshold: float = 0.02,
    keep_padding_steps: int = 3,
) -> CanonicalEpisode:
    """Trim leading idle stillness and trailing inactivity from CanonicalEpisode.

    Trims only if meaningful movement occurs after an initial stationary phase.
    """
    if not episode.state.joint_velocities or len(episode.state.joint_velocities) < 10:
        return episode

    qvel = np.asarray(episode.state.joint_velocities, dtype=np.float64)
    speed = np.max(np.abs(qvel), axis=1)

    # Find first moving step
    moving_indices = np.where(speed > velocity_threshold)[0]
    if len(moving_indices) == 0:
        return episode  # Whole episode is still; do not trim

    start_idx = max(0, int(moving_indices[0]) - keep_padding_steps)
    end_idx = min(len(speed), int(moving_indices[-1]) + keep_padding_steps + 1)

    if start_idx == 0 and end_idx >= len(speed):
        return episode  # No trimming needed

    ep_trimmed = copy.deepcopy(episode)

    # Slice state streams
    ep_trimmed.state.timestamps_ns = ep_trimmed.state.timestamps_ns[start_idx:end_idx]
    ep_trimmed.state.joint_positions = ep_trimmed.state.joint_positions[start_idx:end_idx]
    ep_trimmed.state.joint_velocities = ep_trimmed.state.joint_velocities[start_idx:end_idx]
    ep_trimmed.state.ee_positions = ep_trimmed.state.ee_positions[start_idx:end_idx]
    ep_trimmed.state.ee_quaternions = ep_trimmed.state.ee_quaternions[start_idx:end_idx]
    ep_trimmed.state.gripper_widths = ep_trimmed.state.gripper_widths[start_idx:end_idx]

    # Slice action streams
    if ep_trimmed.action.timestamps_ns:
        ep_trimmed.action.timestamps_ns = ep_trimmed.action.timestamps_ns[start_idx:end_idx]
        if ep_trimmed.action.joint_position_targets:
            ep_trimmed.action.joint_position_targets = ep_trimmed.action.joint_position_targets[
                start_idx:end_idx
            ]
        if ep_trimmed.action.ee_delta_positions:
            ep_trimmed.action.ee_delta_positions = ep_trimmed.action.ee_delta_positions[
                start_idx:end_idx
            ]
        if ep_trimmed.action.gripper_targets:
            ep_trimmed.action.gripper_targets = ep_trimmed.action.gripper_targets[start_idx:end_idx]

    # Trim camera streams to match trimmed time span [t_start, t_end]
    if ep_trimmed.state.timestamps_ns and ep_trimmed.cameras:
        t0 = ep_trimmed.state.timestamps_ns[0]
        t1 = ep_trimmed.state.timestamps_ns[-1]

        for cam in ep_trimmed.cameras.values():
            if not cam.timestamps_ns:
                continue
            cam_ts = np.asarray(cam.timestamps_ns, dtype=np.int64)
            mask = (cam_ts >= t0) & (cam_ts <= t1)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                cam.timestamps_ns = [int(cam_ts[i]) for i in indices]
                cam.frame_count = len(indices)

    return ep_trimmed
