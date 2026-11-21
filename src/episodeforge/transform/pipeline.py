"""Unified transformation pipeline executing ClientSpec requirements on CanonicalEpisode."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from episodeforge.canonical.schema import CanonicalEpisode, QAVerdict
from episodeforge.config.client_spec import ClientSpec
from episodeforge.transform.actions import transform_actions
from episodeforge.transform.resample import (
    compute_target_timestamps,
    resample_continuous,
    resample_nearest_frames,
    resample_quaternions,
)
from episodeforge.transform.rotations import quat_wxyz_to_euler_xyz, quat_wxyz_to_xyzw
from episodeforge.transform.trim import trim_idle_padding


class TransformedEpisode:
    """The outcome of applying a ClientSpec to a CanonicalEpisode."""

    def __init__(
        self,
        episode_id: str,
        source_ref: str,
        task: str,
        instruction: str,
        success: bool,
        timestamps_ns: list[int],
        states: dict[str, np.ndarray],
        actions: np.ndarray,
        action_dim_names: list[str],
        camera_frames: dict[str, list[str]],
        metadata: dict[str, Any],
    ) -> None:
        self.episode_id = episode_id
        self.source_ref = source_ref
        self.task = task
        self.instruction = instruction
        self.success = success
        self.timestamps_ns = timestamps_ns
        self.states = states
        self.actions = actions
        self.action_dim_names = action_dim_names
        self.camera_frames = camera_frames
        self.metadata = metadata

    @property
    def length(self) -> int:
        return len(self.timestamps_ns)


def apply_client_transforms(
    episode: CanonicalEpisode,
    spec: ClientSpec,
) -> TransformedEpisode | None:
    """Transform CanonicalEpisode according to ClientSpec. Returns None if filtered out."""
    # 1. Selection / Filtering
    if spec.selection.require_success and not episode.success:
        return None

    if spec.selection.tasks and episode.task not in spec.selection.tasks:
        return None

    # Verdict hierarchy: PASS is 2, WARN is 1, FAIL is 0
    verdict_rank = {QAVerdict.PASS: 2, QAVerdict.WARN: 1, QAVerdict.FAIL: 0}
    required_rank = 2 if spec.selection.min_qa_verdict == "PASS" else 1
    if verdict_rank[episode.upstream_qa_verdict] < required_rank:
        return None

    # 2. Idle trimming
    ep = copy.deepcopy(episode)
    if spec.trimming.trim_idle_start or spec.trimming.trim_idle_end:
        ep = trim_idle_padding(
            ep,
            velocity_threshold=spec.trimming.velocity_threshold,
            keep_padding_steps=spec.trimming.padding_steps,
        )

    # 3. Resampling to target FPS
    target_fps = spec.timing.target_fps
    t_start = ep.state.timestamps_ns[0] if ep.state.timestamps_ns else 0
    t_end = ep.state.timestamps_ns[-1] if ep.state.timestamps_ns else 0
    target_ts = compute_target_timestamps(t_start, t_end, target_fps)

    # Resample states
    resampled_qpos = resample_continuous(
        ep.state.timestamps_ns, ep.state.joint_positions, target_ts
    )
    resampled_qvel = resample_continuous(
        ep.state.timestamps_ns, ep.state.joint_velocities, target_ts
    )
    resampled_ee_pos = resample_continuous(ep.state.timestamps_ns, ep.state.ee_positions, target_ts)
    resampled_ee_quat = resample_quaternions(
        ep.state.timestamps_ns, ep.state.ee_quaternions, target_ts
    )
    resampled_gripper = resample_continuous(
        ep.state.timestamps_ns, ep.state.gripper_widths, target_ts
    )

    # Resample actions
    resampled_action_qpos = resample_continuous(
        ep.action.timestamps_ns, ep.action.joint_position_targets, target_ts
    )
    resampled_action_ee_delta = resample_continuous(
        ep.action.timestamps_ns, ep.action.ee_delta_positions, target_ts
    )
    resampled_action_gripper = resample_continuous(
        ep.action.timestamps_ns, ep.action.gripper_targets, target_ts
    )

    # Assign resampled back to ep for action transformation
    ep.state.timestamps_ns = target_ts
    ep.state.joint_positions = [list(q) for q in resampled_qpos]  # type: ignore[arg-type]
    ep.state.joint_velocities = [list(qd) for qd in resampled_qvel]  # type: ignore[arg-type]
    ep.state.ee_positions = [list(p) for p in resampled_ee_pos]  # type: ignore[arg-type]
    ep.state.ee_quaternions = resampled_ee_quat
    ep.state.gripper_widths = [float(w) for w in resampled_gripper]  # type: ignore[arg-type]

    ep.action.timestamps_ns = target_ts
    ep.action.joint_position_targets = [list(q) for q in resampled_action_qpos]  # type: ignore[arg-type]
    ep.action.ee_delta_positions = [list(dp) for dp in resampled_action_ee_delta]  # type: ignore[arg-type]
    ep.action.gripper_targets = [float(w) for w in resampled_action_gripper]  # type: ignore[arg-type]

    # 4. Action vector assembly
    actions_mat, action_dim_names = transform_actions(ep, spec.action)

    # 5. State observation assembly
    states: dict[str, np.ndarray] = {}
    for obs_field in spec.state.include_fields:
        if obs_field in ("joint_positions", "qpos"):
            vals = np.asarray(ep.state.joint_positions, dtype=np.float32)
            if spec.state.units_angle == "deg":
                vals = np.rad2deg(vals)
            states[obs_field] = vals
        elif obs_field in ("joint_velocities", "qvel"):
            vals = np.asarray(ep.state.joint_velocities, dtype=np.float32)
            if spec.state.units_angle == "deg":
                vals = np.rad2deg(vals)
            states[obs_field] = vals
        elif obs_field in ("ee_positions", "ee_pos"):
            vals = np.asarray(ep.state.ee_positions, dtype=np.float32)
            if spec.state.units_length == "mm":
                vals = vals * 1000.0
            states[obs_field] = vals
        elif obs_field in ("ee_quaternions", "ee_quat"):
            if spec.action.rotation_format == "quat_xyzw":
                xyzw_quats = [quat_wxyz_to_xyzw(q) for q in ep.state.ee_quaternions]
                states[obs_field] = np.asarray(xyzw_quats, dtype=np.float32)
            else:
                states[obs_field] = np.asarray(ep.state.ee_quaternions, dtype=np.float32)
        elif obs_field in ("ee_euler", "euler"):
            euler_vals = [quat_wxyz_to_euler_xyz(q) for q in ep.state.ee_quaternions]
            if spec.state.units_angle == "deg":
                euler_vals = [list(np.rad2deg(a)) for a in euler_vals]
            states[obs_field] = np.asarray(euler_vals, dtype=np.float32)
        elif obs_field in ("gripper_width", "gripper_widths"):
            gw = np.asarray(ep.state.gripper_widths, dtype=np.float32)
            if spec.state.units_length == "mm":
                gw = gw * 1000.0
            states[obs_field] = gw[:, np.newaxis]

    # 6. Camera frames mapping
    camera_frames: dict[str, list[str]] = {}
    for cam_name, _cam_spec in spec.cameras.items():
        if cam_name in ep.cameras:
            cstream = ep.cameras[cam_name]
            if cstream.frames_dir:
                p = Path(cstream.frames_dir)
                all_frames = sorted([str(f) for f in p.glob("*.png")])
                if all_frames:
                    selected = resample_nearest_frames(
                        cstream.timestamps_ns,
                        all_frames,
                        target_ts,
                        max_error_ms=spec.timing.max_timestamp_error_ms,
                    )
                    camera_frames[cam_name] = [str(f) for f in selected]

    return TransformedEpisode(
        episode_id=ep.episode_id,
        source_ref=ep.source_ref,
        task=ep.task,
        instruction=ep.instruction,
        success=ep.success,
        timestamps_ns=target_ts,
        states=states,
        actions=actions_mat,
        action_dim_names=action_dim_names,
        camera_frames=camera_frames,
        metadata={
            "client": spec.client,
            "format": spec.format,
            "format_version": spec.format_version,
            "target_fps": target_fps,
            "original_length": len(episode.state.timestamps_ns),
            "transformed_length": len(target_ts),
        },
    )
