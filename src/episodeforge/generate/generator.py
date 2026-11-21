"""Core demonstration generation engine for MuJoCo tabletop manipulation."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd
from PIL import Image

from episodeforge.config.generation_config import GenerationConfig
from episodeforge.generate.ik import DLSInverseKinematics
from episodeforge.generate.scene import generate_tabletop_mjcf, sample_randomization
from episodeforge.generate.tasks import TaskStateMachine, TaskType


def compute_camera_intrinsics(fovy_deg: float, width: int, height: int) -> dict[str, Any]:
    fovy_rad = np.deg2rad(fovy_deg)
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy  # Square pixels
    cx = width / 2.0
    cy = height / 2.0
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": width,
        "height": height,
        "distortion_model": "pinhole",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def generate_single_episode(
    episode_idx: int,
    task_name: str,
    output_dir: Path,
    config: GenerationConfig,
    base_seed: int = 0,
    render_cameras: bool = True,
    save_gif: bool = False,
) -> dict[str, Any]:
    """Generate one complete demonstration episode package in MuJoCo."""
    seed = base_seed + episode_idx
    rng = np.random.RandomState(seed)
    task = TaskType(task_name)

    # Sample domain randomization
    rand = sample_randomization(seed=seed, task=task_name)

    # Determine deliberate imperfections
    imperfection_type: str | None = None
    rand_val = rng.rand()
    rates = config.imperfections
    if rand_val < rates.failed_grasp_rate:
        imperfection_type = "failed_grasp"
    elif rand_val < rates.failed_grasp_rate + rates.timeout_rate:
        imperfection_type = "timeout"
    elif rand_val < rates.failed_grasp_rate + rates.timeout_rate + rates.nan_glitch_rate:
        imperfection_type = "nan_glitch"
    elif (
        rand_val
        < rates.failed_grasp_rate
        + rates.timeout_rate
        + rates.nan_glitch_rate
        + rates.duplicate_episode_rate
    ):
        imperfection_type = "duplicate"
    elif (
        rand_val
        < rates.failed_grasp_rate
        + rates.timeout_rate
        + rates.nan_glitch_rate
        + rates.duplicate_episode_rate
        + rates.idle_padding_rate
    ):
        imperfection_type = "idle_padding"

    # Build MJCF & MuJoCo Model
    mjcf_str = generate_tabletop_mjcf(rand, task=task_name)
    model = mujoco.MjModel.from_xml_string(mjcf_str)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    ik_solver = DLSInverseKinematics(model, site_name="ee_site")
    # Initialize arm to standard Franka home posture
    q_home = np.array([0.0, 0.3, 0.0, -1.57, 0.0, 2.0, -0.785])
    for idx, q_val in zip(ik_solver.qpos_indices, q_home, strict=False):
        data.qpos[idx] = q_val
        ik_data.qpos[idx] = q_val
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, ik_data)

    state_machine = TaskStateMachine(task, rand, seed=seed, imperfection_type=imperfection_type)
    instruction = state_machine.sample_instruction()

    # Identifiers
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    cube_jnt_id = model.body_jntadr[cube_body_id]
    cube_qpos_adr = model.jnt_qposadr[cube_jnt_id]
    cube_dof_adr = model.jnt_dofadr[cube_jnt_id]
    ee_site_id = ik_solver.site_id

    # Timing & Frequencies
    dt_physics = model.opt.timestep  # 0.002s (500 Hz)
    control_rate_hz = config.control_rate_hz  # 15 Hz
    camera_fps = 30.0

    control_step_dt = 1.0 / control_rate_hz  # ~0.0667s (33 physics steps)
    physics_per_control = int(round(control_step_dt / dt_physics))
    render_step_dt = 1.0 / camera_fps  # ~0.0333s (16-17 physics steps)
    physics_per_render = int(round(render_step_dt / dt_physics))

    # Setup offscreen renderers if enabled
    cam_renderers: dict[str, mujoco.Renderer] = {}
    cam_resolutions = {
        "front": (640, 480),
        "wrist": (320, 240),
        "side": (640, 480),
    }
    if render_cameras:
        for cam_name, (w, h) in cam_resolutions.items():
            cam_renderers[cam_name] = mujoco.Renderer(model, height=h, width=w)

    # Trajectory logging buffers
    ctrl_timestamps_ns: list[int] = []
    joint_positions_log: list[list[float]] = []
    joint_velocities_log: list[list[float]] = []
    ee_positions_log: list[list[float]] = []
    ee_quaternions_log: list[list[float]] = []
    gripper_widths_log: list[float] = []

    action_qpos_targets_log: list[list[float]] = []
    action_ee_delta_log: list[list[float]] = []
    action_gripper_log: list[float] = []

    camera_frames: dict[str, list[np.ndarray]] = {"front": [], "wrist": [], "side": []}
    camera_timestamps_ns: dict[str, list[int]] = {"front": [], "wrist": [], "side": []}

    sim_time_ns = 0
    total_control_steps = 0
    max_control_steps = 150

    # Idle padding: stay still at beginning if injected
    if imperfection_type == "idle_padding":
        start_delay_steps = 20  # ~1.3 seconds idle
    else:
        start_delay_steps = 0

    done = False
    prev_ee_pos = np.copy(data.site_xpos[ee_site_id])
    held = False
    rel_offset = np.zeros(3)

    while not done and total_control_steps < max_control_steps:
        total_control_steps += 1
        current_cube_pos = np.array(data.qpos[cube_qpos_adr : cube_qpos_adr + 3])

        # Step motion primitive
        if total_control_steps <= start_delay_steps:
            # Idle
            target_ee = np.copy(prev_ee_pos)
            gripper_cmd = 0.0
            held = False
            done = False
        else:
            target_ee, gripper_cmd, held, done = state_machine.step_waypoint(current_cube_pos)
            if held and np.linalg.norm(rel_offset) == 0.0:
                rel_offset = current_cube_pos - data.site_xpos[ee_site_id]

        # Solve IK for arm targets using dedicated ik_data
        q_target = ik_solver.solve(ik_data, target_ee)

        # Action calculations
        current_ee_pos = np.array(data.site_xpos[ee_site_id])
        delta_ee_pos = target_ee - current_ee_pos

        # Record control-rate log
        ctrl_timestamps_ns.append(sim_time_ns)
        qpos_entry = [float(data.qpos[idx]) for idx in ik_solver.qpos_indices]
        qvel_entry = [float(data.qvel[idx]) for idx in ik_solver.dof_indices]
        action_target_entry = q_target.tolist() + [gripper_cmd]

        # Injected NaN glitch imperfection
        if imperfection_type == "nan_glitch" and total_control_steps == 15:
            qpos_entry[2] = float("nan")
            action_target_entry[2] = float("nan")

        joint_positions_log.append(qpos_entry)
        joint_velocities_log.append(qvel_entry)
        ee_positions_log.append(current_ee_pos.tolist())
        # End effector orientation (Hamilton wxyz)
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, data.site_xmat[ee_site_id])
        ee_quaternions_log.append(ee_quat.tolist())
        gripper_widths_log.append(
            float(
                data.qpos[ik_solver.qpos_indices[-1] + 1]
                + data.qpos[ik_solver.qpos_indices[-1] + 2]
            )
        )

        action_qpos_targets_log.append(action_target_entry)
        action_ee_delta_log.append(delta_ee_pos.tolist())
        action_gripper_log.append(gripper_cmd)

        # Apply actuators
        for i, q_val in enumerate(q_target):
            data.ctrl[i] = q_val
        data.ctrl[7] = gripper_cmd
        data.ctrl[8] = gripper_cmd

        # Sub-step physics and render cameras
        for step_i in range(physics_per_control):
            mujoco.mj_step(model, data)
            sim_time_ns += int(dt_physics * 1e9)

            if held:
                data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = (
                    data.site_xpos[ee_site_id] + rel_offset
                )
                data.qvel[cube_dof_adr : cube_dof_adr + 6] = 0

            # Camera rendering at 30 FPS
            if render_cameras and (step_i % physics_per_render == 0):
                for cam_name, renderer in cam_renderers.items():
                    renderer.update_scene(data, camera=cam_name)
                    frame = renderer.render()

                    # Injected glitch imperfection
                    if (
                        imperfection_type == "nan_glitch"
                        and total_control_steps == 15
                        and step_i == 0
                    ):
                        frame = np.copy(frame)
                        frame[100:150, 100:150] = 0  # Black square glitch

                    camera_frames[cam_name].append(frame)
                    camera_timestamps_ns[cam_name].append(sim_time_ns)

    # Check task success termination
    final_cube_pos = np.array(data.qpos[cube_qpos_adr : cube_qpos_adr + 3])
    target_pos = np.array(rand.target_pos)

    if task == TaskType.PICK_PLACE:
        dist_to_bowl = np.linalg.norm(final_cube_pos[:2] - target_pos[:2])
        success = bool(dist_to_bowl < 0.08 and final_cube_pos[2] > 0.40)
    elif task == TaskType.PUSH:
        dist_to_target = np.linalg.norm(final_cube_pos[:2] - target_pos[:2])
        success = bool(dist_to_target < 0.08)
    else:  # STACK
        dist_to_base = np.linalg.norm(final_cube_pos[:2] - target_pos[:2])
        success = bool(dist_to_base < 0.05 and final_cube_pos[2] > 0.44)

    # Injected failure overrides
    if imperfection_type in ("failed_grasp", "timeout"):
        success = False

    # Package serialization
    ep_str_id = f"ep_{episode_idx:06d}"
    ep_dir = output_dir / ep_str_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Save camera frames or video
    cam_meta = {}
    for cam_name, frames in camera_frames.items():
        cam_dir = ep_dir / "cameras" / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        w, h = cam_resolutions[cam_name]
        fovy = 48.0 if cam_name == "front" else (62.0 if cam_name == "wrist" else 52.0)

        # Save PNG frames
        for frame_idx, img_arr in enumerate(frames):
            frame_path = cam_dir / f"frame_{frame_idx:05d}.png"
            Image.fromarray(img_arr).save(frame_path)

        cam_meta[cam_name] = {
            "intrinsics": compute_camera_intrinsics(fovy, w, h),
            "frame_count": len(frames),
            "timestamps_ns": camera_timestamps_ns[cam_name],
            "frames_dir": str(cam_dir.relative_to(output_dir)),
        }

    # Save sample GIF if requested
    if save_gif and len(camera_frames["front"]) > 0:
        gif_path = output_dir / f"{task_name}_sample.gif"
        # Subsample frames for clean compact GIF
        gif_frames = [Image.fromarray(f) for f in camera_frames["front"][::2]]
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=66,
            loop=0,
        )

    # Save trajectory dataframe (Parquet)
    traj_df = pd.DataFrame(
        {
            "timestamp_ns": ctrl_timestamps_ns,
            "joint_positions": joint_positions_log,
            "joint_velocities": joint_velocities_log,
            "ee_positions": ee_positions_log,
            "ee_quaternions": ee_quaternions_log,
            "gripper_widths": gripper_widths_log,
            "action_joint_targets": action_qpos_targets_log,
            "action_ee_delta": action_ee_delta_log,
            "action_gripper": action_gripper_log,
        }
    )
    traj_df.to_parquet(ep_dir / "trajectory.parquet")

    # Save metadata
    meta = {
        "episode_id": ep_str_id,
        "task": task_name,
        "instruction": instruction,
        "success": success,
        "seed": seed,
        "control_steps": total_control_steps,
        "duration_sec": float(sim_time_ns / 1e9),
        "randomization": dataclasses.asdict(rand),
        "cameras": cam_meta,
    }
    with open(ep_dir / "episode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Ground truth QA labels for validation recall testing
    qa_label = {
        "episode_id": ep_str_id,
        "injected_imperfection": imperfection_type is not None,
        "defect_type": imperfection_type,
        "expected_qa_verdict": "FAIL"
        if imperfection_type in ("failed_grasp", "nan_glitch", "timeout")
        else ("WARN" if imperfection_type == "idle_padding" else "PASS"),
        "success": success,
    }
    with open(ep_dir / "ground_truth_qa.json", "w") as f:
        json.dump(qa_label, f, indent=2)

    return {
        "episode_id": ep_str_id,
        "task": task_name,
        "success": success,
        "imperfection": imperfection_type,
        "steps": total_control_steps,
    }


def generate_dataset_parallel(
    config: GenerationConfig,
    output_dir: Path,
    num_episodes: int = 100,
    workers: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate demonstration dataset in parallel across CPU worker processes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = config.tasks
    task_allocations = [tasks[i % len(tasks)] for i in range(num_episodes)]

    results = []
    print(f"Starting parallel generation: {num_episodes} episodes across {workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, task_name in enumerate(task_allocations):
            save_gif = idx < len(tasks)  # Save sample GIF for first of each task
            f = executor.submit(
                generate_single_episode,
                episode_idx=idx,
                task_name=task_name,
                output_dir=output_dir,
                config=config,
                base_seed=seed,
                render_cameras=True,
                save_gif=save_gif,
            )
            futures[f] = (idx, task_name)

        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            results.append(res)
            if len(results) % 10 == 0 or len(results) == num_episodes:
                print(f"Progress: {len(results)}/{num_episodes} episodes generated...")

    elapsed = time.time() - t0
    episodes_per_hour = (num_episodes / elapsed) * 3600.0

    successes = sum(1 for r in results if r["success"])
    success_rate = (successes / num_episodes) * 100.0

    # Save combined ground-truth manifest
    ground_truth_all = {}
    for r in results:
        ep_id = r["episode_id"]
        with open(output_dir / ep_id / "ground_truth_qa.json") as fp:
            ground_truth_all[ep_id] = json.load(fp)

    with open(output_dir / "ground_truth_manifest.json", "w") as fp:
        json.dump(ground_truth_all, fp, indent=2)

    summary = {
        "total_episodes": num_episodes,
        "success_rate_pct": float(round(success_rate, 2)),
        "wall_time_sec": float(round(elapsed, 2)),
        "episodes_per_hour": float(round(episodes_per_hour, 1)),
        "workers": workers,
    }
    with open(output_dir / "generation_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    return summary
