"""Writer for Hugging Face LeRobot dataset format (v3.0 standard with backward compat)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from episodeforge.config.client_spec import ClientSpec
from episodeforge.encode.video import encode_video_from_frames
from episodeforge.transform.pipeline import TransformedEpisode
from episodeforge.transform.stats import WelfordAccumulator


def write_lerobot_dataset(
    episodes: list[TransformedEpisode],
    output_dir: Path | str,
    spec: ClientSpec,
    chunk_size: int = 1000,
) -> Path:
    """Serialize transformed episodes into standard LeRobot v3.0 directory layout."""
    out_path = Path(output_dir).resolve()
    meta_dir = out_path / "meta"
    data_dir = out_path / "data" / "chunk-000"
    videos_dir = out_path / "videos" / "chunk-000"

    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not episodes:
        raise ValueError("Cannot write empty episode list to LeRobot dataset")

    total_frames = sum(ep.length for ep in episodes)
    total_episodes = len(episodes)

    # Unique tasks mapping
    unique_tasks = sorted({ep.task for ep in episodes})
    task_to_idx = {task_name: idx for idx, task_name in enumerate(unique_tasks)}

    # Track Welford stats across full dataset
    action_dim = episodes[0].actions.shape[1]
    action_welford = WelfordAccumulator(dim=action_dim)

    # Detect state dimension
    first_state_arr = next(iter(episodes[0].states.values())) if episodes[0].states else None
    state_dim = first_state_arr.shape[1] if first_state_arr is not None else 0
    state_welford = WelfordAccumulator(dim=state_dim) if state_dim > 0 else None

    episodes_meta = []
    global_frame_idx = 0

    for ep_idx, ep in enumerate(episodes):
        T = ep.length
        t_sec = [(ts - ep.timestamps_ns[0]) / 1e9 for ts in ep.timestamps_ns]

        # Update running stats
        action_welford.update_batch(ep.actions)
        if state_welford and "joint_positions" in ep.states:
            state_welford.update_batch(ep.states["joint_positions"])

        # Build trajectory dataframe for this episode
        traj_data: dict[str, list] = {
            "index": list(range(global_frame_idx, global_frame_idx + T)),
            "episode_index": [ep_idx] * T,
            "frame_index": list(range(T)),
            "timestamp": t_sec,
            "task_index": [task_to_idx[ep.task]] * T,
        }

        # Actions
        traj_data["action"] = [row.tolist() for row in ep.actions]

        # States
        for s_key, s_arr in ep.states.items():
            out_key = f"observation.state.{s_key}" if "." not in s_key else s_key
            if s_key == "joint_positions":
                out_key = "observation.state"
            traj_data[out_key] = [row.tolist() for row in s_arr]

        df = pd.DataFrame(traj_data)
        ep_file = data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(ep_file, index=False)

        # Encode camera video files
        for cam_name, frame_list in ep.camera_frames.items():
            cam_spec = spec.cameras.get(cam_name)
            target_key = cam_spec.target_key if cam_spec else f"observation.images.{cam_name}"
            # Ensure target_key is valid dir
            cam_video_dir = videos_dir / target_key
            cam_video_dir.mkdir(parents=True, exist_ok=True)
            video_file = cam_video_dir / f"episode_{ep_idx:06d}.mp4"

            if frame_list:
                encode_video_from_frames(
                    frame_paths=frame_list,
                    output_path=video_file,
                    fps=spec.timing.target_fps,
                    codec=cam_spec.codec if cam_spec else "av1",
                    crf=cam_spec.crf if cam_spec and cam_spec.crf else 28,
                    bitrate_kbps=cam_spec.bitrate_kbps if cam_spec else None,
                    target_res=cam_spec.resolution if cam_spec else None,
                    resize_mode=cam_spec.resize_mode if cam_spec else "center_crop",
                )

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [ep.task],
                "length": T,
                "success": ep.success,
            }
        )
        global_frame_idx += T

    # Write meta/info.json
    feature_spec: dict[str, dict] = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": ep.action_dim_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for _cam_name, cam_target in spec.cameras.items():
        w, h = cam_target.resolution
        feature_spec[cam_target.target_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": spec.timing.target_fps,
                "video.codec": cam_target.codec,
                "video.pix_fmt": cam_target.pixel_format,
            },
        }

    info_meta = {
        "codebase_version": "v3.0",
        "fps": spec.timing.target_fps,
        "robot_type": "franka_emika_panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "total_chunks": 1,
        "chunks_size": chunk_size,
        "features": feature_spec,
    }

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_meta, f, indent=2)

    # Write meta/stats.json
    stats_dict = {
        "action": action_welford.finalize().model_dump(),
    }
    if state_welford:
        stats_dict["observation.state"] = state_welford.finalize().model_dump()

    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    # Write meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for item in episodes_meta:
            f.write(json.dumps(item) + "\n")

    # Write meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task_name, idx in task_to_idx.items():
            f.write(json.dumps({"task_index": idx, "task": task_name}) + "\n")

    return out_path
