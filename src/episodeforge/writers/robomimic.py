"""Writer for robomimic HDF5 dataset format."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from episodeforge.config.client_spec import ClientSpec
from episodeforge.encode.video import resize_image_array
from episodeforge.transform.pipeline import TransformedEpisode
from episodeforge.transform.splits import compute_deterministic_split


def write_robomimic_dataset(
    episodes: list[TransformedEpisode],
    output_path: Path | str,
    spec: ClientSpec,
) -> Path:
    """Serialize transformed episodes into standard robomimic HDF5 file structure."""
    out_file = Path(output_path).resolve()
    if out_file.suffix != ".hdf5":
        out_file = out_file / "data.hdf5"

    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not episodes:
        raise ValueError("Cannot write empty episode list to robomimic HDF5")

    total_samples = sum(ep.length for ep in episodes)
    demo_keys = [f"demo_{i}" for i in range(len(episodes))]

    # Generate train / val split masks
    splits = compute_deterministic_split(
        demo_keys,
        train_ratio=spec.splits.train_ratio,
        val_ratio=spec.splits.val_ratio,
        test_ratio=spec.splits.test_ratio,
    )

    with h5py.File(out_file, "w") as f:
        # Create root groups
        data_grp = f.create_group("data")
        mask_grp = f.create_group("mask")

        # Global attributes
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(
            {
                "env_name": "PandaTabletop",
                "type": 1,
                "env_kwargs": {
                    "has_renderer": False,
                    "reward_shaping": False,
                    "control_freq": spec.timing.target_fps,
                },
            }
        )

        for ep_idx, ep in enumerate(episodes):
            d_name = f"demo_{ep_idx}"
            demo_grp = data_grp.create_group(d_name)
            T = ep.length

            # Actions: (T, A) float32
            demo_grp.create_dataset(
                "actions",
                data=ep.actions.astype(np.float32),
                dtype="float32",
            )

            # Dones: 0 until terminal step (1)
            dones = np.zeros(T, dtype=np.int64)
            dones[-1] = 1
            demo_grp.create_dataset("dones", data=dones, dtype="int64")

            # Rewards: 0.0 intermediate, 1.0 at terminal if succeeded
            rewards = np.zeros(T, dtype=np.float32)
            if ep.success:
                rewards[-1] = 1.0
            demo_grp.create_dataset("rewards", data=rewards, dtype="float32")

            # States (concatenated joint states)
            if "joint_positions" in ep.states:
                demo_grp.create_dataset(
                    "states",
                    data=ep.states["joint_positions"].astype(np.float32),
                    dtype="float32",
                )
            else:
                demo_grp.create_dataset("states", data=np.zeros((T, 7), dtype=np.float32))

            # Obs group
            obs_grp = demo_grp.create_group("obs")
            for s_key, s_arr in ep.states.items():
                obs_grp.create_dataset(
                    s_key,
                    data=s_arr.astype(np.float32),
                    dtype="float32",
                )

            # Camera frames inside obs: (T, H, W, 3) uint8
            for cam_name, frame_paths in ep.camera_frames.items():
                cam_spec = spec.cameras.get(cam_name)
                w, h = cam_spec.resolution if cam_spec else (84, 84)
                target_key = cam_spec.target_key if cam_spec else f"{cam_name}_image"

                frames_cube = np.zeros((T, h, w, 3), dtype=np.uint8)
                for t, fp in enumerate(frame_paths):
                    with Image.open(fp) as img:
                        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    frames_cube[t] = resize_image_array(
                        arr, (w, h), mode=cam_spec.resize_mode if cam_spec else "center_crop"
                    )

                obs_grp.create_dataset(
                    target_key,
                    data=frames_cube,
                    dtype="uint8",
                )

            demo_grp.attrs["num_samples"] = T
            demo_grp.attrs["success"] = ep.success
            demo_grp.attrs["task"] = ep.task

        # Write masks
        mask_grp.create_dataset(
            "train",
            data=np.array([k.encode("utf-8") for k in splits["train"]]),
        )
        mask_grp.create_dataset(
            "valid",
            data=np.array([k.encode("utf-8") for k in splits["val"]]),
        )
        mask_grp.create_dataset(
            "test",
            data=np.array([k.encode("utf-8") for k in splits["test"]]),
        )

        successful_demos = [f"demo_{i}" for i, ep in enumerate(episodes) if ep.success]
        mask_grp.create_dataset(
            "successful",
            data=np.array([k.encode("utf-8") for k in successful_demos]),
        )

    return out_file
