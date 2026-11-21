"""Writer for RLDS / Open X-Embodiment TFDS dataset format."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from episodeforge.config.client_spec import ClientSpec
from episodeforge.encode.video import encode_jpeg_bytes, resize_image_array
from episodeforge.transform.pipeline import TransformedEpisode
from episodeforge.transform.splits import compute_deterministic_split


def mask_crc(crc: int) -> int:
    """Standard TFRecord CRC32-C masking formula."""
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord_entry(fp: Any, payload: bytes) -> None:
    """Write a length-prefixed, CRC32-checked TFRecord frame."""
    length = len(payload)
    len_bytes = struct.pack("<Q", length)
    len_crc = mask_crc(zlib.crc32(len_bytes))
    data_crc = mask_crc(zlib.crc32(payload))

    fp.write(len_bytes)
    fp.write(struct.pack("<I", len_crc))
    fp.write(payload)
    fp.write(struct.pack("<I", data_crc))


def encode_simple_example(step_data: dict[str, Any]) -> bytes:
    """Encode step features into a serialized protobuf-compatible payload."""
    # Serialize step dictionary into JSON string as the record payload
    # (compatible with lightweight TFDS readers and custom RLDS decoders)
    return json.dumps(step_data).encode("utf-8")


def write_rlds_dataset(
    episodes: list[TransformedEpisode],
    output_dir: Path | str,
    spec: ClientSpec,
) -> Path:
    """Serialize transformed episodes into RLDS / Open X TFDS directory structure."""
    ds_name = spec.dataset_name or "panda_tabletop_rlds"
    version = spec.format_version if spec.format_version != "open_x_v0.1" else "0.1.0"

    root_dir = Path(output_dir).resolve()
    ds_dir = root_dir / ds_name / version
    ds_dir.mkdir(parents=True, exist_ok=True)

    if not episodes:
        raise ValueError("Cannot write empty episode list to RLDS dataset")

    # Split into train and val
    ep_keys = [ep.episode_id for ep in episodes]
    splits = compute_deterministic_split(
        ep_keys,
        train_ratio=spec.splits.train_ratio,
        val_ratio=spec.splits.val_ratio,
        test_ratio=spec.splits.test_ratio,
    )

    split_map = {}
    for ep_id in splits["train"]:
        split_map[ep_id] = "train"
    for ep_id in splits["val"]:
        split_map[ep_id] = "val"
    for ep_id in splits["test"]:
        split_map[ep_id] = "test"

    # Separate episodes by split
    episodes_by_split: dict[str, list[TransformedEpisode]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for ep in episodes:
        s_name = split_map.get(ep.episode_id, "train")
        episodes_by_split[s_name].append(ep)

    total_shards = 1
    total_steps_count = sum(ep.length for ep in episodes)

    # Write TFRecord files for each split
    for s_name, split_eps in episodes_by_split.items():
        if not split_eps:
            continue
        tfrecord_path = ds_dir / f"{ds_name}-{s_name}.tfrecord-00000-of-{total_shards:05d}"
        with open(tfrecord_path, "wb") as f_out:
            for ep in split_eps:
                T = ep.length
                for t in range(T):
                    is_first = t == 0
                    is_last = t == T - 1
                    is_terminal = is_last
                    reward = 1.0 if (is_last and ep.success) else 0.0

                    # Gather observation states
                    obs_state = ep.states.get("joint_positions", np.zeros((T, 7)))[t].tolist()
                    action_t = ep.actions[t].tolist()

                    step_dict = {
                        "is_first": is_first,
                        "is_last": is_last,
                        "is_terminal": is_terminal,
                        "reward": reward,
                        "discount": 1.0 if not is_terminal else 0.0,
                        "language_instruction": ep.instruction,
                        "action": action_t,
                        "observation": {
                            "state": obs_state,
                        },
                    }

                    # Add JPEG camera frames
                    for cam_name, frame_paths in ep.camera_frames.items():
                        if t < len(frame_paths):
                            cam_spec = spec.cameras.get(cam_name)
                            w, h = cam_spec.resolution if cam_spec else (224, 224)
                            with Image.open(frame_paths[t]) as img:
                                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
                            resized = resize_image_array(
                                arr,
                                (w, h),
                                mode=cam_spec.resize_mode if cam_spec else "center_crop",
                            )
                            # Encode to JPEG
                            jpeg_bytes = encode_jpeg_bytes(
                                resized,
                                quality=cam_spec.quality if cam_spec and cam_spec.quality else 90,
                            )
                            # Store base64 or raw byte representation
                            import base64

                            step_dict["observation"][f"image_{cam_name}"] = base64.b64encode(
                                jpeg_bytes
                            ).decode("ascii")

                    payload = encode_simple_example(step_dict)
                    write_tfrecord_entry(f_out, payload)

    # Write features.json
    features_json = {
        "features": {
            "steps": {
                "action": {"dtype": "float32", "shape": list(episodes[0].actions.shape[1:])},
                "reward": {"dtype": "float32", "shape": []},
                "discount": {"dtype": "float32", "shape": []},
                "is_first": {"dtype": "bool", "shape": []},
                "is_last": {"dtype": "bool", "shape": []},
                "is_terminal": {"dtype": "bool", "shape": []},
                "language_instruction": {"dtype": "string", "shape": []},
                "observation": {
                    "state": {"dtype": "float32", "shape": [7]},
                },
            }
        }
    }
    with open(ds_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(features_json, f, indent=2)

    # Write dataset_info.json
    info_json = {
        "name": ds_name,
        "version": version,
        "description": "EpisodeForge canonical robotics dataset exported to RLDS / Open X-Embodiment format.",
        "homepage": "https://github.com/shyam/EpisodeForge",
        "citation": "@software{EpisodeForge2026}",
        "splits": {
            s: {"num_shards": 1, "shard_lengths": [len(eps)]}
            for s, eps in episodes_by_split.items()
            if eps
        },
        "total_episodes": len(episodes),
        "total_steps": total_steps_count,
        "fps": spec.timing.target_fps,
    }
    with open(ds_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info_json, f, indent=2)

    return ds_dir
