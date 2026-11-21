"""Writer for WebDataset sharded tar format (Client D phone/egocentric data)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from episodeforge.config.client_spec import ClientSpec
from episodeforge.transform.pipeline import TransformedEpisode


def write_webdataset_shards(
    episodes: list[TransformedEpisode],
    output_dir: Path | str,
    spec: ClientSpec,
    max_shard_size: int = 100,
) -> Path:
    """Serialize episodes into WebDataset tarball shards."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not episodes:
        raise ValueError("Cannot write empty episode list to WebDataset")

    shard_idx = 0
    curr_tar: tarfile.TarFile | None = None
    episodes_in_shard = 0

    def open_shard(idx: int) -> tarfile.TarFile:
        shard_path = out_dir / f"shard-{idx:05d}.tar"
        return tarfile.open(shard_path, "w")

    curr_tar = open_shard(shard_idx)

    for ep in episodes:
        if episodes_in_shard >= max_shard_size:
            curr_tar.close()
            shard_idx += 1
            curr_tar = open_shard(shard_idx)
            episodes_in_shard = 0

        prefix = f"{ep.episode_id}"

        # 1. Text caption / instruction
        txt_bytes = ep.instruction.encode("utf-8")
        ti_txt = tarfile.TarInfo(name=f"{prefix}.txt")
        ti_txt.size = len(txt_bytes)
        curr_tar.addfile(ti_txt, io.BytesIO(txt_bytes))

        # 2. JSON metadata
        meta_dict = {
            "episode_id": ep.episode_id,
            "source_ref": ep.source_ref,
            "task": ep.task,
            "length": ep.length,
            "success": ep.success,
            "timestamps_ns": ep.timestamps_ns,
            "metadata": ep.metadata,
        }
        json_bytes = json.dumps(meta_dict, indent=2).encode("utf-8")
        ti_json = tarfile.TarInfo(name=f"{prefix}.json")
        ti_json.size = len(json_bytes)
        curr_tar.addfile(ti_json, io.BytesIO(json_bytes))

        # 3. Add video frames / video if present
        for cam_name, frame_paths in ep.camera_frames.items():
            if frame_paths:
                # Add sample representative keyframe or first frame
                first_fp = Path(frame_paths[0])
                if first_fp.exists():
                    with open(first_fp, "rb") as f_img:
                        img_bytes = f_img.read()
                    ti_img = tarfile.TarInfo(name=f"{prefix}_{cam_name}.png")
                    ti_img.size = len(img_bytes)
                    curr_tar.addfile(ti_img, io.BytesIO(img_bytes))

        episodes_in_shard += 1

    if curr_tar is not None:
        curr_tar.close()

    return out_dir
