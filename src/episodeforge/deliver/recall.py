"""Data recall and replacement playbook producing immutable semantic patch releases."""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from episodeforge.deliver.manifest import generate_delivery_manifest


class DataRecallManager:
    """Manages surgical recall of flawed episodes and publishes audited patch releases."""

    def __init__(self, releases_root: Path | str = "data/releases") -> None:
        self.releases_root = Path(releases_root).resolve()

    def execute_recall(
        self,
        release_dir: Path | str,
        recalled_episode_ids: list[str],
        reason: str,
        patch_version: str | None = None,
    ) -> Path:
        """Execute recall on an existing delivery, creating an audited immutable patch release."""
        orig_path = Path(release_dir).resolve()
        if not orig_path.exists():
            raise FileNotFoundError(f"Release directory {orig_path} does not exist")

        manifest_path = orig_path / "delivery_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing delivery_manifest.json in {orig_path}")

        with open(manifest_path, encoding="utf-8") as f:
            orig_manifest = json.load(f)

        client = str(orig_manifest.get("client", "unknown"))
        dataset_name = str(orig_manifest.get("dataset_name", "dataset"))
        orig_version = str(orig_manifest.get("spec_version", "v1"))

        new_patch_version = (
            patch_version
            if patch_version is not None
            else f"{orig_version}.recall_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d%H%M')}"
        )

        patch_dir: Path = self.releases_root / client / f"{dataset_name}_{new_patch_version}"
        patch_dir.mkdir(parents=True, exist_ok=True)

        # Detect format
        is_lerobot = (orig_path / "meta" / "info.json").exists()
        is_robomimic = list(orig_path.glob("**/*.hdf5")) or orig_path.suffix == ".hdf5"

        recalled_set = set(recalled_episode_ids)

        if is_lerobot:
            self._patch_lerobot(orig_path, patch_dir, recalled_set)
        elif is_robomimic:
            self._patch_robomimic(orig_path, patch_dir, recalled_set)
        else:
            # Generic directory copy excluding files matching recalled IDs
            self._patch_generic(orig_path, patch_dir, recalled_set)

        # Write recall notice
        recall_notice: dict[str, Any] = {
            "recall_id": f"recall_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d%H%M%S')}",
            "original_release": str(orig_path),
            "patch_release": str(patch_dir),
            "dataset_name": dataset_name,
            "client": client,
            "recalled_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "recalled_episodes_count": len(recalled_episode_ids),
            "recalled_episode_ids": recalled_episode_ids,
            "reason": reason,
            "remediation_status": "PATCH_ISSUED",
        }
        notice_path = patch_dir / "recall_notice.json"
        with open(notice_path, "w", encoding="utf-8") as f:
            json.dump(recall_notice, f, indent=2)

        # Generate fresh delivery manifest for the patch release
        generate_delivery_manifest(
            release_dir=patch_dir,
            client=client,
            dataset_name=dataset_name,
            spec_version=new_patch_version,
        )

        return patch_dir

    def _patch_lerobot(
        self,
        orig_dir: Path,
        patch_dir: Path,
        recalled_set: set[str],
    ) -> None:
        """Patch a LeRobot release directory by re-indexing without recalled episodes."""
        meta_src = orig_dir / "meta"
        meta_dst = patch_dir / "meta"
        meta_dst.mkdir(parents=True, exist_ok=True)

        # Copy stats and tasks
        if (meta_src / "stats.json").exists():
            shutil.copy2(meta_src / "stats.json", meta_dst / "stats.json")
        if (meta_src / "tasks.jsonl").exists():
            shutil.copy2(meta_src / "tasks.jsonl", meta_dst / "tasks.jsonl")

        # Filter episodes.jsonl
        episodes_jsonl = meta_src / "episodes.jsonl"
        kept_episodes = []
        if episodes_jsonl.exists():
            with open(episodes_jsonl, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    ep_idx_str = f"episode_{item['episode_index']:06d}"
                    ep_idx_short = f"ep_{item['episode_index']:06d}"
                    if (
                        str(item["episode_index"]) not in recalled_set
                        and ep_idx_str not in recalled_set
                        and ep_idx_short not in recalled_set
                    ):
                        kept_episodes.append(item)

        with open(meta_dst / "episodes.jsonl", "w", encoding="utf-8") as f:
            for item in kept_episodes:
                f.write(json.dumps(item) + "\n")

        # Copy data chunk parquets excluding recalled
        data_src = orig_dir / "data"
        data_dst = patch_dir / "data"
        if data_src.exists():
            for p_file in data_src.glob("**/*.parquet"):
                # Check if file name matches any recalled ID
                skip = any(rec_id in p_file.stem for rec_id in recalled_set)
                if not skip:
                    rel_sub = p_file.relative_to(data_src)
                    out_p = data_dst / rel_sub
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p_file, out_p)

        # Copy videos excluding recalled
        vid_src = orig_dir / "videos"
        vid_dst = patch_dir / "videos"
        if vid_src.exists():
            for v_file in vid_src.glob("**/*.mp4"):
                skip = any(rec_id in v_file.stem for rec_id in recalled_set)
                if not skip:
                    rel_sub = v_file.relative_to(vid_src)
                    out_v = vid_dst / rel_sub
                    out_v.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(v_file, out_v)

        # Update info.json
        info_json = meta_src / "info.json"
        if info_json.exists():
            with open(info_json, encoding="utf-8") as f:
                info_data = json.load(f)
            info_data["total_episodes"] = len(kept_episodes)
            info_data["total_frames"] = sum(e.get("length", 0) for e in kept_episodes)
            with open(meta_dst / "info.json", "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2)

    def _patch_robomimic(
        self,
        orig_dir: Path,
        patch_dir: Path,
        recalled_set: set[str],
    ) -> None:
        """Patch a robomimic HDF5 file by copying clean demos and updating masks."""
        import h5py

        h5_src = orig_dir if orig_dir.suffix == ".hdf5" else list(orig_dir.glob("**/*.hdf5"))[0]
        h5_dst = patch_dir / "data.hdf5"

        with (
            h5py.File(h5_src, "r") as src_f,
            h5py.File(h5_dst, "w") as dst_f,
        ):
            src_data = src_f["data"]
            dst_data = dst_f.create_group("data")

            total_frames = 0
            kept_demos = []

            for d_name in src_data.keys():
                skip = any(
                    rec_id in d_name or d_name == rec_id or d_name.replace("demo_", "ep_") == rec_id
                    for rec_id in recalled_set
                )
                if not skip:
                    src_f.copy(f"data/{d_name}", dst_data, name=d_name)
                    kept_demos.append(d_name)
                    total_frames += int(src_data[d_name].attrs.get("num_samples", 0))

            dst_data.attrs["total"] = total_frames
            if "env_args" in src_data.attrs:
                dst_data.attrs["env_args"] = src_data.attrs["env_args"]

            # Update masks
            if "mask" in src_f:
                src_mask = src_f["mask"]
                dst_mask = dst_f.create_group("mask")
                for m_key in src_mask.keys():
                    arr = src_mask[m_key][:]
                    filtered = [k for k in arr if k.decode("utf-8") not in recalled_set]
                    dst_mask.create_dataset(m_key, data=filtered)

    def _patch_generic(
        self,
        orig_dir: Path,
        patch_dir: Path,
        recalled_set: set[str],
    ) -> None:
        """Generic file copy excluding files containing recalled episode IDs."""
        for fp in orig_dir.glob("**/*"):
            if fp.is_file():
                skip = any(rec_id in fp.name for rec_id in recalled_set)
                if not skip:
                    rel_p = fp.relative_to(orig_dir)
                    dst_p = patch_dir / rel_p
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fp, dst_p)
