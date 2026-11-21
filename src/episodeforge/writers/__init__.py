"""Format writers for LeRobot, RLDS, robomimic HDF5, and WebDataset."""

from __future__ import annotations

from pathlib import Path

from episodeforge.config.client_spec import ClientSpec
from episodeforge.transform.pipeline import TransformedEpisode
from episodeforge.writers.lerobot import write_lerobot_dataset
from episodeforge.writers.rlds import write_rlds_dataset
from episodeforge.writers.robomimic import write_robomimic_dataset
from episodeforge.writers.webdataset import write_webdataset_shards


def write_dataset_for_spec(
    episodes: list[TransformedEpisode],
    output_dir: Path | str,
    spec: ClientSpec,
) -> Path:
    """Dispatch to correct format writer based on spec.format."""
    fmt = spec.format.lower()
    if fmt == "lerobot":
        return write_lerobot_dataset(episodes, output_dir, spec)
    elif fmt in ("rlds", "open_x"):
        return write_rlds_dataset(episodes, output_dir, spec)
    elif fmt in ("robomimic", "hdf5", "robomimic_hdf5"):
        return write_robomimic_dataset(episodes, output_dir, spec)
    elif fmt in ("webdataset", "wds"):
        return write_webdataset_shards(episodes, output_dir, spec)
    else:
        raise ValueError(f"Unsupported client format: '{spec.format}'")


__all__ = [
    "write_dataset_for_spec",
    "write_lerobot_dataset",
    "write_rlds_dataset",
    "write_robomimic_dataset",
    "write_webdataset_shards",
]
