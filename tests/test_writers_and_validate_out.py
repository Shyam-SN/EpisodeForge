"""Integration tests for Format Writers (LeRobot, RLDS, robomimic, WebDataset) and Isolated Validation."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import yaml

from episodeforge.config.client_spec import ClientSpec
from episodeforge.ingest.generated import ingest_generated_episode
from episodeforge.transform.pipeline import apply_client_transforms
from episodeforge.validate_out.engine import validate_output_dataset
from episodeforge.writers.lerobot import write_lerobot_dataset
from episodeforge.writers.rlds import write_rlds_dataset
from episodeforge.writers.robomimic import write_robomimic_dataset
from episodeforge.writers.webdataset import write_webdataset_shards


@pytest.fixture
def sample_transformed_episodes():
    """Load and transform sample episodes for writer tests."""
    ep_dirs = sorted(Path("data/raw/m1_episodes").glob("ep_*"))[:3]
    if not ep_dirs:
        pytest.skip("m1_episodes not generated yet")

    with open("configs/clients/client_a_lerobot.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec = ClientSpec.model_validate(spec_data)

    transformed = []
    for d in ep_dirs:
        ep = ingest_generated_episode(d)
        ep.success = True  # Ensure it passes selection
        t_ep = apply_client_transforms(ep, spec)
        if t_ep is not None:
            transformed.append(t_ep)

    assert len(transformed) > 0
    return transformed, spec


def test_write_and_validate_lerobot(tmp_path: Path, sample_transformed_episodes):
    transformed_eps, spec = sample_transformed_episodes
    out_dir = tmp_path / "lerobot_release"

    written_path = write_lerobot_dataset(transformed_eps, out_dir, spec)
    assert (written_path / "meta" / "info.json").exists()
    assert (written_path / "meta" / "stats.json").exists()
    assert len(list(written_path.glob("data/**/*.parquet"))) >= 1

    # Validate output
    val_res = validate_output_dataset(written_path, spec)
    assert val_res["status"] == "PASS"
    assert len(val_res["errors"]) == 0


def test_write_and_validate_robomimic(tmp_path: Path, sample_transformed_episodes):
    transformed_eps, _ = sample_transformed_episodes

    with open("configs/clients/client_c_robomimic.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec_c = ClientSpec.model_validate(spec_data)

    out_file = tmp_path / "robomimic_release" / "data.hdf5"
    written_path = write_robomimic_dataset(transformed_eps, out_file, spec_c)
    assert written_path.exists()

    # Check HDF5 structure directly
    with h5py.File(written_path, "r") as f:
        assert "data" in f
        assert "mask" in f
        assert "demo_0" in f["data"]
        assert "actions" in f["data"]["demo_0"]
        assert "obs" in f["data"]["demo_0"]

    # Validate output
    val_res = validate_output_dataset(written_path, spec_c)
    assert val_res["status"] == "PASS"
    assert len(val_res["errors"]) == 0


def test_write_and_validate_rlds(tmp_path: Path, sample_transformed_episodes):
    transformed_eps, _ = sample_transformed_episodes

    with open("configs/clients/client_b_rlds.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec_b = ClientSpec.model_validate(spec_data)

    out_dir = tmp_path / "rlds_release"
    written_path = write_rlds_dataset(transformed_eps, out_dir, spec_b)
    assert (written_path / "dataset_info.json").exists()
    assert (written_path / "features.json").exists()
    assert len(list(written_path.glob("*.tfrecord*"))) >= 1

    # Validate output
    val_res = validate_output_dataset(written_path, spec_b)
    assert val_res["status"] == "PASS"
    assert len(val_res["errors"]) == 0


def test_write_and_validate_webdataset(tmp_path: Path, sample_transformed_episodes):
    transformed_eps, _ = sample_transformed_episodes

    with open("configs/clients/client_d_egocentric_webdataset.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec_d = ClientSpec.model_validate(spec_data)

    out_dir = tmp_path / "wds_release"
    written_path = write_webdataset_shards(transformed_eps, out_dir, spec_d)
    tars = list(written_path.glob("*.tar"))
    assert len(tars) >= 1

    # Validate output
    val_res = validate_output_dataset(written_path, spec_d)
    assert val_res["status"] == "PASS"
    assert len(val_res["errors"]) == 0
