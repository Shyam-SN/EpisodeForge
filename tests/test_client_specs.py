"""Tests for ClientSpec schema validation and configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from episodeforge.config.client_spec import ClientSpec, FormatType

CLIENT_SPEC_FILES = [
    "configs/clients/client_a_lerobot.v1.yaml",
    "configs/clients/client_a_lerobot.v2.yaml",
    "configs/clients/client_b_rlds.v1.yaml",
    "configs/clients/client_c_robomimic.v1.yaml",
    "configs/clients/client_d_egocentric_webdataset.v1.yaml",
]


@pytest.mark.parametrize("spec_file", CLIENT_SPEC_FILES)
def test_all_client_specs_valid(spec_file: str):
    path = Path(spec_file)
    assert path.exists(), f"Spec file {spec_file} does not exist"

    with open(path) as f:
        raw = yaml.safe_load(f)

    spec = ClientSpec.model_validate(raw)
    assert spec.spec_version.startswith("v")
    assert len(spec.dataset_name) > 0


def test_client_a_v1_properties():
    with open("configs/clients/client_a_lerobot.v1.yaml") as f:
        spec = ClientSpec.model_validate(yaml.safe_load(f))

    assert spec.client == "client_a"
    assert spec.format == FormatType.LEROBOT
    assert spec.timing.target_fps == 30.0
    assert "front" in spec.cameras
    assert "wrist" in spec.cameras
    assert "side" not in spec.cameras
    assert spec.action.representation.value == "joint_pos"


def test_client_a_v2_change_request_properties():
    with open("configs/clients/client_a_lerobot.v2.yaml") as f:
        spec = ClientSpec.model_validate(yaml.safe_load(f))

    assert spec.client == "client_a"
    assert spec.spec_version == "v2"
    assert spec.timing.target_fps == 15.0
    assert "side" in spec.cameras
    assert spec.action.representation.value == "ee_delta_pose"
    assert spec.action.rotation_format.value == "rot6d"


def test_client_b_rlds_properties():
    with open("configs/clients/client_b_rlds.v1.yaml") as f:
        spec = ClientSpec.model_validate(yaml.safe_load(f))

    assert spec.format == FormatType.RLDS
    assert spec.timing.target_fps == 10.0
    assert spec.cameras["front"].codec.value == "jpeg"
    assert spec.action.rotation_format.value == "euler_xyz"


def test_client_c_robomimic_properties():
    with open("configs/clients/client_c_robomimic.v1.yaml") as f:
        spec = ClientSpec.model_validate(yaml.safe_load(f))

    assert spec.format == FormatType.ROBOMIMIC_HDF5
    assert spec.cameras["front"].resolution == (84, 84)
    assert spec.selection.require_success is False  # Stores both success and failure demos


def test_invalid_split_ratios_rejected():
    with open("configs/clients/client_a_lerobot.v1.yaml") as f:
        raw = yaml.safe_load(f)

    raw["splits"]["train_ratio"] = 0.9
    raw["splits"]["val_ratio"] = 0.5  # sum = 1.4, invalid

    with pytest.raises(ValidationError):
        ClientSpec.model_validate(raw)
