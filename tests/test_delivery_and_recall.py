"""Unit and integration tests for Delivery, S3/MinIO Transfer, Manifests, and Recall Playbook."""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
import yaml
from moto import mock_aws

from episodeforge.config.client_spec import ClientSpec
from episodeforge.deliver.manifest import (
    generate_delivery_manifest,
    verify_release_integrity,
)
from episodeforge.deliver.recall import DataRecallManager
from episodeforge.deliver.s3_client import S3DeliveryClient
from episodeforge.ingest.generated import ingest_generated_episode
from episodeforge.transform.pipeline import apply_client_transforms
from episodeforge.writers.lerobot import write_lerobot_dataset


@pytest.fixture
def sample_lerobot_release(tmp_path: Path):
    """Generate a tiny 3-episode LeRobot release for delivery testing."""
    raw_dir = Path("data/raw/m1_episodes")
    if not raw_dir.exists():
        pytest.skip("m1_episodes not generated yet")

    with open("configs/clients/client_a_lerobot.v1.yaml") as f:
        spec_data = yaml.safe_load(f)
    spec = ClientSpec.model_validate(spec_data)

    episodes = []
    for d in sorted(raw_dir.glob("ep_*"))[:3]:
        ep = ingest_generated_episode(d)
        ep.success = True
        t_ep = apply_client_transforms(ep, spec)
        if t_ep:
            episodes.append(t_ep)

    release_dir = tmp_path / "sample_release"
    write_lerobot_dataset(episodes, release_dir, spec)
    return release_dir, spec


def test_manifest_generation_and_integrity_verification(sample_lerobot_release):
    release_dir, spec = sample_lerobot_release

    manifest = generate_delivery_manifest(
        release_dir=release_dir,
        client=spec.client,
        dataset_name=spec.dataset_name,
        spec_version=spec.spec_version,
    )

    assert manifest["total_files"] > 0
    assert manifest["total_bytes"] > 0
    assert (release_dir / "delivery_manifest.json").exists()

    # Integrity verification must pass
    valid, errors = verify_release_integrity(release_dir)
    assert valid is True
    assert len(errors) == 0

    # Tamper with one file: verify that integrity check detects tampering
    info_file = release_dir / "meta" / "info.json"
    with open(info_file, "a") as f:
        f.write(" ")

    valid_tampered, errors_tampered = verify_release_integrity(release_dir)
    assert valid_tampered is False
    assert len(errors_tampered) > 0


def test_s3_delivery_client_with_moto(sample_lerobot_release):
    release_dir, spec = sample_lerobot_release

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-delivery-bucket")

        delivery_client = S3DeliveryClient(
            region_name="us-east-1",
            bucket_name="test-delivery-bucket",
        )

        receipt = delivery_client.deliver_release(
            release_dir=release_dir,
            client=spec.client,
            dataset_name=spec.dataset_name,
            spec_version=spec.spec_version,
        )

        assert receipt["verification_status"] == "VERIFIED"
        assert receipt["uploaded_files_count"] > 0
        assert (release_dir / "delivery_receipt.json").exists()

        # Check S3 object existence
        obj = client.get_object(
            Bucket="test-delivery-bucket",
            Key=f"deliveries/{spec.client}/{spec.dataset_name}/{spec.spec_version}/delivery_manifest.json",
        )
        assert len(obj["Body"].read()) > 0


def test_data_recall_playbook(sample_lerobot_release, tmp_path: Path):
    release_dir, spec = sample_lerobot_release

    # First generate initial manifest
    generate_delivery_manifest(
        release_dir=release_dir,
        client=spec.client,
        dataset_name=spec.dataset_name,
        spec_version=spec.spec_version,
    )

    recall_mgr = DataRecallManager(releases_root=tmp_path / "patched_releases")

    # Execute recall of episode_000001
    recalled_id = "episode_000001"
    patch_dir = recall_mgr.execute_recall(
        release_dir=release_dir,
        recalled_episode_ids=[recalled_id],
        reason="Visual artifact detected in wrist camera",
        patch_version="v1.0.1",
    )

    assert patch_dir.exists()
    assert (patch_dir / "recall_notice.json").exists()
    assert (patch_dir / "delivery_manifest.json").exists()

    with open(patch_dir / "recall_notice.json") as f:
        notice = json.load(f)
    assert notice["recalled_episodes_count"] == 1
    assert recalled_id in notice["recalled_episode_ids"]

    # Verify info.json in patch release has decremented episode count
    with open(patch_dir / "meta" / "info.json") as f:
        patch_info = json.load(f)
    with open(release_dir / "meta" / "info.json") as f:
        orig_info = json.load(f)

    assert patch_info["total_episodes"] == orig_info["total_episodes"] - 1

    # Verify integrity of the new patch release
    valid, errors = verify_release_integrity(patch_dir)
    assert valid is True
