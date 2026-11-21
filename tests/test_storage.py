"""Tests for S3/MinIO client interactions and delivery abstraction."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws


@pytest.fixture
def s3_client():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="robotics-data-lake")
        yield client


def test_s3_mock_upload_and_verify(s3_client):
    test_data = b"episodeforge-delivery-payload-sample"
    s3_client.put_object(
        Bucket="robotics-data-lake",
        Key="client_a/lerobot/v1/meta/info.json",
        Body=test_data,
    )

    obj = s3_client.get_object(
        Bucket="robotics-data-lake",
        Key="client_a/lerobot/v1/meta/info.json",
    )
    assert obj["Body"].read() == test_data
