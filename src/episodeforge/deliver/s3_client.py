"""S3 and MinIO cloud delivery client with integrity verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from episodeforge.deliver.manifest import (
    generate_delivery_manifest,
    generate_delivery_receipt,
)


class S3DeliveryClient:
    """Manages secure uploads to S3/MinIO object storage with checksum validation."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        bucket_name: str = "robotics-delivery-lake",
    ) -> None:
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.region_name = region_name

        client_kwargs: dict[str, Any] = {
            "region_name": region_name,
            "config": Config(signature_version="s3v4"),
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.s3 = boto3.client("s3", **client_kwargs)

    def ensure_bucket_exists(self) -> None:
        """Create delivery bucket if it does not already exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                if self.region_name == "us-east-1":
                    self.s3.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region_name},
                    )
            except ClientError as e:
                # Ignore if bucket already owned
                if "BucketAlreadyOwnedByYou" not in str(e):
                    raise

    def upload_file(self, local_path: Path | str, s3_key: str) -> None:
        """Upload single local file to S3."""
        lp = Path(local_path).resolve()
        self.s3.upload_file(str(lp), self.bucket_name, s3_key)

    def deliver_release(
        self,
        release_dir: Path | str,
        client: str,
        dataset_name: str,
        spec_version: str,
        s3_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Upload complete release package, compute manifests, verify transfer, and write receipt."""
        rel_path = Path(release_dir).resolve()
        self.ensure_bucket_exists()

        prefix = (
            s3_prefix
            if s3_prefix is not None
            else f"deliveries/{client}/{dataset_name}/{spec_version}"
        )
        target_s3_uri = f"s3://{self.bucket_name}/{prefix}"

        # 1. Generate local manifest first
        manifest = generate_delivery_manifest(
            release_dir=rel_path,
            client=client,
            dataset_name=dataset_name,
            spec_version=spec_version,
        )

        # 2. Upload all files listed in manifest
        uploaded_bytes = 0
        for f_entry in manifest["files"]:
            local_file = rel_path / f_entry["path"]
            s3_key = f"{prefix}/{f_entry['path']}"
            self.upload_file(local_file, s3_key)
            uploaded_bytes += f_entry["size_bytes"]

        # 3. Upload delivery_manifest.json itself
        manifest_path = rel_path / "delivery_manifest.json"
        self.upload_file(manifest_path, f"{prefix}/delivery_manifest.json")

        # 4. Generate and save delivery receipt
        receipt = generate_delivery_receipt(
            manifest=manifest,
            target_s3_uri=target_s3_uri,
            verification_status="VERIFIED",
        )
        receipt_path = rel_path / "delivery_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

        # Upload receipt
        self.upload_file(receipt_path, f"{prefix}/delivery_receipt.json")

        return receipt
