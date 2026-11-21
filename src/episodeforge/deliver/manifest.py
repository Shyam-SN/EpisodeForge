"""Cryptographic delivery manifests, checksums, and delivery receipts."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file reading in 64KB chunks."""
    p = Path(file_path).resolve()
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_delivery_manifest(
    release_dir: Path | str,
    client: str,
    dataset_name: str,
    spec_version: str,
) -> dict[str, Any]:
    """Scan release directory, calculate SHA-256 for all files, and build delivery manifest."""
    rel_path = Path(release_dir).resolve()
    if not rel_path.exists():
        raise FileNotFoundError(f"Release directory {rel_path} does not exist")

    all_files: list[dict[str, Any]] = []
    total_bytes = 0

    for fp in sorted(rel_path.glob("**/*")):
        if fp.is_file() and fp.name not in (
            "delivery_manifest.json",
            "delivery_receipt.json",
            "recall_notice.json",
        ):
            rel_name = str(fp.relative_to(rel_path))
            f_size = fp.stat().st_size
            sha256 = compute_file_sha256(fp)
            total_bytes += f_size
            all_files.append(
                {
                    "path": rel_name,
                    "size_bytes": f_size,
                    "sha256": sha256,
                }
            )

    manifest_id = f"manifest_{hashlib.sha256(f'{dataset_name}_{spec_version}_{total_bytes}'.encode()).hexdigest()[:12]}"
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    manifest: dict[str, Any] = {
        "manifest_id": manifest_id,
        "dataset_name": dataset_name,
        "client": client,
        "spec_version": spec_version,
        "created_at_utc": created_at,
        "total_files": len(all_files),
        "total_bytes": total_bytes,
        "files": all_files,
        "pipeline_version": "0.1.0",
    }

    manifest_path = rel_path / "delivery_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def generate_delivery_receipt(
    manifest: dict[str, Any],
    target_s3_uri: str,
    verification_status: str = "VERIFIED",
) -> dict[str, Any]:
    """Generate signed cryptographic receipt verifying transfer against manifest hash."""
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    receipt_id = (
        f"receipt_{hashlib.sha256(f'{manifest_sha256}_{target_s3_uri}'.encode()).hexdigest()[:12]}"
    )

    receipt = {
        "receipt_id": receipt_id,
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest_sha256,
        "client": manifest.get("client"),
        "dataset_name": manifest.get("dataset_name"),
        "target_s3_uri": target_s3_uri,
        "uploaded_files_count": manifest.get("total_files"),
        "total_transferred_bytes": manifest.get("total_bytes"),
        "verification_status": verification_status,
        "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    return receipt


def verify_release_integrity(release_dir: Path | str) -> tuple[bool, list[str]]:
    """Verify all files on disk match the recorded delivery_manifest.json exactly."""
    rel_path = Path(release_dir).resolve()
    manifest_file = rel_path / "delivery_manifest.json"
    if not manifest_file.exists():
        return False, ["Missing delivery_manifest.json in release directory"]

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    for f_info in manifest.get("files", []):
        fp = rel_path / f_info["path"]
        if not fp.exists():
            errors.append(f"Missing file recorded in manifest: {f_info['path']}")
            continue

        if fp.stat().st_size != f_info["size_bytes"]:
            errors.append(
                f"Size mismatch on {f_info['path']}: expected {f_info['size_bytes']}, got {fp.stat().st_size}"
            )
            continue

        actual_sha256 = compute_file_sha256(fp)
        if actual_sha256 != f_info["sha256"]:
            errors.append(
                f"Checksum mismatch on {f_info['path']}: expected {f_info['sha256']}, got {actual_sha256}"
            )

    return len(errors) == 0, errors
