#!/usr/bin/env python3
"""Isolated validator runner for RLDS / Open X TFDS dataset format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def validate_rlds(dataset_dir: Path) -> dict:
    result = {
        "format": "rlds",
        "dataset_dir": str(dataset_dir),
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "checked_episodes": 0,
        "loader_used": "structural_fallback",
    }
    
    # 1. Structural check
    info_files = list(dataset_dir.glob("**/dataset_info.json"))
    if not info_files:
        result["status"] = "FAIL"
        result["errors"].append("Missing 'dataset_info.json' in dataset directory hierarchy")
        return result
        
    try:
        with open(info_files[0]) as f:
            info = json.load(f)
        result["dataset_name"] = info.get("name")
        result["version"] = info.get("version")
        result["splits"] = list(info.get("splits", {}).keys())
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Failed parsing dataset_info.json: {e}")
        return result
        
    # Check for TFRecord shards
    tfrecords = list(dataset_dir.glob("**/*.tfrecord*"))
    if not tfrecords:
        result["status"] = "FAIL"
        result["errors"].append("No .tfrecord shard files found")
        return result
    result["tfrecord_shards_count"] = len(tfrecords)

    # 2. Official loader check if available
    try:
        import tensorflow_datasets as tfds  # type: ignore
        builder = tfds.builder_from_directory(str(info_files[0].parent))
        ds = builder.as_dataset(split=result["splits"][0] if result["splits"] else "train")
        ep_count = 0
        for _ in ds.take(5):
            ep_count += 1
        result["checked_episodes"] = ep_count
        result["loader_used"] = "official_tfds"
    except ImportError:
        result["warnings"].append("Official 'tensorflow_datasets' package not installed; validated structurally.")
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Official TFDS loader raised exception: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate RLDS dataset output")
    parser.add_argument("dataset_dir", type=Path, help="Path to RLDS dataset directory")
    parser.add_argument("--out", type=Path, default=None, help="Output path for JSON results")
    args = parser.parse_args()
    
    res = validate_rlds(args.dataset_dir)
    json_output = json.dumps(res, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(json_output)
    else:
        print(json_output)
        
    if res["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()

