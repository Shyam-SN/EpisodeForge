#!/usr/bin/env python3
"""Isolated validator runner for LeRobot dataset format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def validate_lerobot(dataset_dir: Path) -> dict:
    result = {
        "format": "lerobot",
        "dataset_dir": str(dataset_dir),
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "checked_episodes": 0,
        "total_frames": 0,
        "loader_used": "structural_fallback",
    }
    
    # 1. Structural checks
    meta_dir = dataset_dir / "meta"
    if not meta_dir.exists():
        result["status"] = "FAIL"
        result["errors"].append("Missing 'meta/' directory")
        return result
        
    info_json = meta_dir / "info.json"
    if not info_json.exists():
        result["status"] = "FAIL"
        result["errors"].append("Missing 'meta/info.json'")
        return result
        
    try:
        with open(info_json) as f:
            info = json.load(f)
        result["info"] = {
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "codebase_version": info.get("codebase_version"),
        }
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Failed parsing meta/info.json: {e}")
        return result
        
    # Check stats.json
    stats_json = meta_dir / "stats.json"
    if not stats_json.exists():
        result["warnings"].append("Missing 'meta/stats.json'")

    # 2. Attempt official loader if available
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        ds = LeRobotDataset(root=str(dataset_dir))
        result["checked_episodes"] = len(ds)
        result["loader_used"] = "official_lerobot"
    except ImportError:
        result["warnings"].append("Official 'lerobot' package not installed in environment; validated structurally.")
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Official LeRobotDataset loader raised exception: {e}")
        
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset output")
    parser.add_argument("dataset_dir", type=Path, help="Path to LeRobot dataset directory")
    parser.add_argument("--out", type=Path, default=None, help="Output path for JSON results")
    args = parser.parse_args()
    
    res = validate_lerobot(args.dataset_dir)
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

