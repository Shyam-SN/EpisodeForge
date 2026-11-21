#!/usr/bin/env python3
"""Isolated validator runner for robomimic HDF5 dataset format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def validate_robomimic(hdf5_path: Path) -> dict:
    result = {
        "format": "robomimic_hdf5",
        "hdf5_path": str(hdf5_path),
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "demo_count": 0,
        "total_samples": 0,
        "loader_used": "h5py_direct",
    }
    
    if not hdf5_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"HDF5 file does not exist: {hdf5_path}")
        return result
        
    try:
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                result["status"] = "FAIL"
                result["errors"].append("Missing root 'data' group in HDF5")
                return result
                
            data_group = f["data"]
            if "total" not in data_group.attrs:
                result["warnings"].append("Missing 'data.attrs[\"total\"]'")
            else:
                result["total_samples"] = int(data_group.attrs["total"])
                
            demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
            result["demo_count"] = len(demo_keys)
            if not demo_keys:
                result["status"] = "FAIL"
                result["errors"].append("No 'demo_*' groups found under 'data'")
                return result
                
            # Validate first demo structure
            first_demo = data_group[demo_keys[0]]
            for req_key in ["actions", "dones", "rewards", "obs"]:
                if req_key not in first_demo:
                    result["status"] = "FAIL"
                    result["errors"].append(f"Missing required key '{req_key}' in {demo_keys[0]}")
                    
            if "mask" in f:
                result["masks"] = list(f["mask"].keys())
            else:
                result["warnings"].append("Missing 'mask' group for train/val splits")
                
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Failed opening/reading HDF5 file: {e}")
        
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate robomimic HDF5 dataset output")
    parser.add_argument("hdf5_path", type=Path, help="Path to robomimic .hdf5 file")
    parser.add_argument("--out", type=Path, default=None, help="Output path for JSON results")
    args = parser.parse_args()
    
    res = validate_robomimic(args.hdf5_path)
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
