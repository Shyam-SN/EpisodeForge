"""Output validation engine executing schema checks and isolated official loaders."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from episodeforge.config.client_spec import ClientSpec


def run_isolated_validator(
    format_name: str,
    dataset_path: Path | str,
) -> dict[str, Any]:
    """Invoke the isolated validator subprocess for format_name and parse JSON results."""
    p = Path(dataset_path).resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    validator_script = repo_root / "validators" / format_name / "validate.py"

    if not validator_script.exists():
        return {
            "format": format_name,
            "status": "WARN",
            "errors": [],
            "warnings": [f"No dedicated validator found at {validator_script}"],
        }

    cmd = [sys.executable, str(validator_script), str(p)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )
        if proc.stdout.strip():
            try:
                res = json.loads(proc.stdout.strip())
                if isinstance(res, dict):
                    return res
                return {"result": res}
            except json.JSONDecodeError:
                pass

        if proc.returncode != 0:
            return {
                "format": format_name,
                "status": "FAIL",
                "errors": [proc.stderr.strip() or "Validator process returned non-zero code"],
                "stdout": proc.stdout,
            }
        return {
            "format": format_name,
            "status": "PASS",
            "warnings": ["Validator output was not JSON but returned 0"],
        }
    except Exception as e:
        return {
            "format": format_name,
            "status": "FAIL",
            "errors": [f"Failed executing validator subprocess: {e}"],
        }


def validate_output_dataset(
    release_dir: Path | str,
    spec: ClientSpec,
) -> dict[str, Any]:
    """Perform structural schema checks and run the official validator."""
    rel_path = Path(release_dir).resolve()
    fmt = spec.format.lower()

    summary: dict[str, Any] = {
        "dataset_name": spec.dataset_name,
        "format": spec.format,
        "format_version": spec.format_version,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "checks": [],
    }

    # 1. Structural checks
    if fmt == "lerobot":
        if not (rel_path / "meta" / "info.json").exists():
            summary["errors"].append("Missing 'meta/info.json'")
        if not (rel_path / "meta" / "stats.json").exists():
            summary["errors"].append("Missing 'meta/stats.json'")
        data_parquets = list(rel_path.glob("data/**/*.parquet"))
        if not data_parquets:
            summary["errors"].append("No parquet data files found under 'data/'")

    elif fmt in ("rlds", "open_x"):
        info_files = list(rel_path.glob("**/dataset_info.json"))
        if not info_files:
            summary["errors"].append("Missing 'dataset_info.json'")
        tfrecords = list(rel_path.glob("**/*.tfrecord*"))
        if not tfrecords:
            summary["errors"].append("No TFRecord shards found")

    elif fmt in ("robomimic", "hdf5"):
        hdf5_files = (
            [rel_path]
            if rel_path.is_file() and rel_path.suffix == ".hdf5"
            else list(rel_path.glob("**/*.hdf5"))
        )
        if not hdf5_files:
            summary["errors"].append("No .hdf5 file found in release directory")

    elif fmt in ("webdataset", "wds"):
        tars = list(rel_path.glob("*.tar"))
        if not tars:
            summary["errors"].append("No .tar shard files found")

    # 2. Isolated validator invocation
    val_map = {
        "lerobot": "lerobot",
        "rlds": "rlds",
        "open_x": "rlds",
        "robomimic": "robomimic",
        "hdf5": "robomimic",
    }

    target_val = val_map.get(fmt)
    if target_val:
        # For robomimic, point directly to the HDF5 file if release_dir is a folder
        val_target = rel_path
        if target_val == "robomimic" and rel_path.is_dir():
            h5s = list(rel_path.glob("**/*.hdf5"))
            if h5s:
                val_target = h5s[0]

        validator_res = run_isolated_validator(target_val, val_target)
        summary["validator_result"] = validator_res

        if validator_res.get("status") == "FAIL":
            summary["status"] = "FAIL"
            summary["errors"].extend(validator_res.get("errors", []))
        summary["warnings"].extend(validator_res.get("warnings", []))

    if summary["errors"]:
        summary["status"] = "FAIL"

    return summary
