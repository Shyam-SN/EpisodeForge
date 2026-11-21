"""Tests for isolated format validators running as subprocesses."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_lerobot_validator_subprocess(tmp_path: Path):
    # Create mock invalid directory
    mock_dir = tmp_path / "mock_lerobot"
    mock_dir.mkdir()

    cmd = [sys.executable, "validators/lerobot/validate.py", str(mock_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1

    data = json.loads(proc.stdout)
    assert data["format"] == "lerobot"
    assert data["status"] == "FAIL"
    assert any("Missing 'meta/'" in err for err in data["errors"])


def test_rlds_validator_subprocess(tmp_path: Path):
    mock_dir = tmp_path / "mock_rlds"
    mock_dir.mkdir()

    cmd = [sys.executable, "validators/rlds/validate.py", str(mock_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1

    data = json.loads(proc.stdout)
    assert data["format"] == "rlds"
    assert data["status"] == "FAIL"


def test_robomimic_validator_subprocess(tmp_path: Path):
    mock_file = tmp_path / "nonexistent.hdf5"
    cmd = [sys.executable, "validators/robomimic/validate.py", str(mock_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1

    data = json.loads(proc.stdout)
    assert data["format"] == "robomimic_hdf5"
    assert data["status"] == "FAIL"
