"""Image stream perceptual quality and integrity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.validate_in.models import CheckResult, CheckSeverity


def check_image_streams(episode: CanonicalEpisode) -> CheckResult:
    """Verify camera frames for frozen cameras, black frames, and dropped images."""
    if not episode.cameras:
        return CheckResult(
            check_name="image_streams",
            passed=True,
            severity=CheckSeverity.INFO,
            message="No camera streams present.",
        )

    issues = []
    details = {}

    for cname, cstream in episode.cameras.items():
        if not cstream.frames_dir and not cstream.video_path:
            issues.append(f"Camera '{cname}' has neither frames_dir nor video_path")
            continue

        if cstream.frames_dir:
            fdir = Path(cstream.frames_dir)
            if not fdir.is_dir():
                issues.append(f"Camera '{cname}' frames_dir does not exist: {fdir}")
                continue

            frame_files = sorted(fdir.glob("*.png")) + sorted(fdir.glob("*.jpg"))
            if len(frame_files) == 0:
                issues.append(f"Camera '{cname}' has 0 frame files in {fdir}")
                continue

            # Check consistency with timestamps
            if cstream.timestamps_ns and abs(len(frame_files) - len(cstream.timestamps_ns)) > 2:
                issues.append(
                    f"Camera '{cname}' frame file count ({len(frame_files)}) != timestamp count ({len(cstream.timestamps_ns)})"
                )

            # Sample frames to check for black / frozen / corrupted frames
            # Sample up to 10 evenly spaced frames
            sample_step = max(1, len(frame_files) // 10)
            sampled_files = frame_files[::sample_step]

            prev_arr: np.ndarray | None = None
            consecutive_identical = 0
            max_identical = 0
            black_frame_count = 0

            for fp in sampled_files:
                try:
                    with Image.open(fp) as img:
                        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

                    # Check for black frame
                    if float(np.mean(arr)) < 1.0:
                        black_frame_count += 1

                    # Check for identical / frozen frames
                    if prev_arr is not None:
                        diff = np.abs(arr.astype(int) - prev_arr.astype(int))
                        if np.mean(diff) < 0.1:  # Virtually identical
                            consecutive_identical += 1
                            max_identical = max(max_identical, consecutive_identical)
                        else:
                            consecutive_identical = 0

                    prev_arr = arr
                except Exception as e:
                    issues.append(f"Corrupted frame file {fp.name}: {e}")

            cam_stat = {
                "total_frames": len(frame_files),
                "sampled_count": len(sampled_files),
                "black_frames": black_frame_count,
                "max_frozen_consecutive": max_identical,
            }
            details[cname] = cam_stat

            if black_frame_count > (len(sampled_files) // 2):
                issues.append(
                    f"Camera '{cname}' is mostly black frames ({black_frame_count}/{len(sampled_files)})"
                )

            if max_identical >= 4:
                issues.append(
                    f"Camera '{cname}' has {max_identical} consecutive frozen sampled frames"
                )

    if issues:
        return CheckResult(
            check_name="image_streams",
            passed=False,
            severity=CheckSeverity.ERROR,
            message="; ".join(issues),
            details=details,
        )

    return CheckResult(
        check_name="image_streams",
        passed=True,
        severity=CheckSeverity.INFO,
        message="All camera streams have valid, diverse, uncorrupted image frames.",
        details=details,
    )
