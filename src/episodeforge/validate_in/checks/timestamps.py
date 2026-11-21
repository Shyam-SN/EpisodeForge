"""Timestamp monotonicity, sampling jitter, and cross-stream alignment checks."""

from __future__ import annotations

import numpy as np

from episodeforge.canonical.schema import CanonicalEpisode
from episodeforge.validate_in.models import CheckResult, CheckSeverity


def check_monotonic_timestamps(episode: CanonicalEpisode) -> CheckResult:
    """Verify that all stream timestamps are strictly monotonically increasing."""
    non_monotonic_streams = []

    def verify_stream(name: str, ts: list[int]) -> None:
        if len(ts) < 2:
            return
        arr = np.asarray(ts, dtype=np.int64)
        diffs = np.diff(arr)
        if np.any(diffs <= 0):
            bad_indices = int(np.sum(diffs <= 0))
            non_monotonic_streams.append((name, bad_indices))

    verify_stream("state", episode.state.timestamps_ns)
    verify_stream("action", episode.action.timestamps_ns)
    if episode.imu:
        verify_stream("imu", episode.imu.timestamps_ns)
    for cname, cstream in episode.cameras.items():
        verify_stream(f"camera_{cname}", cstream.timestamps_ns)

    if non_monotonic_streams:
        return CheckResult(
            check_name="monotonic_timestamps",
            passed=False,
            severity=CheckSeverity.ERROR,
            message=f"Non-monotonic timestamps detected in streams: {non_monotonic_streams}",
            details={"violations": non_monotonic_streams},
        )

    return CheckResult(
        check_name="monotonic_timestamps",
        passed=True,
        severity=CheckSeverity.INFO,
        message="All stream timestamps are strictly monotonically increasing.",
    )


def check_sampling_jitter(episode: CanonicalEpisode) -> CheckResult:
    """Calculate sampling jitter and verify no excessive dropped-sample gaps."""
    ts = episode.state.timestamps_ns
    if len(ts) < 5:
        # Check camera if state is empty (phone sessions)
        if episode.cameras:
            first_cam = next(iter(episode.cameras.values()))
            ts = first_cam.timestamps_ns

    if len(ts) < 5:
        return CheckResult(
            check_name="sampling_jitter",
            passed=True,
            severity=CheckSeverity.INFO,
            message="Stream too short to compute sampling jitter.",
        )

    dt_ms = np.diff(np.asarray(ts, dtype=np.float64)) / 1e6
    mean_dt = float(np.mean(dt_ms))
    std_dt = float(np.std(dt_ms))
    max_gap_ms = float(np.max(dt_ms))
    jitter_ratio = std_dt / (mean_dt + 1e-6)

    details = {
        "mean_dt_ms": mean_dt,
        "std_dt_ms": std_dt,
        "max_gap_ms": max_gap_ms,
        "jitter_ratio": jitter_ratio,
    }

    # Warn if jitter ratio > 0.25 or max gap > 3.0 * mean_dt
    if jitter_ratio > 0.25 or max_gap_ms > (3.5 * mean_dt):
        return CheckResult(
            check_name="sampling_jitter",
            passed=False,
            severity=CheckSeverity.WARN,
            message=f"High sampling jitter ({jitter_ratio:.2%}) or large gap ({max_gap_ms:.1f}ms > {3.5 * mean_dt:.1f}ms)",
            details=details,
        )

    return CheckResult(
        check_name="sampling_jitter",
        passed=True,
        severity=CheckSeverity.INFO,
        message=f"Sampling jitter is within bounds (jitter={jitter_ratio:.2%}, max_gap={max_gap_ms:.1f}ms).",
        details=details,
    )


def check_cross_stream_alignment(episode: CanonicalEpisode) -> CheckResult:
    """Check start and end synchronization across cameras and robot state streams."""
    if not episode.cameras or not episode.state.timestamps_ns:
        return CheckResult(
            check_name="cross_stream_alignment",
            passed=True,
            severity=CheckSeverity.INFO,
            message="Single stream or phone session; cross-stream check skipped.",
        )

    state_t0 = episode.state.timestamps_ns[0]
    state_t1 = episode.state.timestamps_ns[-1]

    align_offsets: dict[str, dict[str, float]] = {}
    exceeded = False

    for cname, cstream in episode.cameras.items():
        if not cstream.timestamps_ns:
            continue
        cam_t0 = cstream.timestamps_ns[0]
        cam_t1 = cstream.timestamps_ns[-1]

        start_offset_ms = abs(cam_t0 - state_t0) / 1e6
        end_offset_ms = abs(cam_t1 - state_t1) / 1e6

        align_offsets[cname] = {
            "start_offset_ms": start_offset_ms,
            "end_offset_ms": end_offset_ms,
        }

        # Allow up to 150 ms tolerance between camera capture start and control loop
        if start_offset_ms > 150.0 or end_offset_ms > 150.0:
            exceeded = True

    if exceeded:
        return CheckResult(
            check_name="cross_stream_alignment",
            passed=False,
            severity=CheckSeverity.WARN,
            message=f"Cross-stream temporal drift exceeds 150ms: {align_offsets}",
            details=align_offsets,
        )

    return CheckResult(
        check_name="cross_stream_alignment",
        passed=True,
        severity=CheckSeverity.INFO,
        message="All cameras and state streams are synchronized within 150ms.",
        details=align_offsets,
    )
