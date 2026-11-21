"""Multi-rate stream resampling: linear, SLERP, Zero-Order-Hold, and nearest-frame."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from episodeforge.transform.rotations import slerp


def compute_target_timestamps(
    t_start_ns: int,
    t_end_ns: int,
    target_fps: float,
) -> list[int]:
    """Generate monotonic integer nanosecond timestamps at target_fps."""
    if t_end_ns <= t_start_ns or target_fps <= 0:
        return [t_start_ns]

    dt_ns = int(round(1e9 / target_fps))
    timestamps = []
    curr = t_start_ns
    while curr <= t_end_ns:
        timestamps.append(curr)
        curr += dt_ns
    return timestamps


def resample_continuous(
    src_times_ns: list[int] | np.ndarray,
    src_values: list[list[float]] | list[float] | np.ndarray,
    target_times_ns: list[int] | np.ndarray,
) -> list[list[float]] | list[float]:
    """Resample continuous float values using piece-wise linear interpolation."""
    src_t = np.asarray(src_times_ns, dtype=np.float64)
    tgt_t = np.asarray(target_times_ns, dtype=np.float64)
    vals = np.asarray(src_values, dtype=np.float64)

    if len(src_t) == 0:
        return []
    if len(src_t) == 1:
        if vals.ndim == 1:
            return [float(vals[0])] * len(tgt_t)
        return [vals[0].tolist()] * len(tgt_t)

    is_1d = vals.ndim == 1
    if is_1d:
        res = np.interp(tgt_t, src_t, vals)
        return [float(x) for x in res]

    # Multi-dimensional (T, D)
    num_dim = vals.shape[1]
    res_matrix = np.zeros((len(tgt_t), num_dim), dtype=np.float64)
    for d in range(num_dim):
        res_matrix[:, d] = np.interp(tgt_t, src_t, vals[:, d])

    return [[float(x) for x in row] for row in res_matrix]


def resample_quaternions(
    src_times_ns: list[int] | np.ndarray,
    src_quats: list[list[float]] | np.ndarray,
    target_times_ns: list[int] | np.ndarray,
) -> list[list[float]]:
    """Resample canonical [w, x, y, z] quaternions using SLERP with shortest antipodal path."""
    src_t = np.asarray(src_times_ns, dtype=np.float64)
    tgt_t = np.asarray(target_times_ns, dtype=np.float64)
    quats = np.asarray(src_quats, dtype=np.float64)

    if len(src_t) == 0:
        return []
    if len(src_t) == 1:
        return [quats[0].tolist()] * len(tgt_t)

    out_quats: list[list[float]] = []

    for t in tgt_t:
        if t <= src_t[0]:
            out_quats.append(quats[0].tolist())
            continue
        if t >= src_t[-1]:
            out_quats.append(quats[-1].tolist())
            continue

        # Find interval [idx - 1, idx]
        idx = int(np.searchsorted(src_t, t))
        t0, t1 = src_t[idx - 1], src_t[idx]
        dt = t1 - t0
        alpha = float((t - t0) / dt) if dt > 0 else 0.0

        q_interp = slerp(quats[idx - 1], quats[idx], alpha)
        out_quats.append(q_interp)

    return out_quats


def resample_actions_zoh(
    src_times_ns: list[int] | np.ndarray,
    src_actions: list[list[float]] | list[float] | np.ndarray,
    target_times_ns: list[int] | np.ndarray,
) -> list[list[float]] | list[float]:
    """Resample action trajectory using Zero-Order Hold (ZOH).

    Each target timestamp adopts the latest action where t_action <= t_target.
    """
    src_t = np.asarray(src_times_ns, dtype=np.int64)
    tgt_t = np.asarray(target_times_ns, dtype=np.int64)
    actions = np.asarray(src_actions, dtype=np.float64)

    if len(src_t) == 0:
        return []
    if len(src_t) == 1:
        if actions.ndim == 1:
            return [float(actions[0])] * len(tgt_t)
        return [actions[0].tolist()] * len(tgt_t)

    is_1d = actions.ndim == 1
    out_actions: list[list[float]] | list[float] = []

    for t in tgt_t:
        idx = int(np.searchsorted(src_t, t, side="right")) - 1
        idx = max(0, min(idx, len(src_t) - 1))
        val = actions[idx]
        if is_1d:
            out_actions.append(float(val))  # type: ignore[arg-type]
        else:
            out_actions.append(val.tolist())  # type: ignore[arg-type]

    return out_actions


def resample_nearest_frames(
    src_times_ns: list[int] | np.ndarray,
    frame_paths: Sequence[str | Path],
    target_times_ns: list[int] | np.ndarray,
    max_error_ms: float = 100.0,
) -> list[str]:
    """Map each target timestamp to the nearest video frame path.

    Raises RuntimeError if nearest frame drift exceeds max_error_ms.
    """
    src_t = np.asarray(src_times_ns, dtype=np.float64)
    tgt_t = np.asarray(target_times_ns, dtype=np.float64)

    if len(src_t) != len(frame_paths):
        raise ValueError(f"Frame count {len(frame_paths)} != timestamp count {len(src_t)}")

    out_frames = []
    max_error_ns = max_error_ms * 1e6

    for t in tgt_t:
        idx = int(np.argmin(np.abs(src_t - t)))
        drift_ns = abs(src_t[idx] - t)
        if drift_ns > max_error_ns:
            raise RuntimeError(
                f"Resampling timestamp drift ({drift_ns / 1e6:.1f}ms) exceeds tolerance ({max_error_ms}ms) at target t={t / 1e9:.3f}s"
            )
        out_frames.append(str(frame_paths[idx]))

    return out_frames
