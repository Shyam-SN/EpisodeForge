"""Streaming Welford's algorithm for online dataset mean, std, min, and max calculation."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class StreamStatistics(BaseModel):
    count: int = 0
    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    min: list[float] = Field(default_factory=list)
    max: list[float] = Field(default_factory=list)


class WelfordAccumulator:
    """Computes running mean, sample variance/std, min, and max in a single streaming pass."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.min_vals = np.full(dim, np.inf, dtype=np.float64)
        self.max_vals = np.full(dim, -np.inf, dtype=np.float64)

    def update_batch(self, batch: np.ndarray) -> None:
        """Update running statistics with a 2D batch of shape (T, dim)."""
        arr = np.asarray(batch, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[1] != self.dim:
            raise ValueError(f"Batch dimension {arr.shape[1]} does not match expected {self.dim}")

        for i in range(arr.shape[0]):
            x = arr[i]
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
            self.min_vals = np.minimum(self.min_vals, x)
            self.max_vals = np.maximum(self.max_vals, x)

    def finalize(self) -> StreamStatistics:
        """Compute final standard deviation and return StreamStatistics model."""
        if self.count < 2:
            std = np.zeros(self.dim, dtype=np.float64)
        else:
            var = self.M2 / (self.count - 1)
            var = np.maximum(0.0, var)
            std = np.sqrt(var)

        min_res = self.min_vals if self.count > 0 else np.zeros(self.dim)
        max_res = self.max_vals if self.count > 0 else np.zeros(self.dim)

        return StreamStatistics(
            count=self.count,
            mean=[float(x) for x in self.mean],
            std=[float(x) for x in std],
            min=[float(x) for x in min_res],
            max=[float(x) for x in max_res],
        )
