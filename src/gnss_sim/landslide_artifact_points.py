"""Observation-only daily artifact evidence, separate from deformation activity."""

from __future__ import annotations

import numpy as np


def artifact_point_features(values, noise):
    """Describe current NEU innovations without using centered velocity labels.

    Independent left/right raw blocks allow a point spike to differ from a
    permanent shift. Support requires the original current observation and
    at least four observations on each side. A detected artifact cannot veto
    coexisting continuous deformation.
    """
    values = np.asarray(values, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Artifact evidence requires the original NEU calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Artifact evidence requires three positive noise proxies")
    observed = np.isfinite(values).all(axis=1)
    matrix = np.full((len(values), 21), np.nan)
    supported = np.zeros(len(values), dtype=bool)
    for day in np.flatnonzero(observed):
        left = values[max(0, day - 7) : day]
        right = values[day + 1 : min(len(values), day + 8)]
        left = left[np.isfinite(left).all(axis=1)]
        right = right[np.isfinite(right).all(axis=1)]
        if len(left) < 4 or len(right) < 4:
            continue
        left_median = np.median(left, axis=0)
        right_median = np.median(right, axis=0)
        before = (values[day] - left_median) / noise
        after = (right_median - values[day]) / noise
        innovation_before = np.full(3, np.nan)
        innovation_after = np.full(3, np.nan)
        if day > 0 and observed[day - 1]:
            innovation_before = (values[day] - values[day - 1]) / noise
        if day + 1 < len(values) and observed[day + 1]:
            innovation_after = (values[day + 1] - values[day]) / noise
        residual = (values[day] - (left_median + right_median) / 2) / noise
        before_size = np.linalg.norm(before)
        after_size = np.linalg.norm(after)
        direction = float(np.dot(before, after) / max(before_size * after_size, 1e-12))
        matrix[day] = np.r_[
            innovation_before,
            innovation_after,
            before,
            after,
            residual,
            before_size,
            after_size,
            direction,
            np.linalg.norm(before + after),
            len(left),
            len(right),
        ]
        supported[day] = True
    return matrix, observed, supported
