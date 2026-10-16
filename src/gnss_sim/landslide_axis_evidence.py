"""Preserve component evidence before pooling heterogeneous GNSS axis noise."""

from __future__ import annotations

import numpy as np


def axis_persistence_evidence(values, noise, separations=(21, 42, 84)):
    """Compare the same disjoint raw blocks separately for each physical axis.

    A positive score requires two successive changes with the same sign on
    that axis. A sign reversal is negative; a flat side gives zero. This is
    an observation descriptor, not an activity label or a significance test.
    Missing blocks remain NaN. No interpolation or gap compression is used.
    """
    values = np.asarray(values, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Axis evidence requires original NEU observations")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Axis evidence requires three positive noise proxies")
    if any(type(span) is not int or span < 14 for span in separations):
        raise ValueError("Median blocks must be disjoint")
    medians = np.full_like(values, np.nan)
    for day in range(7, len(values) - 7):
        sample = values[day - 7 : day + 7]
        complete = sample[np.isfinite(sample).all(axis=1)]
        if len(complete) >= 8:
            medians[day] = np.median(complete, axis=0)
    scores = np.full((len(values), 3 * len(separations)), np.nan)
    for index, span in enumerate(separations):
        for day in range(span + 7, len(values) - span - 7):
            if not np.isfinite(values[day]).all():
                continue
            before = medians[day] - medians[day - span]
            after = medians[day + span] - medians[day]
            score = np.sign(before * after) * np.minimum(np.abs(before), np.abs(after)) / noise
            scores[day, 3 * index : 3 * index + 3] = score
    return scores


def pool_axis_evidence(scores):
    """Same new raw blocks, with component identity discarded for an ablation."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] % 3:
        raise ValueError("Scores must contain three axes per scale")
    shaped = scores.reshape(len(scores), -1, 3)
    positive = np.maximum(shaped, 0)
    return np.sqrt(np.sum(positive**2, axis=-1))
