"""Original-calendar raw blocks for limited visual feature discovery."""

from __future__ import annotations

import warnings

import numpy as np


def discovery_block_matrix(values, noise):
    """Keep thirteen nonoverlapping 14-day blocks around each queried date.

    The evidence envelope is [t-91,t+91); block centers are t-84.5+14j.
    Each block needs eight complete NEU observations. Missing slots stay NaN,
    including record edges. No compacted calendar or gap interpolation is used.
    Subtracting each axis's first available median removes reference offsets;
    division by an existing daily-noise proxy expresses descriptive signal size.
    """
    values = np.asarray(values, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("Feature discovery requires an original NEU calendar")
    if noise.shape != (3,) or np.any(noise <= 0) or not np.isfinite(noise).all():
        raise ValueError("Feature discovery requires three positive noise proxies")
    padded = np.pad(values, ((91, 91), (0, 0)), constant_values=np.nan)
    indices = np.arange(len(values))[:, None, None] + np.arange(0, 182, 14)[None, :, None] + np.arange(14)[None, None, :]
    samples = padded[indices]
    complete = np.isfinite(samples).all(axis=-1)
    samples = np.where(complete[..., None], samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(samples, axis=2)
    available = complete.sum(axis=2) >= 8
    medians[~available] = np.nan
    first = np.argmax(available, axis=1)
    offsets = medians[np.arange(len(values)), first]
    normalized = (medians - offsets[:, None, :]) / noise
    return normalized, available, complete.sum(axis=2), medians
