"""A single raw-block jump can request one alternative visual observation."""

from __future__ import annotations

import warnings

import numpy as np


def jump_view_inputs(case, noise):
    """Build raw and jump-subtracted chart patches without changing observations.

    Only adjacent available three-day blocks can propose a jump. Subtract the
    largest observed NEU increment from later supported columns, preserving
    their subsequent differences. This is an alternative representation, never
    an instrument diagnosis, a corrected raw series, or an activity label.
    """
    values = np.array(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm],
        dtype=float,
    )
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Jump views require original NEU observations")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Jump views require three positive noise proxies")
    padded = np.pad(values, ((91, 91), (0, 0)), constant_values=np.nan)
    centers = np.arange(len(values))[:, None, None] + 91
    offsets = np.arange(-90, 91, 3)[None, :, None]
    samples = padded[centers + offsets + np.array([-1, 0, 1])[None, None, :]]
    complete = np.isfinite(samples).all(axis=-1)
    samples = np.where(complete[..., None], samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(samples, axis=2)
    medians[complete.sum(axis=2) < 2] = np.nan
    adjacent = np.isfinite(medians[:, :-1]).all(axis=-1) & np.isfinite(medians[:, 1:]).all(axis=-1)
    changes = np.diff(medians, axis=1)
    strength = np.linalg.norm(np.nan_to_num(changes / noise), axis=-1)
    strength[~adjacent] = 0
    largest_index = np.argmax(strength, axis=1)
    ranked = np.sort(strength, axis=1)
    largest, second = ranked[:, -1], ranked[:, -2]
    # Two strong opposing increments describe a spike or reversal, not one
    # dominant persistent offset. Unavailable adjacent blocks never qualify.
    dominant = (largest >= 6) & (largest > 2 * second)
    delta = changes[np.arange(len(values)), largest_index]
    delta = np.where((largest > 0)[:, None], delta, 0)
    after = np.arange(61)[None, :] > largest_index[:, None]
    alternative = medians - after[:, :, None] * delta[:, None, :]
    raw = rasterize_medians(medians, noise)
    fixed = rasterize_medians(alternative, noise)
    adaptive = np.where(dominant[:, None, None, None], fixed, raw)
    evidence = {
        "dominant": dominant,
        "largest_noise_units": largest,
        "second_noise_units": second,
        "jump_right_column": largest_index + 1,
        "adjacent_supported_pairs": adjacent.sum(axis=1),
    }
    return raw, adaptive, fixed, evidence


def rasterize_medians(medians, noise):
    """Render the same six-channel geometry as the existing raw-chart model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reference = np.nanmedian(medians, axis=1, keepdims=True)
    centered = medians - reference
    available = np.isfinite(centered)
    magnitude = np.max(np.where(available, np.abs(centered), 0), axis=1)
    span = np.maximum(2 * magnitude, 6 * noise[None, :])
    coordinates = np.transpose(31 * (0.5 - centered / span[:, None, :]), (0, 2, 1))
    available = np.transpose(available, (0, 2, 1))
    rows = np.arange(32)[None, None, :, None]
    distance = np.abs(rows - np.nan_to_num(coordinates)[:, :, None, :])
    trace = np.maximum(1 - distance, 0) * available[:, :, None, :]
    mask = np.broadcast_to(available[:, :, None, :], trace.shape)
    return np.concatenate((trace, mask), axis=1).astype(np.float32)
