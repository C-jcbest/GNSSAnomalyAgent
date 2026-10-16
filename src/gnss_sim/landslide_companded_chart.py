"""Noise-scaled chart companding that retains original GNSS sample slots."""

from __future__ import annotations

import warnings

import numpy as np


def raw_window_medians(case):
    """Return the existing disjoint three-day medians and a local raw anchor."""
    values = np.array(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm], dtype=float
    )
    complete = np.isfinite(values).all(axis=1)
    values[~complete] = np.nan
    padded = np.pad(values, ((91, 91), (0, 0)), constant_values=np.nan)
    centers = np.arange(len(values))[:, None, None] + 91
    indices = centers + np.arange(-90, 91, 3)[None, :, None] + np.array([-1, 0, 1])[None, None, :]
    samples = padded[indices]
    sample_complete = np.isfinite(samples).all(axis=-1)
    medians = np.full((len(values), 61, 3), np.nan)
    anchor = np.full((len(values), 1, 3), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians[:] = np.nanmedian(samples, axis=2)
        medians[sample_complete.sum(axis=2) < 2] = np.nan
        for center in np.flatnonzero(complete):
            sample = values[max(0, center - 7) : min(len(values), center + 7)]
            # The anchor sets the coordinate origin only. Its observation count
            # is not an activity threshold and cannot confer a definite label.
            anchor[center] = np.nanmedian(sample, axis=0)
    return medians, anchor


def companded_chart_patches(medians, anchor, noise, target_anchored=True, height=32):
    """Compress remote large excursions without removing a jump or filling gaps.

    The local anchor and noise proxy use observations only. The monotone asinh
    map is approximately linear near the anchor and logarithmic far away; the
    full transformed range remains visible. A globally anchored asinh control
    isolates target anchoring from generic nonlinear plotting.
    """
    medians = np.asarray(medians, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if medians.ndim != 3 or medians.shape[1:] != (61, 3):
        raise ValueError("Companded chart requires original 61x3 raw median slots")
    if anchor.shape != (len(medians), 1, 3):
        raise ValueError("Chart anchors must share the original query calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Companded chart requires three positive raw noise proxies")
    if height < 2:
        raise ValueError("Chart needs at least two rows")
    if not target_anchored:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            anchor = np.nanmedian(medians, axis=1, keepdims=True)
    transformed = np.arcsinh((medians - anchor) / (3 * noise[None, None, :]))
    available = np.isfinite(transformed)
    magnitude = np.max(np.where(available, np.abs(transformed), 0), axis=1)
    # Preserve the original six-noise-unit display floor in transformed units.
    half_span = np.maximum(magnitude, np.arcsinh(1.0))
    coordinates = (height - 1) * (0.5 - transformed / (2 * half_span[:, None, :]))
    coordinates = np.transpose(coordinates, (0, 2, 1))
    available = np.transpose(available, (0, 2, 1))
    distance = np.abs(
        np.arange(height)[None, None, :, None] - np.nan_to_num(coordinates)[:, :, None, :]
    )
    trace = np.maximum(1 - distance, 0) * available[:, :, None, :]
    mask = np.broadcast_to(available[:, :, None, :], trace.shape)
    return np.concatenate((trace, mask), axis=1).astype(np.float32)


def chart_ordinates(patches):
    """Recover plotted row positions for display diagnostics, not motion labels."""
    trace = patches[:, :3]
    mass = trace.sum(axis=2)
    positions = (trace * np.arange(trace.shape[2])[None, None, :, None]).sum(axis=2)
    positions = positions / np.maximum(mass, 1e-12)
    positions[patches[:, 3:, 0] == 0] = np.nan
    return positions
