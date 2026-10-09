"""Observation-only retrospective baselines; optional ML packages are imported locally."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction, runs


@dataclass
class MotionFeatures:
    matrix: np.ndarray
    names: list[str]
    observed: np.ndarray
    supported: np.ndarray
    speed: np.ndarray
    relative_acceleration: np.ndarray
    progressive_rate: np.ndarray
    signal_to_noise: np.ndarray
    fitted: np.ndarray
    noise: np.ndarray


def observation_features(case, diagnostics) -> MotionFeatures:
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    days = len(values)
    observed = np.isfinite(values).all(axis=1)
    differences = np.diff(values, axis=0)
    noise = np.maximum(1.4826 * np.nanmedian(np.abs(differences - np.nanmedian(differences, axis=0)),
                                           axis=0) / np.sqrt(2), 0.1)
    # Three disjoint 14-day blocks require movement on both sides. A single jump alone
    # cannot satisfy this raw-displacement check, even if a centered derivative rises early.
    medians = np.full((days, 3, 3), np.nan)
    for center in range(28, days - 28):
        for block, offset in enumerate((-21, 0, 21)):
            start = center + offset - 7
            sample = values[start:start + 14]
            if np.isfinite(sample).all(axis=1).sum() >= 8:
                medians[center, block] = np.nanmedian(sample, axis=0)
    before = medians[:, 1] - medians[:, 0]
    after = medians[:, 2] - medians[:, 1]
    lengths = np.column_stack((np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)))
    direction = np.sum(before * after, axis=1) / np.maximum(lengths.prod(axis=1), 1e-8)
    progressive_rate = np.min(lengths, axis=1) / 21
    progressive_rate[direction <= 0] = 0
    signal_to_noise = np.minimum(np.linalg.norm(before / noise, axis=1),
                                np.linalg.norm(after / noise, axis=1))
    columns, names = [], []
    for window in (31, 61, 91):
        motion = diagnostics[window]
        speed = np.asarray(motion.speed_mm_day, dtype=float)
        acceleration = np.asarray(motion.tangential_acceleration_mm_day2, dtype=float)
        columns.extend([speed, acceleration, acceleration / np.maximum(speed, 0.01)])
        names.extend([f"speed_{window}", f"acceleration_{window}", f"relative_acceleration_{window}"])
    motion = diagnostics[61]
    speed = np.asarray(motion.speed_mm_day, dtype=float)
    relative = np.asarray(motion.tangential_acceleration_mm_day2, dtype=float) / np.maximum(speed, 0.01)
    columns.extend([progressive_rate, signal_to_noise, direction])
    names.extend(["raw_progressive_rate", "raw_signal_to_noise", "raw_direction_consistency"])
    fitted = np.array([row if row is not None else [np.nan] * 3
                       for row in motion.fitted_displacement_mm])
    supported = observed & np.isfinite(speed) & np.isfinite(progressive_rate)
    return MotionFeatures(np.column_stack(columns), names, observed, supported, speed,
                          relative, progressive_rate, signal_to_noise, fitted, noise)


def activity_from_mask(mask, supported, minimum_days=14):
    activity = np.zeros(len(mask), dtype=int)
    activity[~supported] = -1
    for start, stop in runs(mask & supported):
        if stop - start >= minimum_days:
            activity[start:stop] = 1
    return activity


def rule_stages(features, activity, relative_change=0.5, gated=True):
    stage = np.full(len(activity), -1, dtype=int)
    if gated:
        stage[activity == 0] = 0
    eligible = features.supported & np.isfinite(features.relative_acceleration)
    if gated:
        eligible &= activity == 1
    stage[eligible] = 2
    change = 60 * features.relative_acceleration
    stage[eligible & (change > relative_change)] = 1
    stage[eligible & (change < -relative_change)] = 3
    return ActivityPrediction(activity.copy(), stage, gated=gated)


def robust_activity(features, minimum_rate, minimum_snr):
    mask = (features.progressive_rate >= minimum_rate) & (features.signal_to_noise >= minimum_snr)
    return activity_from_mask(mask, features.supported)


def pelt_segment_rates(features, penalty):
    import ruptures

    rates = np.full(len(features.speed), np.nan)
    # A separate response per axis avoids the erroneous N=response,E/U=covariates setup.
    valid = np.isfinite(features.fitted).all(axis=1)
    for start, stop in runs(valid):
        length = stop - start
        if length < 60:
            continue
        time = np.linspace(-1, 1, length)
        boundaries = {0, length}
        for axis in range(3):
            response = features.fitted[start:stop, axis] / features.noise[axis]
            signal = np.column_stack((response, np.ones(length), time))
            ends = ruptures.Pelt(model="linear", min_size=30, jump=7).fit(signal).predict(pen=penalty)
            boundaries.update(ends)
        edges = sorted(boundaries)
        for left, right in zip(edges, edges[1:]):
            if right - left < 14:
                continue
            design = np.column_stack((np.ones(right - left), np.arange(right - left)))
            coefficients = np.linalg.lstsq(design, features.fitted[start + left:start + right], rcond=None)[0]
            rates[start + left:start + right] = np.linalg.norm(coefficients[1])
    return rates


def pelt_activity(features, rates, minimum_rate):
    mask = (rates >= minimum_rate) & (features.progressive_rate >= minimum_rate / 2)
    mask &= features.signal_to_noise >= 1
    return activity_from_mask(mask, features.supported & np.isfinite(rates))
