"""Frozen, independent P5 numerical baselines on signed daily displacement."""

from __future__ import annotations

import numpy as np
import ruptures as rpt
import stumpy
from scipy.stats import theilslopes

DAYS = 365
WINDOW = 30
CENTER = WINDOW // 2
SPECTRAL_EPSILON = 1e-12
MAD_FLOOR = 1e-9


def _series(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.shape != (DAYS,) or not np.all(np.isfinite(series)):
        raise ValueError("numerical input must be 365 finite daily values")
    return series


def _scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (DAYS,) or not np.all(np.isfinite(scores)):
        raise ValueError("numerical score must contain 365 finite values")
    return scores


def spectral_residual(values: np.ndarray) -> np.ndarray:
    """Three-bin circular log-spectrum residual; no CNN or local score smoothing."""
    series = _series(values)
    centered = series - np.median(series)
    if not np.any(centered):
        return np.zeros(DAYS, dtype=np.float64)
    spectrum = np.fft.fft(centered)
    log_amplitude = np.log(np.maximum(np.abs(spectrum), SPECTRAL_EPSILON))
    smoothed = (np.roll(log_amplitude, 1) + log_amplitude
                + np.roll(log_amplitude, -1)) / 3
    residual = log_amplitude - smoothed
    return _scores(np.abs(np.fft.ifft(np.exp(residual + 1j * np.angle(spectrum)))))


def _standardized(values: np.ndarray) -> np.ndarray:
    series = _series(values)
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    return (series - median) / max(1.4826 * mad, MAD_FLOOR)


def pelt_points(values: np.ndarray, beta: int) -> list[int]:
    """Return internal segment starts; ruptures' terminal 365 is not an event."""
    if beta not in (1, 2, 4, 8, 16, 32):
        raise ValueError("beta is outside the frozen candidate set")
    standardized = _standardized(values)
    if not np.any(standardized):
        return []
    breakpoints = rpt.Pelt(model="l2", min_size=3, jump=1).fit(
        standardized).predict(pen=beta * np.log(DAYS))
    if not breakpoints or breakpoints[-1] != DAYS:
        raise ValueError("PELT did not return the terminal boundary")
    return [int(day) for day in breakpoints[:-1]]


def _center_scores(window_scores: np.ndarray) -> np.ndarray:
    windows = np.asarray(window_scores, dtype=np.float64)
    if windows.shape != (DAYS - WINDOW + 1,) or not np.all(np.isfinite(windows)):
        raise ValueError("window score has invalid length or non-finite values")
    scores = np.zeros(DAYS, dtype=np.float64)
    scores[CENTER:CENTER + len(windows)] = windows
    return scores


def matrix_profile(values: np.ndarray) -> np.ndarray:
    """Thirty-day self-join discord distance at the right center of each window."""
    profile = stumpy.stump(_series(values), m=WINDOW, normalize=True)
    return _scores(_center_scores(np.asarray(profile[:, 0], dtype=np.float64)))


def theilsen_score(values: np.ndarray) -> np.ndarray:
    """Absolute robust slope in mm/day at each thirty-day window's right center."""
    series = _series(values)
    days = np.arange(WINDOW)
    windows = np.asarray([abs(theilslopes(series[i:i + WINDOW], days).slope)
                          for i in range(DAYS - WINDOW + 1)])
    return _scores(_center_scores(windows))


def score_to_points(values: np.ndarray, threshold: float) -> list[int]:
    if not np.isfinite(threshold):
        raise ValueError("point threshold must be finite")
    return np.flatnonzero(_scores(values) > threshold).tolist()


def score_to_ranges(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    if not np.isfinite(threshold):
        raise ValueError("range threshold must be finite")
    active = _scores(values) > threshold
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))
