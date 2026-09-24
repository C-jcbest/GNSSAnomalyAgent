"""Daily, retrospective baselines using observed cells without imputation."""

import numpy as np
from sklearn.ensemble import IsolationForest

from .contracts import Prediction, mask_events
from .detectors import record_coverage, robust_scale


def daily_features(x, width, minimum):
    n, half = len(x), width // 2
    median = np.full(n, np.nan)
    spread, slope, volatility = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        indices = np.arange(max(0, i - half), min(n, i + half + 1))
        indices = indices[np.isfinite(x[indices])]
        if len(indices) < minimum:
            continue
        values = x[indices]
        median[i] = np.median(values)
        # Residual scale retains the Hampel baseline's protection against local trends.
        slope[i] = np.polyfit(indices - i, values, 1)[0]
        volatility[i] = np.std(values, ddof=1)
    residual = x - median
    for i in range(n):
        local = residual[max(0, i - half) : min(n, i + half + 1)]
        local = local[np.isfinite(local)]
        if len(local) >= 3:
            spread[i] = robust_scale(local)
    delta = np.r_[np.nan, np.diff(x)]  # NaN across every unobserved calendar day.
    adjacent = delta[np.isfinite(delta)]
    scale = max(robust_scale(adjacent) / np.sqrt(2), 0.05) if len(adjacent) else 0.05
    spread = np.where(np.isfinite(spread), np.maximum(spread, scale), scale)
    matrix = np.column_stack((residual / scale, delta / scale, slope / scale, volatility / scale))
    return residual, spread, matrix


def daily_cusum(x, config):
    delta = np.r_[np.nan, np.diff(x)]
    scorable = np.isfinite(delta)
    flags = np.zeros(len(x), dtype=bool)
    if scorable.sum() < config["min_samples"]:
        return flags, np.zeros(len(x), dtype=bool)
    sample = delta[scorable]
    center, scale = np.median(sample), robust_scale(sample)
    pos = neg = 0.0
    gap = 0
    for i, value in enumerate(x):
        if not np.isfinite(value):
            gap += 1
            if gap > config["reset_gap_days"]:
                pos = neg = 0.0
            continue
        gap = 0
        if not scorable[i]:
            continue
        z = (delta[i] - center) / scale
        pos = max(0.0, pos + z - config["cusum_drift"])
        neg = max(0.0, neg - z - config["cusum_drift"])
        flags[i] = max(pos, neg) >= config["cusum_threshold"]
    return flags, scorable


def detect_daily(window, name, config, seed, diagnostics=None):
    if window.sampling_hours != 24 or config.get("max_interpolation_samples", 0) != 0:
        raise ValueError("daily-observed-v2 requires daily inputs and no interpolation")
    values = window.array()
    flags, scorable = (np.zeros_like(values, dtype=bool) for _ in range(2))
    for c in range(3):
        x = values[:, c]
        if name == "cusum":
            flags[:, c], scorable[:, c] = daily_cusum(x, config)
            continue
        residual, spread, matrix = daily_features(x, config["window"], config["min_window_samples"])
        if name == "hampel":
            scorable[:, c] = np.isfinite(residual)
            flags[:, c] = scorable[:, c] & (np.abs(residual) / spread >= config["hampel_threshold"])
        else:
            valid = np.isfinite(matrix).all(axis=1)
            if valid.sum() < config["min_samples"]:
                continue
            model = IsolationForest(
                n_estimators=config["iforest_trees"],
                random_state=seed,
                contamination="auto",
                n_jobs=1,
            )
            model.fit(matrix[valid])
            scorable[:, c] = valid
            flags[valid, c] = -model.score_samples(matrix[valid]) >= config["iforest_threshold"]
    record_coverage(diagnostics, scorable, values)
    if not scorable.any():
        return Prediction(status="insufficient", reason="no scorable daily observations")
    return Prediction(events=mask_events(flags))
