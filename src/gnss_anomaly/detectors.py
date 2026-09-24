import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .contracts import Prediction, Window, mask_events

NAMES = ("hampel", "cusum", "iforest")
NUMERICAL_NAMES = (
    *NAMES,
    "reference_trend",
)  # Standalone development candidate; not an Agent tool.


def segments(valid: np.ndarray):
    edges = np.diff(np.r_[False, valid, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def robust_scale(values: np.ndarray) -> float:
    return max(float(1.4826 * np.median(np.abs(values - np.median(values)))), 0.05)


def fill_short_gaps(values: np.ndarray, maximum: int = 2) -> np.ndarray:
    """Interpolate entire bounded short gaps only; never partly fill a long gap."""
    result = values.copy()
    for a, b in segments(~np.isfinite(values)):
        if b - a <= maximum and a > 0 and b < len(values):
            result[a:b] = np.linspace(values[a - 1], values[b], b - a + 2)[1:-1]
    return result


def features(x: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, float]:
    series = pd.Series(x)
    trend = series.rolling(width, center=True, min_periods=1).median().to_numpy()
    residual = x - trend
    delta = np.r_[0.0, np.diff(x)]
    scale = robust_scale(delta[1:]) / np.sqrt(2) if len(x) > 1 else 0.05
    scale = max(scale, 0.05)
    slope = pd.Series(delta).rolling(width, min_periods=1).mean().to_numpy()
    volatility = series.rolling(width, min_periods=2).std().fillna(0).to_numpy()
    return (
        np.column_stack((residual / scale, delta / scale, slope / scale, volatility / scale)),
        residual,
        scale,
    )


def record_coverage(diagnostics, scorable, values):
    if diagnostics is not None:
        observed = np.isfinite(values)
        diagnostics.update(
            scorable_mask=scorable.tolist(),
            scorable_cells=int(scorable.sum()),
            observed_cells=int(observed.sum()),
            numerical_coverage=float(scorable.sum() / observed.sum()) if observed.any() else None,
        )


def detect(window: Window, name: str, config: dict, seed: int = 42, diagnostics=None) -> Prediction:
    if window.sampling_hours != config.get("sampling_hours", 1):
        raise ValueError("detector sampling interval must match the window")
    if name not in NUMERICAL_NAMES:
        raise ValueError("unknown numerical detector")
    if name == "reference_trend":
        if config.get("max_interpolation_samples", 0) != 0:
            raise ValueError("reference_trend does not support interpolation")
        from .reference_trend import detect_reference_trend

        return detect_reference_trend(window, config["reference_trend"], diagnostics)
    if config.get("implementation") == "daily-observed-v2":
        from .daily_detection import detect_daily

        return detect_daily(window, name, config, seed, diagnostics)
    values = window.array()
    flags = np.zeros_like(values, dtype=bool)
    scorable = np.zeros_like(values, dtype=bool)
    usable = 0
    for c in range(3):
        # Short gaps support feature calculation only. Long gaps reset detector state.
        filled = fill_short_gaps(values[:, c], config.get("max_interpolation_samples", 2))
        for a, b in segments(np.isfinite(filled)):
            x = filled[a:b]
            if len(x) < config["min_samples"]:
                continue
            usable += len(x)
            scorable[a:b, c] = np.isfinite(values[a:b, c])
            matrix, residual, scale = features(x, config["window"])
            if name == "hampel":
                local_scale = (
                    pd.Series(residual)
                    .rolling(config["window"], center=True, min_periods=3)
                    .apply(lambda y: robust_scale(y), raw=True)
                    .fillna(scale)
                    .to_numpy()
                )
                score = np.abs(residual) / np.maximum(local_scale, scale)
                detected = score >= config["hampel_threshold"]
            elif name == "cusum":
                # Velocity departures; accumulate within each continuous segment only.
                delta = np.diff(x, prepend=x[0])
                z = (delta - np.median(delta[1:])) / max(robust_scale(delta[1:]), 0.05)
                pos = neg = 0.0
                detected = np.zeros(len(x), dtype=bool)
                for i, v in enumerate(z):
                    pos = max(0.0, pos + v - config["cusum_drift"])
                    neg = max(0.0, neg - v - config["cusum_drift"])
                    detected[i] = max(pos, neg) >= config["cusum_threshold"]
            else:
                # Explicit unsupervised/transductive baseline; no labels or test threshold search.
                model = IsolationForest(
                    n_estimators=config["iforest_trees"],
                    random_state=seed,
                    n_jobs=1,
                    contamination="auto",
                )
                model.fit(matrix)
                detected = -model.score_samples(matrix) >= config["iforest_threshold"]
            flags[a:b, c] = detected & np.isfinite(values[a:b, c])
    record_coverage(diagnostics, scorable, values)
    if not usable:
        return Prediction(status="insufficient", reason="no sufficiently long continuous segment")
    return Prediction(events=mask_events(flags))


def quality(window: Window) -> dict:
    values = window.array()
    missing = ~np.isfinite(values)
    gaps = [b - a for a, b in segments(missing.any(axis=1))]
    slopes = []
    for c in range(3):
        valid = np.isfinite(values[:, c])
        idx = np.flatnonzero(valid)
        slopes.append(float(np.polyfit(idx, values[valid, c], 1)[0]) if len(idx) >= 2 else None)
    result = {
        "hours": len(values) * window.sampling_hours,
        "missing_fraction": float(missing.mean()),
        "max_gap_hours": int(max(gaps, default=0)) * window.sampling_hours,
        "observed_per_channel": (~missing).sum(axis=0).tolist(),
        "linear_slope_mm_per_hour": [
            s / window.sampling_hours if s is not None else None for s in slopes
        ],
    }
    if window.sampling_hours == 24:
        result.update(
            days=len(values), max_gap_days=int(max(gaps, default=0)), linear_slope_mm_per_day=slopes
        )
    return result
