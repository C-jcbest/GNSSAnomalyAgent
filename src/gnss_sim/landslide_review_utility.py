"""Calibrate bounded observation choice on actual visual correction utility."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["initial_active_fraction", "initial_unknown_fraction", "missed_raw_motion_fraction",
                 "unsupported_initial_activity_fraction", "expected_initial_error_fraction"]


def window_review_features(initial, features, long_evidence, probability, windows, policy):
    """Summarize existing GNSS conflicts and a common trained numerical prior.

    The probability is from the frozen common train24 raw84 XGBoost. It is a
    numerical prediction, not independent evidence or a calibrated uncertainty.
    Unknown initial decisions have expected error1 because they cannot already
    be a correct definite binary decision. No reference or local answer is used.
    """
    initial, probability = np.asarray(initial), np.asarray(probability)
    if initial.shape != features.observed.shape or probability.shape != initial.shape:
        raise ValueError("Utility features require the original observation calendar")
    if not np.isin(initial, [-1, 0, 1]).all() or np.any(initial[~features.observed] != -1):
        raise ValueError("Utility features require valid activity and missing states")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("The numerical prior must be a finite probability")
    short_supported = np.isfinite(features.progressive_rate) & features.observed
    long_supported = np.isfinite(long_evidence["rate"]) & features.observed
    short_motion = ((features.progressive_rate >= policy.minimum_rate)
                    & (features.signal_to_noise >= policy.minimum_signal_to_noise))
    long_motion = ((long_evidence["rate"] >= policy.minimum_rate)
                   & (long_evidence["signal_to_noise"] >= policy.minimum_signal_to_noise))
    supported = short_supported | long_supported
    motion = (short_supported & short_motion) | (long_supported & long_motion)
    missed = supported & motion & (initial != 1)
    unsupported = supported & ~motion & (initial == 1)
    expected_error = np.where(initial == 1, 1 - probability, probability)
    expected_error[initial == -1] = 1
    rows = []
    for start, stop in windows:
        if not 0 <= start < stop <= len(initial):
            raise ValueError("Utility window lies outside the observation calendar")
        observed = features.observed[start:stop]
        count = int(observed.sum())
        if count == 0:
            rows.append([0.] * len(FEATURE_NAMES))
            continue
        rows.append([float(((initial[start:stop] == 1) & observed).sum() / count),
                     float(((initial[start:stop] == -1) & observed).sum() / count),
                     float(missed[start:stop].sum() / count), float(unsupported[start:stop].sum() / count),
                     float(expected_error[start:stop][observed].mean())])
    return np.asarray(rows)


def correction_utility(actual, initial, reviewed, permission):
    """Count net newly correct decisions; unknown outcomes do not earn TN credit."""
    for values in (actual, initial, reviewed):
        if values.shape != permission.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Utility requires aligned binary/unknown decisions")
    eligible = permission & (actual >= 0)
    correct_before = initial == actual
    correct_after = reviewed == actual
    gained = int((eligible & ~correct_before & correct_after).sum())
    lost = int((eligible & correct_before & ~correct_after).sum())
    return {"eligible_days": int(eligible.sum()), "gained_correct": gained, "lost_correct": lost,
            "net_correct": gained - lost,
            "utility_fraction": (gained - lost) / int(eligible.sum()) if eligible.any() else 0.}


def fit_utility_model(matrix, target, feature_indices):
    """Fit one fixed Ridge model; no tuning split or test-score search is used."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    matrix, target = np.asarray(matrix), np.asarray(target)
    scaler = StandardScaler().fit(matrix[:, feature_indices])
    regression = Ridge(alpha=1.).fit(scaler.transform(matrix[:, feature_indices]), target)
    return {"feature_indices": list(feature_indices), "feature_names": [FEATURE_NAMES[i] for i in feature_indices],
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficients": regression.coef_.tolist(), "intercept": float(regression.intercept_),
            "ridge_alpha": 1., "training_windows": len(target)}


def predict_utility(model, matrix):
    scaled = (matrix[:, model["feature_indices"]] - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return scaled @ np.asarray(model["coefficients"]) + model["intercept"]


def select_utility_windows(windows, predicted_fractions, editable_counts, maximum_reviews=2):
    """Choose up to two disjoint views with positive expected net corrections.

    This is an observation decision only. Predictions cannot assign activity or
    choose exact boundaries; the unchanged visual reviewer supplies those states.
    Zero is the literal no-expected-benefit point, not a swept admission threshold.
    """
    if len(windows) != len(predicted_fractions) or len(windows) != len(editable_counts):
        raise ValueError("Utility scores must match every candidate window")
    priorities = [{"index": index, "start": start, "stop": stop,
                   "predicted_utility_fraction": float(predicted_fractions[index]),
                   "editable_days": int(editable_counts[index]),
                   "expected_net_corrections": float(predicted_fractions[index] * editable_counts[index])}
                  for index, (start, stop) in enumerate(windows)]
    selected = []
    for row in sorted(priorities, key=lambda item: (-item["expected_net_corrections"], item["start"])):
        if row["expected_net_corrections"] <= 0:
            break
        if any(row["start"] < item["stop"] and item["start"] < row["stop"] for item in selected):
            continue
        selected.append(row)
        if len(selected) == maximum_reviews:
            break
    return priorities, sorted(selected, key=lambda item: item["start"])
