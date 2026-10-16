"""Learn whether a bounded visual revision should replace the initial decision."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "changed_observed_fraction",
    "initial_unknown_fraction_on_changes",
    "review_unknown_fraction_on_changes",
    "raw_agreement_gain_on_changes",
    "numeric_expected_correctness_gain_on_changes",
]


def proposal_features(initial, reviewed, permission, features, long_evidence, probability, policy):
    """Describe the supplied visual changes without assigning new activity dates.

    Unknown states have zero expected correctness. Raw motion is an observation
    proxy, and XGBoost probabilities are an uncalibrated prior, not ground truth.
    The unit of acceptance is the entire permitted proposal in one local view.
    """
    observed = features.observed
    initial = np.asarray(initial)
    reviewed = np.asarray(reviewed)
    probability = np.asarray(probability)
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Proposal decisions must match the observation calendar")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing dates must remain unknown")
    if permission.shape != observed.shape or permission.dtype != bool or np.any(permission & ~observed):
        raise ValueError("Proposal permission must contain observed dates only")
    if probability.shape != observed.shape or np.any(~np.isfinite(probability)):
        raise ValueError("The numerical prior must match the calendar and be finite")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("The numerical prior must be a probability")
    changed = permission & (initial != reviewed)
    if not changed.any():
        return np.zeros(len(FEATURE_NAMES)), changed
    supported = observed & (
        np.isfinite(features.progressive_rate) | np.isfinite(long_evidence["rate"])
    )
    motion = (
        (features.progressive_rate >= policy.minimum_rate)
        & (features.signal_to_noise >= policy.minimum_signal_to_noise)
    ) | (
        (long_evidence["rate"] >= policy.minimum_rate)
        & (long_evidence["signal_to_noise"] >= policy.minimum_signal_to_noise)
    )
    raw_before = supported & (initial >= 0) & ((initial == 1) == motion)
    raw_after = supported & (reviewed >= 0) & ((reviewed == 1) == motion)
    expected_before = np.where(initial == 1, probability, 1 - probability)
    expected_after = np.where(reviewed == 1, probability, 1 - probability)
    expected_before[initial == -1] = 0
    expected_after[reviewed == -1] = 0
    row = np.array([
        changed.sum() / permission.sum(),
        np.mean(initial[changed] == -1),
        np.mean(reviewed[changed] == -1),
        np.mean(raw_after[changed].astype(float) - raw_before[changed]),
        np.mean(expected_after[changed] - expected_before[changed]),
    ])
    return row, changed


def fit_proposal_model(matrix, target, feature_indices):
    """Fit one predeclared Ridge model; alpha and admission threshold are fixed."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(matrix)
    scaler = StandardScaler().fit(matrix[:, feature_indices])
    regression = Ridge(alpha=1).fit(scaler.transform(matrix[:, feature_indices]), target)
    return {
        "feature_indices": list(feature_indices),
        "feature_names": [FEATURE_NAMES[index] for index in feature_indices],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": regression.coef_.tolist(),
        "intercept": float(regression.intercept_),
        "ridge_alpha": 1,
        "training_proposals": len(target),
    }


def expected_proposal_utility(model, row):
    selected = row[model["feature_indices"]]
    standardized = (selected - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(standardized @ np.asarray(model["coefficients"]) + model["intercept"])


def apply_proposal(initial, reviewed, permission, expected_utility):
    """A rejected proposal preserves the initial decision, including unknowns."""
    if not np.isfinite(expected_utility):
        raise ValueError("Proposal utility must be finite")
    result = initial.copy()
    if expected_utility > 0:
        result[permission] = reviewed[permission]
    return result
