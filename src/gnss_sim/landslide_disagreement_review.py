"""Restrict numerical revision to disagreements with positive learned utility."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def review_features(visual_probability, numeric_probability, numbers, threshold):
    visual_probability = np.asarray(visual_probability)
    numeric_probability = np.asarray(numeric_probability)
    scores = np.column_stack(
        (
            visual_probability,
            numeric_probability,
            visual_probability - numeric_probability,
            np.abs(visual_probability - threshold),
            np.abs(numeric_probability - threshold),
            visual_probability * numeric_probability,
        )
    )
    if numbers is None:
        return scores
    return np.column_stack((scores, numbers))


def fit_review(features, visual_probability, numeric_probability, actual, threshold, alpha):
    """Estimate the change in correct decisions from out-of-record predictions.

    The two experts' training-held-out decisions define benefit, not confidence
    or artifact truth. Matching decisions supply no permission to revise.
    """
    visual = np.asarray(visual_probability) >= threshold
    numeric = np.asarray(numeric_probability) >= threshold
    actual = np.asarray(actual)
    selected = (actual >= 0) & (visual != numeric)
    if not np.any(selected):
        return dict(
            status="no_training_disagreement",
            samples=0,
            mean=None,
            scale=None,
            coefficient=None,
            intercept=0.0,
        )
    benefit = (numeric[selected] == actual[selected]).astype(float) - (
        visual[selected] == actual[selected]
    ).astype(float)
    scaler = StandardScaler().fit(features[selected])
    model = Ridge(alpha=alpha).fit(scaler.transform(features[selected]), benefit)
    return dict(
        status="fitted",
        samples=int(selected.sum()),
        benefit_sum=float(benefit.sum()),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coefficient=model.coef_.tolist(),
        intercept=float(model.intercept_),
    )


def review_benefit(model, features):
    if model["status"] == "no_training_disagreement":
        return np.zeros(len(features))
    scaled = (features - np.array(model["mean"])) / np.array(model["scale"])
    return scaled @ np.array(model["coefficient"]) + model["intercept"]


def revise_disagreements(visual_probability, numeric_probability, supported, benefit, threshold):
    """Make at most one source switch; never manufacture activity or fill gaps."""
    visual_probability = np.asarray(visual_probability)
    numeric_probability = np.asarray(numeric_probability)
    supported = np.asarray(supported, dtype=bool)
    benefit = np.asarray(benefit)
    if not (
        visual_probability.shape == numeric_probability.shape == supported.shape == benefit.shape
    ):
        raise ValueError("Review evidence must share the original query calendar")
    if (
        not np.isfinite(visual_probability[supported]).all()
        or not np.isfinite(numeric_probability[supported]).all()
        or not np.isfinite(benefit[supported]).all()
    ):
        raise ValueError("Supported review evidence must be finite")
    disagreement = (visual_probability >= threshold) != (numeric_probability >= threshold)
    switched = supported & disagreement & (benefit > 0)
    result = visual_probability.copy()
    result[switched] = numeric_probability[switched]
    return result, switched
