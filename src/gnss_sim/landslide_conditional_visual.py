"""Keep visual variation that cannot be linearly predicted from motion evidence."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def fit_visual_expectation(numbers, embeddings, shuffled=False):
    """Use eligible TRAIN observations only; activity labels are not inputs.

    A fixed ridge coefficient avoids a second validation search. The shuffled
    control removes observation correspondence while retaining both feature
    collections. Neither projection is described as learning artifact truth.
    """
    numbers = np.asarray(numbers, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if numbers.ndim != 2 or embeddings.ndim != 2 or len(numbers) != len(embeddings):
        raise ValueError("Numerical and visual TRAIN queries must correspond")
    if len(numbers) < 2 or not np.isfinite(numbers).all() or not np.isfinite(embeddings).all():
        raise ValueError("Visual expectation requires finite paired TRAIN features")
    target = embeddings
    if shuffled:
        indices = np.random.default_rng(2026101326).permutation(len(embeddings))
        target = embeddings[indices]
    model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky").fit(numbers, target)
    return dict(
        coefficient=model.coef_.tolist(),
        intercept=model.intercept_.tolist(),
        visual_mean=embeddings.mean(axis=0).tolist(),
        alpha=1.0,
        samples=len(numbers),
        shuffled=shuffled,
    )


def conditional_visual_features(embeddings, numbers, expectation, kind):
    """Return a single registered representation on the original query calendar."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.float64)
    if kind == "raw":
        return embeddings.astype(np.float32)
    if kind == "mean":
        residual = embeddings - np.array(expectation["visual_mean"])
    elif kind in {"conditional", "unnormalized"}:
        predictable = numbers @ np.array(expectation["coefficient"]).T
        predictable += np.array(expectation["intercept"])
        residual = embeddings - predictable
    else:
        raise ValueError("Unregistered conditional visual representation: " + kind)
    if kind != "unnormalized":
        # A zero residual carries no visual information and stays zero. The
        # floor is numerical protection, not a movement threshold or veto.
        denominator = np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
        residual = residual / denominator
    return residual.astype(np.float32)
