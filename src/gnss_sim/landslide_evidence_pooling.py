"""Use adjacent raw displacement continuity to weight existing visual features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


def continuity_guides(evidence):
    """Map original noise-normalized changes into eight original chart columns.

    A continuous-change weight needs two consecutive supported, same-sign
    changes. Its harmonic magnitude suppresses a large isolated jump followed
    by a small fluctuation. This is an observation descriptor, never an activity
    label. Absolute changes form the discontinuity-emphasis control.
    """
    blocks = np.asarray(evidence["blocks"], dtype=float)
    edges = np.diff(blocks, axis=1)
    available = np.asarray(evidence["edge_available"], dtype=bool)
    valid_neighbors = available[:, :-1] & available[:, 1:]
    left, right = edges[:, :-1], edges[:, 1:]
    magnitude_left, magnitude_right = np.abs(left), np.abs(right)
    denominator = magnitude_left + magnitude_right
    same_direction = left * right > 0
    paired = np.divide(
        2 * magnitude_left * magnitude_right,
        denominator,
        out=np.zeros_like(left),
        where=(denominator > 1e-12) & same_direction & valid_neighbors[:, :, None],
    )
    changes = np.where(available[:, :, None], np.abs(edges), 0)
    continuous = np.zeros((len(blocks), 3, 8))
    discontinuous = np.zeros_like(continuous)
    # The unchanged 256px source chart has original relative days -90..90
    # at x=29..245. Its 8x8 backbone map has 32px column footprints.
    edge_times = -77.5 + 14 * np.arange(12)
    pair_times = -70.5 + 14 * np.arange(11)
    edge_columns = np.floor((29 + (edge_times + 90) * 216 / 180) / 32).astype(int)
    pair_columns = np.floor((29 + (pair_times + 90) * 216 / 180) / 32).astype(int)
    for edge, column in enumerate(edge_columns):
        discontinuous[:, :, column] += changes[:, edge]
    for pair, column in enumerate(pair_columns):
        continuous[:, :, column] += paired[:, pair]
    return continuous, discontinuous


def feature_weights(guide, permuted=False):
    """Weight coarse NEU plot footprints, retaining a uniform background floor.

    The floor is one daily-noise unit, fixed before fitting. This keeps
    unsupported or zero-continuity input on the exact native global-pool path.
    Final feature-map tokens have already mixed context; these are coarse
    readout weights, not isolated raw-date attribution.
    """
    guide = np.asarray(guide, dtype=float)
    if guide.ndim != 3 or guide.shape[1:] != (3, 8):
        raise ValueError("The unchanged chart needs three axes and eight columns")
    if not np.isfinite(guide).all() or np.any(guide < 0):
        raise ValueError("Continuity guides must be finite nonnegative evidence")
    weights = np.ones((len(guide), 8, 8), dtype=float)
    for axis in range(3):
        top, bottom = 12 + axis * 73, 72 + axis * 73
        for row in range(8):
            overlap = max(0, min(bottom, (row + 1) * 32) - max(top, row * 32))
            weights[:, row] += (overlap / 32) * guide[:, axis]
    if permuted:
        random = np.random.default_rng(2026101331)
        weights = weights.reshape(len(guide), 64)
        for index in range(len(weights)):
            weights[index] = weights[index, random.permutation(64)]
        weights = weights.reshape(len(guide), 8, 8)
    return (weights / weights.sum(axis=(1, 2), keepdims=True)).astype(np.float32)


def evidence_weighted_embedding(maps, projection, weights):
    """Change pooling only; reuse the exact pretrained projection and L2 norm."""
    if maps.ndim != 4 or maps.shape[1:] != (1024, 8, 8):
        raise ValueError("Evidence pooling requires the original MobileCLIP2-S0 map")
    weights = torch.as_tensor(weights, dtype=maps.dtype, device=maps.device)
    if weights.shape != (len(maps), 8, 8):
        raise ValueError("Every original query needs aligned coarse feature weights")
    pooled = (maps * weights[:, None]).sum(dim=(2, 3))
    return functional.normalize(projection(pooled), dim=-1)
