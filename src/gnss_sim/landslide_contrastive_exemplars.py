"""Match center-labelled training plots by calendar-preserving motion shape."""

from __future__ import annotations

import warnings

import numpy as np


def motion_shape(blocks):
    """Describe original adjacent increments, never join across missing slots.

    The twelve log increment lengths preserve noise-normalized signal strength.
    Eleven successive cosines describe continuity without choosing a physical
    direction or treating the U axis as equal-quality motion evidence.
    This descriptor is a retrieval proxy, not an activity classifier or confidence.
    """
    blocks = np.asarray(blocks, dtype=float)
    if blocks.shape != (13, 3):
        raise ValueError("Motion retrieval requires thirteen original NEU slots")
    changes = np.diff(blocks, axis=0)
    lengths = np.linalg.norm(changes, axis=1)
    products = lengths[:-1] * lengths[1:]
    cosines = np.divide(np.sum(changes[:-1] * changes[1:], axis=1), products,
                        out=np.full(11, np.nan), where=products > 0)
    return np.r_[np.log1p(lengths), cosines]


def shape_distance(query, candidates):
    """Mean squared distance on at least twelve jointly available coordinates."""
    query = np.asarray(query, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if query.shape != (23,) or candidates.ndim != 2 or candidates.shape[1] != 23:
        raise ValueError("Query and bank must use the same23-coordinate descriptor")
    shared = np.isfinite(candidates) & np.isfinite(query)
    count = shared.sum(axis=1)
    squared = np.where(shared, (candidates - query) ** 2, 0)
    distance = np.divide(squared.sum(axis=1), count, out=np.full(len(count), np.inf),
                         where=count >= 12)
    # Both increments around the queried center must be available; a remote
    # matching shape cannot substitute for the actual center observations.
    center_available = shared[:, 5] & shared[:, 6]
    distance[~center_available] = np.inf
    return distance


def contrastive_neighbors(query, bank):
    """Retrieve one labelled example per class; retain explicit distances."""
    distances = shape_distance(query, np.array([row["descriptor"] for row in bank], dtype=float))
    result = {}
    for code, name in ((0, "inactive"), (1, "active")):
        candidates = [index for index, row in enumerate(bank)
                      if row["center_label"] == code and np.isfinite(distances[index])]
        if not candidates:
            return None
        index = min(candidates, key=lambda value: (distances[value], bank[value]["example_id"]))
        result[name] = {"index": index, "example_id": bank[index]["example_id"],
                        "distance": float(distances[index])}
    return result


def fixed_class_examples(bank):
    """Choose class medoids once from training, independently of any query."""
    result = {}
    for code, name in ((0, "inactive"), (1, "active")):
        indices = [index for index, row in enumerate(bank) if row["center_label"] == code]
        matrix = np.array([bank[index]["descriptor"] for index in indices], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(matrix, axis=0)
        distance = shape_distance(center, matrix)
        eligible = [offset for offset in range(len(indices)) if np.isfinite(distance[offset])]
        if not eligible:
            raise ValueError("Training class has no eligible fixed example")
        offset = min(eligible, key=lambda value: (distance[value], bank[indices[value]]["example_id"]))
        index = indices[offset]
        result[name] = {"index": index, "example_id": bank[index]["example_id"],
                        "distance": float(distance[offset])}
    return result
