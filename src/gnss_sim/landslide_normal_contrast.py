"""Observed training examples for contrasting GNSS background and target motion."""

from __future__ import annotations

import numpy as np


def anchored_blocks(values, start, stop):
    """Describe six raw 30-day blocks in physical units, without filling gaps."""
    if stop - start != 180 or not 0 <= start < stop <= len(values):
        raise ValueError("A normal contrast target must contain 180 original calendar days")
    medians = []
    for left in range(start, stop, 30):
        sample = values[left:left + 30]
        complete = sample[np.isfinite(sample).all(axis=1)]
        if len(complete) < 15:
            return None
        medians.append(np.median(complete, axis=0))
    medians = np.array(medians)
    return (medians[1:] - medians[0]).ravel()


def select_normal_examples(query_descriptor, bank, noise, count=3):
    """Retrieve similar raw change shapes from distinct training backgrounds.

    Similarity is a retrieval device, not an anomaly score or activity decision.
    A separate fixed-example control tests whether retrieval adds any value.
    """
    if count < 1 or noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Expected a positive three-axis observation noise proxy")
    backgrounds = {item["background"] for item in bank}
    if len(backgrounds) < count:
        raise ValueError("Not enough separate normal training backgrounds")
    if query_descriptor is None:
        return [], "query_block_support_unavailable"
    query_descriptor = np.asarray(query_descriptor, dtype=float)
    if query_descriptor.shape != (15,) or not np.isfinite(query_descriptor).all():
        raise ValueError("Expected five three-axis changes")
    ranked = []
    for item in bank:
        difference = (query_descriptor - np.array(item["descriptor"])) / np.tile(noise, 5)
        distance = float(np.sqrt(np.mean(difference ** 2)))
        ranked.append({**item, "distance": distance})
    ranked.sort(key=lambda item: (item["distance"], item["case_id"], item["start"]))
    selected, used_backgrounds = [], set()
    for item in ranked:
        if item["background"] in used_backgrounds:
            continue
        selected.append(item)
        used_backgrounds.add(item["background"])
        if len(selected) == count:
            break
    return selected, "available"


def anchored_series(values, start, stop):
    """Remove a constant first-block median; preserve amplitude and missing dates."""
    series = values[start:stop].copy()
    complete = series[:30][np.isfinite(series[:30]).all(axis=1)]
    if len(complete) < 15:
        raise ValueError("A supporting example lacks first-block observations")
    return series - np.median(complete, axis=0)
