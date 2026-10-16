"""Label proposed temporal pieces instead of asking a VLM to invent event bounds."""

from __future__ import annotations

import json

import numpy as np

from gnss_sim.landslide_adaptive_review import review_prompt
from gnss_sim.landslide_window_experiment import WINDOW_TASK


def rate_partition(rates, start, stop, maximum_piece_days=30):
    """Recover constant-rate runs from the existing PELT numerical baseline.

    The existing numerical feedback reports 30-day blocks. Bound the piece
    length by that same reporting scale, without losing fitted-rate transitions.
    Unavailable fitted-rate spans remain pieces that can be inspected in raw
    displacement. They are not automatically stationary or unknown. Coincident
    adjacent rate estimates are merged; these are not all native PELT edges.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or not 0 <= start < stop <= len(rates) or np.isinf(rates).any():
        raise ValueError("Rate partition requires a valid one-dimensional calendar")
    if type(maximum_piece_days) is not int or maximum_piece_days < 1:
        raise ValueError("Maximum piece length must be a positive integer")
    sample = rates[start:stop]
    same = (sample[1:] == sample[:-1]) | (np.isnan(sample[1:]) & np.isnan(sample[:-1]))
    native_edges = np.r_[start, np.flatnonzero(~same) + start + 1, stop]
    edges = [start]
    for left, right in zip(native_edges[:-1], native_edges[1:], strict=True):
        edges.extend(range(int(left) + maximum_piece_days, int(right), maximum_piece_days))
        edges.append(int(right))
    return [
        {"id": index, "start": int(left), "stop": int(right)}
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def uniform_partition(count, start, stop):
    """Match the number of decisions while removing data-derived boundary placement."""
    if type(count) is not int or not 1 <= count <= stop - start or not 0 <= start < stop:
        raise ValueError("Uniform partition needs positive, distinct calendar pieces")
    edges = np.linspace(start, stop, count + 1, dtype=int)
    return [
        {"id": index, "start": int(left), "stop": int(right)}
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def validate_partition(pieces, days):
    if not pieces:
        raise ValueError("A review needs at least one proposed temporal piece")
    previous = pieces[0]["start"]
    for index, piece in enumerate(pieces):
        if set(piece) != {"id", "start", "stop"}:
            raise ValueError("Piece requires exactly id, start and stop")
        if any(type(piece[key]) is not int for key in piece):
            raise ValueError("Piece indices must be integers")
        if piece["id"] != index or not 0 <= previous == piece["start"] < piece["stop"] <= days:
            raise ValueError("Pieces must be ordered and cover a contiguous calendar")
        previous = piece["stop"]


def partition_review_prompt(features, long_evidence, pieces, numeric_feedback):
    validate_partition(pieces, len(features.observed))
    start, stop = pieces[0]["start"], pieces[-1]["stop"]
    original = WINDOW_TASK.format(start=start, end=stop - 1)
    task = original.split("All endpoints are inclusive integers.", 1)[0]
    task += (
        "\nThe target is partitioned into proposed temporal pieces below. These "
        "are numerical/sampling proposals, NOT confirmed event boundaries or "
        "activity answers. Inspect the raw displacement within EACH piece. "
        "Return 1 only when sustained continuing displacement is supported "
        "throughout its observed portion, 0 when that portion is stationary "
        "without sustained motion, and -1 when evidence is inconclusive or a "
        "motion/stationary transition lies inside the piece. A numerical fit "
        "being unavailable does not prevent raw visual judgment. Missing dates "
        "remain unknown. Do not invent new boundaries, remove pieces, or infer "
        "generating parameters. Respond only with JSON with exactly two keys: "
        '{"piece_states":[[0,state],[1,state]],"evidence":"brief raw observations"}. '
        "Include EVERY piece id exactly once in printed order; states must be "
        "integer -1,0,1. Evidence should describe inspected raw motion, not hidden "
        "reasoning. No stage labels or extra keys. Piece start/stop below are "
        "half-open: start is included and stop is excluded.\n" + json.dumps(pieces)
    )
    if numeric_feedback:
        # Retain the exact existing norm evidence and explanation. Only the
        # representation of the resulting activity decision changes.
        task += review_prompt(features, long_evidence, start, stop, True)[len(original) :]
    return task


def parse_piece_activity(payload, observed, pieces):
    validate_partition(pieces, len(observed))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"piece_states", "evidence"}
        or not isinstance(payload["evidence"], str)
        or not payload["evidence"].strip()
        or not isinstance(payload["piece_states"], list)
        or len(payload["piece_states"]) != len(pieces)
    ):
        raise ValueError("Expected exactly one decision per piece and a raw evidence string")
    activity = np.full(len(observed), -1, dtype=int)
    for piece, decision in zip(pieces, payload["piece_states"], strict=True):
        if (
            not isinstance(decision, list)
            or len(decision) != 2
            or any(type(value) is not int for value in decision)
            or decision[0] != piece["id"]
            or decision[1] not in (-1, 0, 1)
        ):
            raise ValueError("Piece decisions must contain ordered ids and valid integer states")
        activity[piece["start"] : piece["stop"]] = decision[1]
    activity[~observed] = -1
    return activity


def numeric_piece_activity(probability, observed, pieces, threshold):
    """A same-partition numerical control; calibrate its threshold on validation only."""
    validate_partition(pieces, len(observed))
    probability = np.asarray(probability)
    if probability.shape != observed.shape or not np.isfinite(probability).all():
        raise ValueError("Numerical piece control requires aligned finite probabilities")
    if np.any((probability < 0) | (probability > 1)) or not 0 < threshold < 1:
        raise ValueError("Numerical piece control requires probabilities in [0,1]")
    activity = np.full(len(observed), -1, dtype=int)
    for piece in pieces:
        start, stop = piece["start"], piece["stop"]
        available = observed[start:stop]
        if available.any():
            activity[start:stop] = int(probability[start:stop][available].mean() >= threshold)
    activity[~observed] = -1
    return activity
