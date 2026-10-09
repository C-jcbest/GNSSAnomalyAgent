"""Raw-displacement window judgments and fixed aggregation for visual experiments."""
from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction, event_masks
from gnss_sim.landslide_visual import parse_activity

WINDOW_TASK = """Review the observed GNSS raw displacement retrospectively.
The first image is global context. The second image contains local raw evidence.
Judge ONLY original calendar days {start}..{end}, inclusive, even if other
windows or earlier/later days are visible. Do not return intervals outside this target.
Confirm sustained cumulative displacement, including weak movement when visible.
A high stationary plateau after movement is NOT continuing activity. Separate
movement, intermediate plateaus, restart and uncertain boundaries. Noise, isolated
spikes, a single instrument-like step, or a derivative peak alone are insufficient.
Inspect raw N/E/U in the beginning, middle and end of the target. Movement need not
have positive slopes, or the same sign on all axes. Do not infer physical failure,
generating stages, hidden ground truth, or movement not supported by raw observations.
Missing dates keep their original positions. Leave inconclusive dates uncertain.
All endpoints are inclusive integers. Each list must be sorted and non-overlapping;
activity and uncertain must be disjoint. Adjacent intervals must not share endpoints:
[100,149] and [150,200], never [100,150] and [150,200]. Empty lists are valid.
Return ONLY JSON with exactly three keys:
{{"activity":[[start,end]],"uncertain":[[start,end]],"evidence":"brief visible observations"}}.
The evidence is a short, checkable description of raw displacement, not hidden reasoning.
No stage labels, Markdown, extra keys or numerical series values.
"""

REVIEW_TASK = """This is a separate confirmation review of a visual candidate.
Judge ONLY days {start}..{end}, inclusive. The first image is global raw context,
and the second is an enlarged raw view with context around the target.
First check whether the candidate contains sustained continuing displacement in its
beginning, middle and end. Split any stationary plateau from active motion, even if
the plateau retains a large accumulated offset. Do not bridge a stationary plateau
between two movements. Check weak motion cautiously; subtle visible accumulation
must not be rejected solely because earlier motion was larger. Noise or drift that
cannot be distinguished from motion remains uncertain. Do not require all axes to
move with the same sign. Neither derivatives nor prior stage labels are provided.
Revise candidate boundaries only inside the target using visible raw observations.
Do not label context outside the target, infer failure, or invent missing observations.
All endpoints are inclusive integer original day indices. Each list must be sorted
and non-overlapping, activity and uncertain disjoint; no shared adjacent endpoints.
Return ONLY JSON with exactly:
{{"activity":[[start,end]],"uncertain":[[start,end]],"evidence":"brief visible observations"}}.
Use empty lists when appropriate. No stages, Markdown or extra keys.
"""


def parse_window_activity(payload, observed, start, stop):
    """Reject coordinate mistakes; never clip a response to make it valid."""
    if not isinstance(payload, dict) or set(payload) != {"activity", "uncertain", "evidence"}:
        raise ValueError("Expected activity, uncertain and evidence")
    evidence = payload["evidence"]
    if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 1600:
        raise ValueError("A short non-empty observation description is required")
    activity = parse_activity(
        {"activity": payload["activity"], "uncertain": payload["uncertain"]}, observed
    )
    for field in ("activity", "uncertain"):
        for left, right in payload[field]:
            if left < start or right >= stop:
                raise ValueError("Output extends beyond target window; no automatic clipping")
    activity[:start] = -1
    activity[stop:] = -1
    return activity


def aggregate_windows(window_results, observed):
    """Strict majority of all covering views; unknown and failures retain votes."""
    days = len(observed)
    coverage = np.zeros(days, dtype=int)
    positive = np.zeros(days, dtype=int)
    negative = np.zeros(days, dtype=int)
    for result in window_results:
        start, stop = result["start"], result["stop"]
        values = np.asarray(result["activity"], dtype=int)
        if values.shape != (days,) or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Window output has an invalid calendar or state")
        coverage[start:stop] += 1
        positive[start:stop] += values[start:stop] == 1
        negative[start:stop] += values[start:stop] == 0
    if np.any(coverage == 0):
        raise ValueError("Frozen windows do not cover the full calendar")
    activity = np.full(days, -1, dtype=int)
    activity[2 * positive > coverage] = 1
    activity[2 * negative > coverage] = 0
    activity[~observed] = -1
    stage = np.where(activity == 0, 0, -1)
    failed = all(result["status"] == "failed" for result in window_results)
    prediction = ActivityPrediction(activity, stage, status="failed" if failed else "success")
    prediction.validate(days)
    return prediction


def review_candidates(activity):
    """Use the common event rule: bridge <=7 unknown days, never confirmed negatives."""
    candidates = []
    for mask in event_masks(activity):
        indices = np.flatnonzero(mask)
        candidates.append((int(indices[0]), int(indices[-1]) + 1))
    return candidates


def apply_reviews(base_activity, reviews, observed):
    """Explicit first-layer revisions stay within frozen candidate spans."""
    activity = base_activity.copy()
    previous_stop = 0
    for review in reviews:
        start, stop = review["start"], review["stop"]
        if start < previous_stop or not 0 <= start < stop <= len(activity):
            raise ValueError("Candidate reviews overlap or leave the calendar")
        revised = np.asarray(review["activity"], dtype=int)
        if revised.shape != activity.shape or not np.isin(revised, [-1, 0, 1]).all():
            raise ValueError("Review output has an invalid calendar or state")
        activity[start:stop] = revised[start:stop]
        previous_stop = stop
    activity[~observed] = -1
    prediction = ActivityPrediction(activity, np.where(activity == 0, 0, -1))
    prediction.validate(len(activity))
    return prediction
