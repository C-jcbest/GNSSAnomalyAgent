"""Validate novel visual intervals against preceding raw displacement direction."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import event_masks, runs


def raw_block_changes(case, separation=42):
    """Return two successive NEU changes from the existing disjoint14-day blocks."""
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm], dtype=float)
    before = np.full(values.shape, np.nan)
    after = np.full(values.shape, np.nan)
    for day in range(separation + 7, len(values) - separation - 7):
        medians = []
        for center in (day - separation, day, day + separation):
            sample = values[center - 7:center + 7]
            complete = sample[np.isfinite(sample).all(axis=1)]
            if len(complete) < 8:
                break
            medians.append(np.median(complete, axis=0))
        if len(medians) == 3:
            before[day] = medians[1] - medians[0]
            after[day] = medians[2] - medians[1]
    return values, before, after


def strong_direction_anchors(before, after, observed, noise, rate=.12, snr=3.0):
    """Estimate directions only from sustained calibrated raw42 strong support.

    These are numerical anchors, not independently confirmed slope activity.
    Their complete observation footprint must precede a proposal. A single step
    and a rotating anchor cannot by themselves establish a usable direction.
    """
    lengths = np.column_stack((np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)))
    consistency = np.sum(before * after, axis=1) / np.maximum(lengths.prod(axis=1), 1e-8)
    signal = np.minimum(np.linalg.norm(before / noise, axis=1), np.linalg.norm(after / noise, axis=1))
    strong = observed & (lengths.min(axis=1) / 42 >= rate) & (signal >= snr) & (consistency > 0)
    anchors = []
    for start, stop in runs(strong):
        if stop - start < 14:
            continue
        changes = np.vstack((before[start:stop], after[start:stop]))
        unit_changes = changes / np.linalg.norm(changes, axis=1)[:, None]
        direction = np.median(unit_changes, axis=0)
        norm = np.linalg.norm(direction)
        if norm <= 1e-8:
            continue
        direction /= norm
        # Sixty degrees is a fixed geometric tolerance, not a fitted GNSS limit.
        if np.median(unit_changes @ direction) < .5:
            continue
        anchors.append({"start": start, "stop": stop, "available_from": stop + 49,
                        "direction": direction.tolist()})
    return anchors


def preceding_anchor(anchors, start):
    eligible = [row for row in anchors if row["available_from"] <= start]
    return eligible[-1] if eligible else None


def interval_motion(values, observed, start, stop):
    """Check a whole proposed interval; never turn rolling support into boundaries."""
    if stop - start < 42:
        return None, "short_proposal"
    if any(right - left > 7 for left, right in runs(~observed[start:stop])):
        return None, "long_missing_gap"
    middle = (start + stop) // 2
    blocks = [(start, start + 14), (middle - 7, middle + 7), (stop - 14, stop)]
    medians = []
    for left, right in blocks:
        sample = values[left:right][observed[left:right]]
        if len(sample) < 8:
            return None, "insufficient_observations"
        medians.append(np.median(sample, axis=0))
    spacings = np.diff([(left + right - 1) / 2 for left, right in blocks])
    return (np.diff(medians, axis=0), spacings), "available"


def review_directional_additions(initial, reviewed, scope, observed, values, noise, anchors,
                                variant="direction_and_motion"):
    """Keep initial positives, and admit complete novel visual proposals or abstain.

    Only short missing gaps are grouped for proposal evidence. No gap receives
    a label, no known negative gap is joined, and no proposal is extended.
    Turning or absent anchor evidence leaves novel dates unknown. Numerical
    evidence alone cannot create activity: every addition needs local vision.
    """
    variants = {"direction_and_motion", "without_direction", "without_motion", "visual_only"}
    if variant not in variants:
        raise ValueError("Unknown direction review variant")
    for prediction in (initial, reviewed):
        if prediction.shape != observed.shape or not np.isin(prediction, [-1, 0, 1]).all():
            raise ValueError("Direction review requires aligned activity states")
        if np.any(prediction[~observed] != -1):
            raise ValueError("Missing dates must remain unknown")
    if scope.shape != observed.shape or scope.dtype != bool or np.any(scope & ~observed):
        raise ValueError("Direction review requires observed-only authority")
    activity = initial.copy()
    eligible = scope & (initial != 1)
    activity[eligible] = -1
    proposal_states = np.where(eligible & (reviewed == 1), 1, 0)
    proposal_states[~observed] = -1
    decisions = []
    for mask in event_masks(proposal_states):
        indices = np.flatnonzero(mask)
        start, stop = int(indices[0]), int(indices[-1]) + 1
        decision = {"start": start, "stop": stop, "accepted": False}
        evidence, reason = interval_motion(values, observed, start, stop)
        decision["reason"] = reason
        if evidence is None:
            decisions.append(decision)
            continue
        changes, spacings = evidence
        if variant == "visual_only":
            accepted = True
        else:
            anchor = preceding_anchor(anchors, start)
            decision["anchor"] = anchor
            if variant != "without_direction" and anchor is None:
                decision["reason"] = "no_preceding_anchor"
                decisions.append(decision)
                continue
            if variant == "without_direction":
                direction = changes.sum(axis=0)
                direction /= max(np.linalg.norm(direction), 1e-8)
            else:
                direction = np.asarray(anchor["direction"])
            cosines = changes @ direction / np.maximum(np.linalg.norm(changes, axis=1), 1e-8)
            projections = changes @ direction
            rates = projections / spacings
            projected_noise = np.linalg.norm(noise * direction)
            signals = projections / projected_noise
            direction_pass = bool(np.all(cosines >= .5))
            motion_pass = bool(np.all(rates >= .03) and np.all(signals >= 2))
            accepted = direction_pass and (motion_pass or variant == "without_motion")
            decision.update({"cosines": cosines.tolist(), "projected_rates": rates.tolist(),
                             "projected_noise_ratios": signals.tolist(),
                             "direction_pass": direction_pass, "motion_pass": motion_pass})
            decision["reason"] = "accepted" if accepted else "conflicting_raw_evidence"
        decision["accepted"] = accepted
        if accepted:
            activity[mask & eligible & (reviewed == 1)] = 1
        decisions.append(decision)
    return activity, decisions


def direction_information(case, features, anchors, before, after):
    """Expose the same preceding anchor information to numerical/ML controls."""
    extra = np.full((len(case.dates), 11), np.nan)
    for day in range(len(case.dates)):
        anchor = preceding_anchor(anchors, day)
        if anchor is None:
            continue
        direction = np.asarray(anchor["direction"])
        changes = np.vstack((before[day], after[day]))
        cosines = changes @ direction / np.maximum(np.linalg.norm(changes, axis=1), 1e-8)
        projections = changes @ direction
        signal = projections / np.linalg.norm(features.noise * direction)
        extra[day] = [*cosines, *(projections / 42), *signal,
                      day - anchor["available_from"], 1, *direction]
    return extra
