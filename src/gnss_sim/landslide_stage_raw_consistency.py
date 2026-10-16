"""Check claimed stages against activity-contained raw displacement contrasts."""

from dataclasses import dataclass

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction, runs


@dataclass(frozen=True)
class StageContrastPolicy:
    separations: tuple[int, ...] = (21, 42)
    relative_change: float = 0.25
    noise_multiplier: float = 2.0
    minimum_conflict_days: int = 14


def raw_stage_contrasts(case, activity, noise, separation):
    """Compare successive raw-median displacement rates without derivative caches.

    Every observed day in the evidence envelope must already be confirmed active.
    Missing days retain their calendar positions and cannot receive a stage. The
    noise margin is descriptive: correlated GNSS errors preclude a confidence claim.
    """
    values = np.asarray([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ])
    observed = np.isfinite(values).all(axis=1)
    activity = np.asarray(activity)
    if activity.shape != observed.shape or separation < 14:
        raise ValueError("Raw stage evidence requires an aligned activity calendar")
    result = np.full((len(values), 3), np.nan)
    for day in range(separation + 7, len(values) - separation - 7):
        if activity[day] != 1 or not observed[day]:
            continue
        left, right = day - separation - 7, day + separation + 7
        if np.any(observed[left:right] & (activity[left:right] != 1)):
            continue
        observed_positions = np.flatnonzero(np.r_[True, observed[left:right], True])
        if np.max(np.diff(observed_positions) - 1) > 7:
            continue
        medians = []
        for center in (day - separation, day, day + separation):
            block = values[center - 7:center + 7]
            complete = block[np.isfinite(block).all(axis=1)]
            if len(complete) < 8:
                break
            medians.append(np.median(complete, axis=0))
        if len(medians) != 3:
            continue
        before, after = medians[1] - medians[0], medians[2] - medians[1]
        if np.dot(before, after) <= 0:
            continue
        before_rate = np.linalg.norm(before) / separation
        after_rate = np.linalg.norm(after) / separation
        result[day] = (before_rate, after_rate, np.linalg.norm(noise) / separation)
    return result


def stage_conflict_mask(stage, contrasts, policy):
    """Flag opposite A/D evidence or materially changing rates inside an S claim.

    This never assigns a replacement class. An unavailable or inconclusive
    contrast is not a contradiction and must not silently reject weak motion.
    """
    stage = np.asarray(stage)
    masks = []
    for separation in policy.separations:
        before, after, noise = np.asarray(contrasts[separation]).T
        change = after - before
        margin = np.maximum(policy.relative_change * np.maximum(before, after),
                            policy.noise_multiplier * noise)
        increasing = change > margin
        decreasing = change < -margin
        masks.append(((stage == 1) & decreasing) | ((stage == 3) & increasing)
                     | ((stage == 2) & (increasing | decreasing)))
    conflicts = np.logical_and.reduce(masks)
    # A short fluctuation does not warrant review. Keep the exact contradictory
    # dates within each existing stage span rather than relabelling the whole span.
    retained = np.zeros(len(stage), dtype=bool)
    for code in (1, 2, 3):
        for left, right in runs(stage == code):
            if conflicts[left:right].sum() >= policy.minimum_conflict_days:
                retained[left:right] = conflicts[left:right]
    return retained


def abstain_on_stage_conflicts(prediction, conflicts):
    stage = prediction.stage.copy()
    stage[conflicts] = -1
    result = ActivityPrediction(prediction.activity.copy(), stage,
                                stage_status=prediction.stage_status)
    result.validate(len(stage))
    return result
