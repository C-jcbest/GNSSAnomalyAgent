"""Offline activity/stage contracts. Unknown is never counted as a true negative."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAGES = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}


@dataclass
class ActivityPrediction:
    activity: np.ndarray  # -1 unknown, 0 not confirmed, 1 confirmed
    stage: np.ndarray  # -1 unknown, 0 none, 1 acceleration, 2 steady, 3 deceleration
    status: str = "success"
    gated: bool = True
    stage_status: str = "success"

    def validate(self, days: int):
        if self.activity.shape != (days,) or self.stage.shape != (days,):
            raise ValueError("Prediction length differs from the observation calendar")
        if not np.isin(self.activity, [-1, 0, 1]).all():
            raise ValueError("Invalid activity code")
        if not np.isin(self.stage, [-1, 0, 1, 2, 3]).all():
            raise ValueError("Invalid stage code")
        if self.status not in {"success", "failed"}:
            raise ValueError("Invalid prediction status")
        if self.stage_status not in {"success", "failed"}:
            raise ValueError("Invalid stage status")
        if self.stage_status == "failed" and np.any(self.stage != -1):
            raise ValueError("Failed stage predictions must remain unknown")
        if self.gated and np.any((self.stage > 0) & (self.activity != 1)):
            raise ValueError("A definite stage requires confirmed displacement")
        if self.status == "failed" and (np.any(self.activity != -1) or np.any(self.stage != -1)):
            raise ValueError("Failed predictions must remain unknown")

    def to_dict(self):
        return {"activity": self.activity.tolist(), "stage": self.stage.tolist(),
                "status": self.status, "gated": self.gated, "stage_status": self.stage_status}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs, preserving the last sample."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def event_masks(activity: np.ndarray) -> list[np.ndarray]:
    """Bridge at most seven unknown days; never bridge a predicted negative day."""
    mask = activity == 1
    for start, stop in runs(activity == -1):
        if stop - start <= 7 and start > 0 and stop < len(mask) and mask[start - 1] and mask[stop]:
            mask[start:stop] = True
    events = []
    for start, stop in runs(mask):
        event = np.zeros(len(mask), dtype=bool)
        event[start:stop] = True
        events.append(event)
    return events


def match_events(actual, predicted, known, threshold=0.3):
    """Maximum-cardinality one-to-one IoU matching on known annotation days."""
    adjacency = []
    for target in actual:
        neighbors = []
        for index, candidate in enumerate(predicted):
            union = np.count_nonzero((target | candidate) & known)
            overlap = np.count_nonzero(target & candidate & known)
            if union and overlap / union >= threshold:
                neighbors.append(index)
        adjacency.append(neighbors)
    owner = {}

    def augment(target_index, visited):
        for candidate_index in adjacency[target_index]:
            if candidate_index in visited:
                continue
            visited.add(candidate_index)
            if candidate_index not in owner or augment(owner[candidate_index], visited):
                owner[candidate_index] = target_index
                return True
        return False

    for target_index in range(len(actual)):
        augment(target_index, set())
    return len(owner)


def score_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "f1": float(2 * tp / denominator) if denominator else None}


def evaluate_case(rows, prediction: ActivityPrediction, iou_threshold=0.3):
    prediction.validate(len(rows))
    actual = np.array([int(row["activity_label"]) for row in rows])
    predicted = prediction.activity
    known = actual >= 0
    positive = actual == 1
    negative = actual == 0
    weak = np.array([row["feature"] == "slow_displacement" for row in rows])
    target_stage = np.array([STAGES.get(row["feature"], -1) for row in rows])
    target_stage[negative] = 0
    stage_known = target_stage >= 0
    episodes = sorted({row["episode_id"] for row in rows if int(row["activity_label"]) == 1})
    actual_events = [np.array([row["episode_id"] == episode for row in rows]) & positive
                     for episode in episodes]
    all_predicted_events = event_masks(predicted)
    predicted_events = [event for event in all_predicted_events if np.any(event & known)]
    matched = match_events(actual_events, predicted_events, known, iou_threshold)
    stages = {}
    for name, code in STAGES.items():
        stages[name] = score_counts(
            np.count_nonzero((prediction.stage == code) & (target_stage == code)),
            np.count_nonzero((prediction.stage == code) & (target_stage != code) & stage_known),
            np.count_nonzero((prediction.stage != code) & (target_stage == code)),
        )
    return {
        "daily": score_counts(np.count_nonzero((predicted == 1) & positive),
                              np.count_nonzero((predicted == 1) & negative),
                              np.count_nonzero((predicted != 1) & positive)),
        "events": score_counts(matched, len(predicted_events) - matched, len(actual_events) - matched),
        "stages": stages,
        "weak_days": int(weak.sum()), "weak_detected_days": int(np.count_nonzero(weak & (predicted == 1))),
        "known_days": int(known.sum()), "unscored_label_days": int((~known).sum()),
        "abstained_known_days": int(np.count_nonzero(known & (predicted == -1))),
        "negative_days": int(negative.sum()),
        "decided_negative_days": int(np.count_nonzero(negative & (predicted >= 0))),
        "true_negative_days": int(np.count_nonzero(negative & (predicted == 0))),
        "false_acceleration_days": int(np.count_nonzero(negative & (prediction.stage == 1))),
        "unscored_prediction_events": len(all_predicted_events) - len(predicted_events),
        "negative_exposure_alarm_events": sum(bool(np.any(event & negative)) for event in predicted_events),
        "failed_cases": int(prediction.status == "failed"), "cases": 1,
        "stage_failed_cases": int(prediction.status == "failed" or prediction.stage_status == "failed"),
        "stage_known_days": int(stage_known.sum()),
        "stage_abstained_known_days": int(np.count_nonzero(stage_known & (prediction.stage == -1))),
    }


def aggregate_scores(case_scores):
    if not case_scores:
        raise ValueError("At least one case is required")
    result = {}
    for field in ("daily", "events"):
        result[field] = score_counts(*(sum(row[field][key] for row in case_scores)
                                       for key in ("tp", "fp", "fn")))
    result["stages"] = {
        name: score_counts(*(sum(row["stages"][name][key] for row in case_scores)
                             for key in ("tp", "fp", "fn"))) for name in STAGES
    }
    for field, value in case_scores[0].items():
        if isinstance(value, int):
            result[field] = sum(row[field] for row in case_scores)
    values = [value["f1"] for value in result["stages"].values() if value["f1"] is not None]
    result["stage_macro_f1"] = float(np.mean(values)) if values else None
    result["weak_recall"] = (result["weak_detected_days"] / result["weak_days"]
                             if result["weak_days"] else None)
    result["coverage"] = 1 - result["abstained_known_days"] / result["known_days"]
    # The rate is conditional on decisions; the missing exposure remains explicitly reported.
    years = result["decided_negative_days"] / 365.25
    result["negative_alarm_events_per_decided_station_year"] = (
        result["negative_exposure_alarm_events"] / years if years else None
    )
    return result
