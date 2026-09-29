"""Point and Range metrics against native per-axis labels."""
from __future__ import annotations

import numpy as np
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

from gnss_sim.schemas import PointResult, RangeResult

AXES = ("N", "E", "U")
DAYS = 365


def matched_count(actual: set[int], predicted: set[int], tolerance: int) -> int:
    """Maximum cardinality for a common symmetric date tolerance, within one axis."""
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    truth, pred = sorted(actual), sorted(predicted)
    i = j = matched = 0
    while i < len(truth) and j < len(pred):
        if pred[j] < truth[i] - tolerance:
            j += 1
        elif pred[j] > truth[i] + tolerance:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def range_days(intervals) -> set[int]:
    return {day for start, end in intervals for day in range(start, end + 1)}


def _affiliation(gt, pred):
    """Each observed day occupies [day, day + 1), including isolated point targets."""
    if not pred:
        return 0.0, 0.0, 0.0
    score = pr_from_events(
        convert_vector_to_events([int(i in pred) for i in range(DAYS)]),
        convert_vector_to_events([int(i in gt) for i in range(DAYS)]), Trange=(0, DAYS))
    p, r = float(score["precision"]), float(score["recall"])
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def scores(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None}


def target_days(truth, task, axis):
    events = [event for event in truth.events if event.task == task and event.axis == axis]
    if task == "point":
        return {event.start_index for event in events}
    return range_days((event.start_index, event.end_index) for event in events)


def validate_truth(truth):
    labels = np.zeros((365, 3), dtype=int)
    for event in truth.events:
        axis = AXES.index(event.axis)
        labels[event.start_index:event.end_index + 1, axis] = 1
    if not np.array_equal(labels, truth.axis_labels):
        raise ValueError("Event intervals disagree with native per-axis labels")
    if not np.array_equal(labels.max(axis=1), truth.native_labels):
        raise ValueError("Native time labels disagree with channel labels")


def evaluate(truths, predictions, task):
    if task not in ("point", "range") or set(predictions) - set(truths):
        raise ValueError("Invalid task or unknown prediction case")
    model = PointResult if task == "point" else RangeResult
    totals = [0, 0, 0]
    tolerance = [0, 0, 0]
    affiliation = []
    successful = normal_cases = normal_success = normal_alarm_cases = 0
    normal_alarm_axes = normal_days = negative_axes = negative_alarm_axes = failed_negative_axes = 0
    for cid, truth in truths.items():
        validate_truth(truth)
        row = predictions.get(cid)
        if row is not None and (not isinstance(row, model) or row.case_id != cid):
            raise ValueError("Prediction task or identity mismatch")
        valid = row is not None and row.status == "success"
        successful += valid
        normal = not truth.events
        normal_cases += normal
        normal_success += normal and valid
        normal_has_alarm = False
        for axis in AXES:
            gt = target_days(truth, task, axis)
            values = getattr(row.predictions, axis) if valid else []
            pred = set(values) if task == "point" else range_days(values)
            count = (len(gt & pred), len(pred - gt), len(gt - pred))
            totals = [a + b for a, b in zip(totals, count)]
            if not gt:
                negative_axes += valid
                negative_alarm_axes += valid and bool(pred)
                failed_negative_axes += not valid
            if normal and valid:
                normal_alarm_axes += bool(pred)
                normal_has_alarm |= bool(pred)
                normal_days += len(pred)
            if task == "point":
                hits = matched_count(gt, pred, 3)
                tolerance = [a + b for a, b in zip(tolerance,
                            (hits, len(pred) - hits, len(gt) - hits))]
            if gt:
                affiliation.append(_affiliation(gt, pred))
        normal_alarm_cases += normal and valid and normal_has_alarm
    report = {"task": task, "cases": len(truths), "successful_cases": successful,
              "failed_cases": len(truths) - successful, "daily": scores(*totals),
              "normal": {"cases": normal_cases, "successful_cases": normal_success,
                         "failed_cases": normal_cases - normal_success,
                         "alarm_cases": normal_alarm_cases, "alarm_axes": normal_alarm_axes,
                         "predicted_axis_days": normal_days,
                         "valid_axes": normal_success * 3},
              "negative_axes": {"valid": negative_axes, "alarms": negative_alarm_axes,
                                "failed": failed_negative_axes}}
    if task == "point":
        report["tolerance_3d"] = scores(*tolerance)
    report["affiliation"] = {"positive_axes": len(affiliation), **{
        name: float(np.mean([x[i] for x in affiliation])) if affiliation else None
        for i, name in enumerate(("precision", "recall", "f1"))}}
    return report
