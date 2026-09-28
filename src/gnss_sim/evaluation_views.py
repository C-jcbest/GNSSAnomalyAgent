"""Versioned companion metrics. P4 evaluation.py and predictions remain unchanged."""

from __future__ import annotations

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

from gnss_sim.evaluation import AXES, DAYS, POINT_TYPES, RANGE_TYPES
from gnss_sim.schemas import PointResult, RangeResult

PROTOCOL = "comparison-views-v1"


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


def point_scores(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _report_counts(counts: list[int]) -> dict:
    scores = point_scores(*counts)
    if counts[0] + counts[2] == 0:
        scores.update(precision=None, recall=None, f1=None)
    return scores


def _mean(values):
    return sum(values) / len(values) if values else None


def _affiliation(gt, pred):
    if not pred:
        return 0.0, 0.0, 0.0
    score = pr_from_events(
        convert_vector_to_events([int(i in pred) for i in range(DAYS)]),
        convert_vector_to_events([int(i in gt) for i in range(DAYS)]), Trange=(0, DAYS))
    p, r = float(score["precision"]), float(score["recall"])
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def evaluate_views(truths: dict, results: dict, task: str, axes=AXES) -> dict:
    """Failures remain in the positive denominator; negative FAR uses successful axes."""
    if task not in ("point", "range") or not axes or set(axes) - set(AXES):
        raise ValueError("Unknown task or axes")
    if set(results) - set(truths):
        raise ValueError("Unknown result case")
    model = PointResult if task == "point" else RangeResult
    if any(not isinstance(row, model) for row in results.values()):
        raise ValueError("Result schema does not match task")
    counts = {tol: [0, 0, 0] for tol in (0, 1, 3, 7)}
    daily = [0, 0, 0]
    success = negative = alarms = failed_negative = 0
    normal_cases = normal_success = normal_axes = normal_alarms = normal_days = 0
    aff, iou, positive_daily = [], [], []
    for case_id, truth in truths.items():
        row = results.get(case_id)
        if row is not None and row.case_id != case_id:
            raise ValueError("Case identity mismatch")
        valid = row is not None and row.status == "success"
        success += valid
        normal_cases += not truth.events
        normal_success += not truth.events and valid
        for axis in axes:
            events = [event for event in truth.events if event.axis == axis
                      and event.type in (POINT_TYPES if task == "point" else RANGE_TYPES)]
            gt = ({event.start_index for event in events} if task == "point" else
                  range_days((event.start_index, event.end_index) for event in events))
            values = getattr(row.predictions, axis) if valid else []
            pred = set(values) if task == "point" else range_days(values)
            if not gt:
                negative += valid
                alarms += valid and bool(pred)
                failed_negative += not valid
            if not truth.events and valid:
                normal_axes += 1
                normal_alarms += bool(pred)
                normal_days += len(pred)
            if task == "point":
                for tol in counts:
                    hits = matched_count(gt, pred, tol)
                    for k, value in enumerate((hits, len(pred) - hits, len(gt) - hits)):
                        counts[tol][k] += value
            else:
                hits = len(gt & pred)
                values = (hits, len(pred - gt), len(gt - pred))
                for k, value in enumerate(values):
                    daily[k] += value
                if gt:
                    aff.append(_affiliation(gt, pred))
                    iou.append(hits / len(gt | pred))
                    positive_daily.append(point_scores(*values))
    report = {"protocol": PROTOCOL, "task": task, "cases": len(truths),
              "successful_cases": success,
              "execution_success_rate": success / len(truths) if truths else None,
              "negative_axes": {"valid": negative, "alarms": alarms,
                                "failed": failed_negative,
                                "far": alarms / negative if negative else None},
              "normal": {"cases": normal_cases, "successful_cases": normal_success,
                         "failure_rate": 1 - normal_success / normal_cases if normal_cases else None,
                         "valid_axes": normal_axes, "alarm_axes": normal_alarms,
                         "axis_far": normal_alarms / normal_axes if normal_axes else None,
                         "predicted_days": normal_days}}
    if task == "point":
        report["point_exact"] = _report_counts(counts[0])
        report["point_tolerance_3d"] = _report_counts(counts[3])
        report["sensitivity"] = {str(tol): _report_counts(counts[tol]) for tol in (1, 7)}
    else:
        report["range_affiliation"] = {"positive_axes": len(aff), **{
            key: _mean([item[k] for item in aff])
            for k, key in enumerate(("precision", "recall", "f1"))}}
        report["range_daily_positive_axes"] = {"positive_axes": len(aff), "mean_iou": _mean(iou),
            **{key: _mean([item[key] for item in positive_daily])
               for key in ("precision", "recall", "f1")}}
        report["range_daily_all_axes_micro"] = _report_counts(daily)
    return report
