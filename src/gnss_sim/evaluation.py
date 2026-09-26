"""Point and range evaluation for the frozen P4 development pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult

Task = Literal["point", "range"]
AXES = ("N", "E", "U")
DAYS = 365
POINT_TYPES = {"spike", "step"}
RANGE_TYPES = {"slow_trend", "acceleration", "transient_shift"}


def _point_vector(days: list[int]) -> list[int]:
    vector = [0] * DAYS
    for day in days:
        vector[day] = 1
    return vector


def _range_vector(intervals: list[tuple[int, int]]) -> list[int]:
    vector = [0] * DAYS
    for start, end in intervals:
        vector[start:end + 1] = [1] * (end - start + 1)
    return vector


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _point_case(truth: CaseTruth, result: PointResult | None) -> tuple[int, int, int, int, int]:
    """Return TP/FP/FN and successful negative/false-alarm axis counts."""
    tp = fp = fn = negative = alarms = 0
    successful = result is not None and result.status == "success"
    for axis in AXES:
        gt = _point_vector([event.start_index for event in truth.events
                            if event.type in POINT_TYPES and event.axis == axis])
        pred = _point_vector(getattr(result.predictions, axis)) if successful else [0] * DAYS
        tp += sum(actual and detected for actual, detected in zip(gt, pred))
        fp += sum(not actual and detected for actual, detected in zip(gt, pred))
        fn += sum(actual and not detected for actual, detected in zip(gt, pred))
        if not any(gt) and successful:
            negative += 1
            alarms += int(any(pred))
    return tp, fp, fn, negative, alarms


def _range_case(
    truth: CaseTruth, result: RangeResult | None
) -> tuple[list[tuple[float, float, float]], int, int]:
    """Return affiliation scores for positive axes and negative-axis FAR counts."""
    scores = []
    negative = alarms = 0
    successful = result is not None and result.status == "success"
    for axis in AXES:
        gt = _range_vector([(event.start_index, event.end_index) for event in truth.events
                            if event.type in RANGE_TYPES and event.axis == axis])
        pred = (_range_vector(getattr(result.predictions, axis)) if successful
                else [0] * DAYS)
        if not any(gt):
            if successful:
                negative += 1
                alarms += int(any(pred))
            continue
        if not any(pred):
            scores.append((0.0, 0.0, 0.0))
            continue
        aff = pr_from_events(
            convert_vector_to_events(pred),
            convert_vector_to_events(gt),
            Trange=(0, DAYS),
        )
        precision = float(aff["precision"])
        recall = float(aff["recall"])
        scores.append((precision, recall, _f1(precision, recall)))
    return scores, negative, alarms


def evaluate_pilot(
    directory: Path, results: list[PointResult] | list[RangeResult], method: str, task: Task
) -> dict:
    """Score every pilot case; absent or failed outputs remain in the fixed case set."""
    if task not in ("point", "range"):
        raise ValueError("task must be point or range")
    manifest = verify_pilot(directory)
    result_type = PointResult if task == "point" else RangeResult
    by_id = {}
    for result in results:
        if not isinstance(result, result_type):
            raise ValueError("result schema does not match task")
        if result.method != method or result.case_id in by_id:
            raise ValueError("method mismatch or duplicate case result")
        by_id[result.case_id] = result
    expected = {entry["case_id"] for entry in manifest["cases"]}
    if by_id.keys() - expected:
        raise ValueError("result contains an unknown case ID")

    success = negative = alarms = tp = fp = fn = 0
    scores: list[tuple[float, float, float]] = []
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        truth = CaseTruth.model_validate_json(
            (directory / "cases" / case_id / "truth.json").read_bytes())
        result = by_id.get(case_id)
        success += int(result is not None and result.status == "success")
        if task == "point":
            case_tp, case_fp, case_fn, case_negative, case_alarms = _point_case(truth, result)
            tp += case_tp
            fp += case_fp
            fn += case_fn
        else:
            case_scores, case_negative, case_alarms = _range_case(truth, result)
            scores.extend(case_scores)
        negative += case_negative
        alarms += case_alarms

    far = alarms / negative if negative else None
    execution_success_rate = success / len(manifest["cases"])
    if task == "point":
        return {"task": task,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "far": far, "execution_success_rate": execution_success_rate}
    return {"task": task,
            "affiliation_precision": sum(score[0] for score in scores) / len(scores)
            if scores else 0.0,
            "affiliation_recall": sum(score[1] for score in scores) / len(scores)
            if scores else 0.0,
            "affiliation_f1": sum(score[2] for score in scores) / len(scores)
            if scores else 0.0,
            "far": far, "execution_success_rate": execution_success_rate}


def load_results_jsonl(
    path: Path, task: Task
) -> tuple[list[PointResult] | list[RangeResult], list[dict]]:
    """Skip malformed rows and report their line numbers; their cases become failures."""
    if task not in ("point", "range"):
        raise ValueError("task must be point or range")
    model = PointResult if task == "point" else RangeResult
    results = []
    errors = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(model.model_validate(json.loads(line)))
        except (ValueError, TypeError) as exc:
            errors.append({"line": number, "error": type(exc).__name__})
    return results, errors
