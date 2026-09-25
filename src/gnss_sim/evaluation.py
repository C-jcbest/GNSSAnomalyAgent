"""P4 event-level evaluation against CaseTruth.events only."""

from __future__ import annotations

import json
from pathlib import Path

from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseTruth, DetectionResult, PredictedEvent

POINT_TOLERANCE = {"spike": 1, "step": 3}
INTERVAL_TYPES = {"slow_trend", "acceleration", "transient_shift"}
IOU_THRESHOLD = 0.5


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return overlap / union


def compatibility(truth_event, prediction: PredictedEvent) -> tuple[bool, float]:
    """Return compatibility and normalized quality; predicted type is deliberately ignored."""
    if prediction.axes != [truth_event.axis]:
        return False, 0.0
    if truth_event.type in POINT_TOLERANCE:
        tolerance = POINT_TOLERANCE[truth_event.type]
        error = abs(prediction.start_index - truth_event.start_index)
        if prediction.start_index != prediction.end_index or error > tolerance:
            return False, 0.0
        return True, 1 - error / (tolerance + 1)
    if truth_event.type in INTERVAL_TYPES:
        if prediction.start_index >= prediction.end_index:
            return False, 0.0
        iou = temporal_iou(truth_event.start_index, truth_event.end_index,
                           prediction.start_index, prediction.end_index)
        return iou >= IOU_THRESHOLD, iou
    raise ValueError(f"unknown truth event type: {truth_event.type}")


def match_events(truth: CaseTruth, predictions: list[PredictedEvent]) -> list[tuple[int, int]]:
    """Maximize cardinality, then match quality, independent of input ordering."""
    gt_order = sorted(range(len(truth.events)), key=lambda i: (
        truth.events[i].axis, truth.events[i].start_index, truth.events[i].end_index,
        truth.events[i].type, truth.events[i].event_id))
    pred_order = sorted(range(len(predictions)), key=lambda i: (
        tuple(predictions[i].axes), predictions[i].start_index,
        predictions[i].end_index, predictions[i].prediction_id))
    # The frozen pilot has at most six truth events. A GT-subset DP also handles
    # arbitrary numbers of predictions without letting duplicates inflate TP.
    if len(gt_order) > 20:
        raise ValueError("matching supports at most 20 ground-truth events")
    states: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {0: (0, ())}
    for pi in pred_order:
        next_states = states.copy()  # this prediction may remain unmatched
        for mask, (score, pairs) in states.items():
            for rank, gi in enumerate(gt_order):
                if mask & (1 << rank):
                    continue
                allowed, quality = compatibility(truth.events[gi], predictions[pi])
                if not allowed:
                    continue
                new_mask = mask | (1 << rank)
                candidate = (score + round(quality * 1_000_000_000), pairs + ((gi, pi),))
                prior = next_states.get(new_mask)
                if prior is None or candidate[0] > prior[0] or (
                    candidate[0] == prior[0] and candidate[1] < prior[1]
                ):
                    next_states[new_mask] = candidate
        states = next_states
    best_mask, (_, best_pairs) = max(
        states.items(), key=lambda item: (item[0].bit_count(), item[1][0],
                                          tuple(-i for pair in item[1][1] for i in pair)))
    assert len(best_pairs) == best_mask.bit_count()
    return sorted(best_pairs, key=lambda pair: (
        truth.events[pair[0]].axis, truth.events[pair[0]].start_index,
        truth.events[pair[0]].end_index, predictions[pair[1]].start_index))


def evaluate_case(truth: CaseTruth, result: DetectionResult, group: str) -> dict:
    if result.case_id != truth.case_id:
        raise ValueError("result case ID does not match truth")
    if group not in ("normal", "single", "multi"):
        raise ValueError("invalid pilot group")
    predictions = result.events if result.status == "success" else []
    if len({event.prediction_id for event in predictions}) != len(predictions):
        raise ValueError("prediction IDs must be unique within a case")
    pairs = match_events(truth, predictions)
    interval_pairs = [(gi, pi) for gi, pi in pairs
                      if truth.events[gi].type in INTERVAL_TYPES]
    onset_errors = [abs(truth.events[gi].start_index - predictions[pi].start_index)
                    for gi, pi in pairs]
    end_errors = [abs(truth.events[gi].end_index - predictions[pi].end_index)
                  for gi, pi in interval_pairs]
    ious = [temporal_iou(truth.events[gi].start_index, truth.events[gi].end_index,
                         predictions[pi].start_index, predictions[pi].end_index)
            for gi, pi in interval_pairs]
    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(truth.events) - tp
    return {
        "case_id": truth.case_id, "group": group, "status": result.status,
        "tp": tp, "fp": fp, "fn": fn,
        "matches": [{"truth_id": truth.events[gi].event_id,
                     "prediction_id": predictions[pi].prediction_id} for gi, pi in pairs],
        "onset_error_sum_days": sum(onset_errors), "matched_count": tp,
        "interval_iou_sum": sum(ious), "interval_end_error_sum_days": sum(end_errors),
        "matched_interval_count": len(interval_pairs),
    }


def aggregate_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("cannot aggregate zero cases")
    tp = sum(case["tp"] for case in cases)
    fp = sum(case["fp"] for case in cases)
    fn = sum(case["fn"] for case in cases)
    success = sum(case["status"] == "success" for case in cases)
    matched = sum(case["matched_count"] for case in cases)
    intervals = sum(case["matched_interval_count"] for case in cases)
    normal = [case for case in cases if case["group"] == "normal"]
    valid_normal = [case for case in normal if case["status"] == "success"]
    multi = [case for case in cases if case["group"] == "multi"]
    return {
        "cases": len(cases), "tp": tp, "fp": fp, "fn": fn,
        "event_precision": tp / (tp + fp) if tp + fp else None,
        "event_recall": tp / (tp + fn) if tp + fn else None,
        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "onset_mae_days": sum(c["onset_error_sum_days"] for c in cases) / matched
        if matched else None,
        "interval_mean_iou": sum(c["interval_iou_sum"] for c in cases) / intervals
        if intervals else None,
        "interval_end_mae_days": sum(c["interval_end_error_sum_days"] for c in cases)
        / intervals if intervals else None,
        "normal_far": sum(c["fp"] > 0 for c in valid_normal) / len(valid_normal)
        if valid_normal else None,
        "fp_per_normal": sum(c["fp"] for c in valid_normal) / len(valid_normal)
        if valid_normal else None,
        "normal_failure_rate": (len(normal) - len(valid_normal)) / len(normal)
        if normal else None,
        "multi_mean_case_recall": sum(c["tp"] / (c["tp"] + c["fn"])
                                      for c in multi) / len(multi) if multi else None,
        "multi_complete_case_rate": sum(c["fn"] == 0 and c["fp"] == 0
                                        for c in multi) / len(multi) if multi else None,
        "execution_success_rate": success / len(cases),
        "execution_failures": len(cases) - success,
    }


def evaluate_pilot(directory: Path, results: list[DetectionResult], method: str) -> dict:
    """Evaluate all 300 pilot cases; absent/failed outputs remain in the denominator."""
    manifest = verify_pilot(directory)
    by_id = {}
    for result in results:
        if result.method != method or result.case_id in by_id:
            raise ValueError("method mismatch or duplicate case result")
        by_id[result.case_id] = result
    expected = {entry["case_id"] for entry in manifest["cases"]}
    if by_id.keys() - expected:
        raise ValueError("result contains an unknown case ID")
    cases = []
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        truth = CaseTruth.model_validate_json(
            (directory / "cases" / case_id / "truth.json").read_bytes())
        result = by_id.get(case_id, DetectionResult(
            case_id=case_id, method=method, status="failed", events=[]))
        kind = entry["case_type"]
        group = "normal" if kind == "normal" else "multi" if truth.scenario_type else "single"
        cases.append(evaluate_case(truth, result, group))
    return {"pilot_id": manifest["pilot_id"], "method": method,
            "summary": aggregate_cases(cases), "cases": cases}


def load_results_jsonl(path: Path) -> tuple[list[DetectionResult], list[dict]]:
    """Malformed rows are recorded; their absent cases are scored as failures."""
    results = []
    errors = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(DetectionResult.model_validate(json.loads(line)))
        except (ValueError, TypeError) as exc:
            errors.append({"line": number, "error": type(exc).__name__})
    return results, errors
