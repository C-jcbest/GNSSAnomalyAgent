"""Range evaluation with unmodified Huet et al. affiliation source and GNSS masks."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from gnss_research.baselines import contiguous_runs
from gnss_research.metrics import counts_metrics, daily_metrics, match_events
from gnss_research.schema import Prediction, event_masks, observed_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/affiliation-metrics-py"))
from affiliation.generics import convert_vector_to_events  # noqa: E402
from affiliation.metrics import pr_from_events  # noqa: E402

RANGE_TYPES = ("transient", "deformation", "range_union")


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def affiliation(target: np.ndarray, predicted: np.ndarray, start: int) -> dict:
    truth_events = convert_vector_to_events(target.tolist())
    pred_events = convert_vector_to_events(predicted.tolist())
    if not truth_events:
        return {"precision": None, "recall": None, "f1": None, "empty_prediction": not pred_events}
    report = pr_from_events(pred_events, truth_events, (start, len(target)))
    precision, recall = report["precision"], report["recall"]
    # Preserve undefined author precision. Only our complete-station summary uses zero.
    valid_precision = precision if math.isfinite(precision) else None
    return {"precision": valid_precision, "recall": recall,
            "f1": harmonic(valid_precision or 0.0, recall), "empty_prediction": not pred_events}


def event_spans(prediction: Prediction, kinds: tuple[str, ...], eligible: np.ndarray,
                start: int) -> list[tuple[int, int]]:
    spans = []
    for event in prediction.events:
        if event.kind not in kinds or event.status != "confirmed" or event.end < start:
            continue
        left, right = max(start, event.start), event.end
        if eligible[left:right + 1].any():
            spans.append((left, right))
    # Merge only overlapping/adjacent semantic spans, without splitting at missing days.
    mask = np.zeros(len(eligible), dtype=bool)
    for left, right in spans:
        mask[left:right + 1] = True
    return contiguous_runs(mask)


def spans_mask(spans: list[tuple[int, int]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for left, right in spans:
        mask[left:right + 1] = True
    return mask


def overlap_quality(truth_spans, pred_spans, eligible, predicted) -> dict:
    best_ious = []
    coverages = []
    for left, right in truth_spans:
        truth_mask = np.zeros(len(eligible), dtype=bool)
        truth_mask[left:right + 1] = eligible[left:right + 1]
        coverages.append(float((truth_mask & predicted).sum() / truth_mask.sum()))
        best = 0.0
        for start, end in pred_spans:
            pred_mask = np.zeros(len(eligible), dtype=bool)
            pred_mask[start:end + 1] = eligible[start:end + 1]
            union = (truth_mask | pred_mask).sum()
            best = max(best, float((truth_mask & pred_mask).sum() / union))
        best_ious.append(best)
    return {"truth_best_ious": best_ious, "truth_coverages": coverages}


def evaluate(observation: dict, truth: dict, result: dict) -> dict | None:
    if result.get("status") == "failed":
        result = {**result, "prediction": {"events": []}}
    if result.get("prediction") is None:
        return None
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    start = observation["background_end"] + 1
    eligible = observed.copy()
    eligible[:start] = False
    reference = Prediction.model_validate({"events": truth["events"]})
    prediction = Prediction.model_validate(result["prediction"]).validate_length(len(observed))
    target_masks = event_masks(reference, eligible)
    pred_masks = event_masks(prediction, eligible)
    report = {}
    for kind in RANGE_TYPES:
        kinds = ("transient", "deformation") if kind == "range_union" else (kind,)
        target = np.any(np.column_stack([target_masks[name] for name in kinds]), axis=1)
        predicted = np.any(np.column_stack([pred_masks[name] for name in kinds]), axis=1)
        true_spans = event_spans(reference, kinds, eligible, start)
        pred_spans = event_spans(prediction, kinds, eligible, start)
        negative = eligible & ~target
        daily = daily_metrics(target[eligible], predicted[eligible])
        report[kind] = {
            "daily": daily,
            "affiliation": affiliation(target, predicted, start),
            "affiliation_gap_spans_sensitivity": affiliation(
                spans_mask(true_spans, len(eligible)), spans_mask(pred_spans, len(eligible)), start),
            "events": {f"iou_{threshold}": match_events(true_spans, pred_spans, eligible, threshold)
                       for threshold in (0.3, 0.5)},
            **overlap_quality(true_spans, pred_spans, eligible, predicted),
            "truth_events": len(true_spans), "predicted_events": len(pred_spans),
            "truth_observed_fragments": len(contiguous_runs(target)),
            "predicted_observed_fragments": len(contiguous_runs(predicted)),
            "negative_station": not target.any(), "eligible_days": int(eligible.sum()),
            "false_positive_days": daily["fp"],
            "false_positive_runs": len(contiguous_runs(predicted & negative))}
    return report


def mean_or_none(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def aggregate_affiliation(rows: list[dict], key: str) -> dict:
    positives = [row[key] for row in rows if row[key]["f1"] is not None]
    if not positives:
        return {"positive_stations": 0, "precision": None, "recall": None,
                "station_macro_f1": None, "harmonic_of_macro_pr": None}
    precision = float(np.mean([row["precision"] or 0.0 for row in positives]))
    recall = float(np.mean([row["recall"] for row in positives]))
    return {"positive_stations": len(positives), "precision": precision, "recall": recall,
            "station_macro_f1": float(np.mean([row["f1"] for row in positives])),
            "harmonic_of_macro_pr": harmonic(precision, recall),
            "defined_author_precision_mean": mean_or_none(
                row["precision"] for row in positives if row["precision"] is not None),
            "empty_prediction_stations": sum(row["empty_prediction"] for row in positives)}


def aggregate(rows: list[dict]) -> dict:
    report = {}
    for kind in RANGE_TYPES:
        typed = [row["metrics"][kind] for row in rows]
        events = {}
        for threshold in ("iou_0.3", "iou_0.5"):
            event_rows = [row["events"][threshold] for row in typed]
            pooled = counts_metrics(*(sum(row[key] for row in event_rows) for key in ("tp", "fp", "fn")))
            for boundary in ("start", "end"):
                key = f"matched_{boundary}_mae_days"
                pooled[key] = sum(row[key] * row["tp"] for row in event_rows if row["tp"]) / pooled["tp"] \
                    if pooled["tp"] else None
            events[threshold] = pooled
        negatives = [row for row in typed if row["negative_station"]]
        negative_days = sum(row["eligible_days"] for row in negatives)
        report[kind] = {
            "affiliation": aggregate_affiliation(typed, "affiliation"),
            "affiliation_gap_spans_sensitivity": aggregate_affiliation(typed, "affiliation_gap_spans_sensitivity"),
            "daily_micro": counts_metrics(*(sum(row["daily"][key] for row in typed)
                                             for key in ("tp", "fp", "fn"))),
            "events_micro": events,
            "truth_event_mean_best_iou": mean_or_none(x for row in typed for x in row["truth_best_ious"]),
            "truth_event_mean_coverage": mean_or_none(x for row in typed for x in row["truth_coverages"]),
            "truth_events": sum(row["truth_events"] for row in typed),
            "predicted_events": sum(row["predicted_events"] for row in typed),
            "predicted_observed_fragments": sum(row["predicted_observed_fragments"] for row in typed),
            "false_positive_days": sum(row["false_positive_days"] for row in typed),
            "negative_stations": len(negatives),
            "negative_station_fp_days": sum(row["false_positive_days"] for row in negatives),
            "negative_station_fp_runs": sum(row["false_positive_runs"] for row in negatives),
            "negative_station_fp_runs_per_station_year":
                sum(row["false_positive_runs"] for row in negatives) * 365.25 / negative_days
                if negative_days else None}
    type_scores = [report[kind]["affiliation"]["station_macro_f1"]
                   for kind in ("transient", "deformation")]
    report["range_types_affiliation_macro_f1"] = float(np.mean(type_scores)) \
        if all(value is not None for value in type_scores) else None
    return report
