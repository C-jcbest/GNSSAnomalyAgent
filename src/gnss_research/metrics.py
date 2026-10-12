"""Unadjusted daily scores and one-to-one event matching on observed dates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

from gnss_research.baselines import contiguous_runs
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks


def counts_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision,
            "recall": recall, "f1": f1}


def daily_metrics(target: np.ndarray, predicted: np.ndarray, scores=None) -> dict:
    result = counts_metrics(int(np.sum(target & predicted)), int(np.sum(~target & predicted)),
                            int(np.sum(target & ~predicted)))
    if scores is not None:
        result["average_precision"] = float(average_precision_score(target, scores)) \
            if target.any() else None
    return result


def match_events(truth: list[tuple[int, int]], predictions: list[tuple[int, int]],
                 observed: np.ndarray, iou_threshold: float, point_tolerance: int = 0) -> dict:
    truth = [(a, b) for a, b in truth if observed[a:b + 1].any()]
    predictions = [(a, b) for a, b in predictions if observed[a:b + 1].any()]
    similarities = np.zeros((len(truth), len(predictions)))
    for i, (start, end) in enumerate(truth):
        for j, (left, right) in enumerate(predictions):
            if start == end and left == right and abs(start - left) <= point_tolerance:
                similarities[i, j] = 1.0
                continue
            overlap = int(observed[max(start, left):min(end, right) + 1].sum()) \
                if max(start, left) <= min(end, right) else 0
            union = int(observed[start:end + 1].sum() + observed[left:right + 1].sum()) - overlap
            similarities[i, j] = overlap / union if union else 0
    eligible = similarities >= iou_threshold
    # Maximize match count first, then IoU. A greedy match can underestimate event recall.
    weight = eligible * (min(len(truth), len(predictions)) + 1 + similarities)
    rows, columns = linear_sum_assignment(weight, maximize=True)
    matches = [(i, j) for i, j in zip(rows, columns) if eligible[i, j]]
    result = counts_metrics(len(matches), len(predictions) - len(matches), len(truth) - len(matches))
    result["matched_start_mae_days"] = float(np.mean([
        abs(truth[i][0] - predictions[j][0]) for i, j in matches])) if matches else None
    result["matched_end_mae_days"] = float(np.mean([
        abs(truth[i][1] - predictions[j][1]) for i, j in matches])) if matches else None
    return result


def evaluate_record(truth: dict, result: dict, observed: np.ndarray, background_end: int) -> dict:
    eligible = observed.copy()
    eligible[:background_end + 1] = False
    reference = Prediction.model_validate({"events": truth["events"]})
    target_masks = event_masks(reference, observed)
    generic_target = np.any(np.column_stack(list(target_masks.values())), axis=1)
    generic_predicted = np.array(result["generic_mask"], dtype=bool)
    if generic_predicted.shape != observed.shape:
        raise ValueError("generic prediction does not preserve daily index length")
    scores = np.array(result["generic_score"]) if result.get("generic_score") is not None else None
    generic = daily_metrics(generic_target[eligible], generic_predicted[eligible],
                            scores[eligible] if scores is not None else None)
    negative = ~generic_target & eligible
    false_events = len(contiguous_runs(generic_predicted & negative))
    report = {"generic_daily": generic, "observed_test_days": int(eligible.sum()),
              "missing_test_days": int((~observed[background_end + 1:]).sum()),
              "false_alarm_runs": false_events, "normal_exposure_days": int(negative.sum()),
              "false_alarm_runs_per_normal_station_year": false_events * 365.25 / negative.sum()
              if negative.any() else None, "types": None,
              "normal_station": not generic_target[eligible].any()}
    if result.get("prediction") is None:
        return report
    predicted = Prediction.model_validate(result["prediction"]).validate_length(len(observed))
    masks = event_masks(predicted, observed)
    types = {}
    for kind in EVENT_TYPES:
        true_spans = [(event.start, event.end) for event in reference.events if event.kind == kind]
        spans = [(max(event.start, background_end + 1), event.end) for event in predicted.events
                 if event.kind == kind and event.status == "confirmed"
                 and event.end > background_end]
        type_scores = result.get("type_scores", {}).get(kind)
        daily = daily_metrics(target_masks[kind][eligible], masks[kind][eligible],
                              np.array(type_scores)[eligible] if type_scores is not None else None)
        event_report = {f"iou_{threshold}": match_events(true_spans, spans, eligible, threshold)
                        for threshold in (0.3, 0.5)}
        if kind in ("spike", "step"):
            event_report["tolerance_3_days_secondary"] = match_events(
                true_spans, spans, eligible, 0.5, point_tolerance=3)
        types[kind] = {"daily": daily, "events": event_report}
    report["types"] = types
    report["uncertain_events"] = sum(event.status == "uncertain" for event in predicted.events)
    return report


def aggregate(rows: list[dict]) -> dict:
    daily = counts_metrics(*(sum(row["generic_daily"][key] for row in rows)
                             for key in ("tp", "fp", "fn")))
    exposure = sum(row["normal_exposure_days"] for row in rows)
    false_runs = sum(row["false_alarm_runs"] for row in rows)
    summary = {"stations": len(rows), "generic_daily_micro": daily,
               "generic_daily_station_macro_f1": float(np.mean([
                   row["generic_daily"]["f1"] for row in rows])),
               "false_alarm_runs_per_normal_station_year": false_runs * 365.25 / exposure
               if exposure else None}
    normal = [row for row in rows if row["normal_station"]]
    normal_days = sum(row["observed_test_days"] for row in normal)
    summary["normal_stations"] = len(normal)
    summary["false_alarm_runs_per_normal_station_year_controls"] = sum(
        row["false_alarm_runs"] for row in normal) * 365.25 / normal_days if normal_days else None
    typed = [row for row in rows if row["types"] is not None]
    if typed:
        summary["types"] = {}
        for kind in EVENT_TYPES:
            type_daily = counts_metrics(*(sum(row["types"][kind]["daily"][key] for row in typed)
                                          for key in ("tp", "fp", "fn")))
            events = {}
            for threshold in ("iou_0.3", "iou_0.5"):
                events[threshold] = counts_metrics(*(sum(
                    row["types"][kind]["events"][threshold][key] for row in typed)
                    for key in ("tp", "fp", "fn")))
            summary["types"][kind] = {"daily_micro": type_daily, "events_micro": events}
    return summary


def paired_bootstrap(left: list[dict], right: list[dict], seed: int, repeats: int = 2000,
                     endpoint: str = "generic") -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("paired station reports required")
    if endpoint == "generic":
        differences = np.array([a["generic_daily"]["f1"] - b["generic_daily"]["f1"]
                                for a, b in zip(left, right)])
    elif endpoint == "deformation":
        differences = np.array([a["types"]["deformation"]["daily"]["f1"]
                                - b["types"]["deformation"]["daily"]["f1"]
                                for a, b in zip(left, right)])
    else:
        raise ValueError("unknown bootstrap endpoint")
    rng = np.random.default_rng(seed)
    bootstrap = differences[rng.integers(0, len(left), (repeats, len(left)))].mean(axis=1)
    return {"endpoint": f"station macro {endpoint} daily F1 difference", "stations": len(left),
            "difference": float(differences.mean()),
            "percentile_ci_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "repeats": repeats, "seed": seed}
