"""Boundary, scenario and paired-record analyses of fully frozen Agent results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_comparison import STAGES, save_json, sha
from run_landslide_window_optimization import read_json
from scipy.optimize import linear_sum_assignment

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    event_masks,
    score_counts,
)


def boundary_errors(rows, activity, threshold):
    """Match by maximum cardinality then IoU; report errors only for matched events."""
    actual = np.array([int(row["activity_label"]) for row in rows])
    known = actual >= 0
    episodes = sorted({row["episode_id"] for row in rows if int(row["activity_label"]) == 1})
    targets = [np.array([row["episode_id"] == episode for row in rows]) & (actual == 1)
               for episode in episodes]
    predicted = [mask for mask in event_masks(activity) if np.any(mask & known)]
    pairs = []
    if targets and predicted:
        overlaps = np.zeros((len(targets), len(predicted)))
        for left, target in enumerate(targets):
            for right, candidate in enumerate(predicted):
                union = np.count_nonzero((target | candidate) & known)
                overlaps[left, right] = np.count_nonzero(target & candidate & known) / union
        # One additional legal match outweighs the sum of all possible IoU differences.
        eligible = overlaps >= threshold
        rewards = eligible * (min(overlaps.shape) + 1 + overlaps)
        target_indices, prediction_indices = linear_sum_assignment(-rewards)
        for left, right in zip(target_indices, prediction_indices):
            if not eligible[left, right]:
                continue
            target_days = np.flatnonzero(targets[left])
            predicted_days = np.flatnonzero(predicted[right])
            pairs.append({"episode_id": episodes[left], "iou": overlaps[left, right],
                          "reference_start": int(target_days[0]),
                          "reference_end": int(target_days[-1]),
                          "predicted_start": int(predicted_days[0]),
                          "predicted_end": int(predicted_days[-1]),
                          "onset_error_days": int(predicted_days[0] - target_days[0]),
                          "offset_error_days": int(predicted_days[-1] - target_days[-1])})
    return {"matched_pairs": pairs, "missed_events": len(targets) - len(pairs),
            "unmatched_prediction_events": len(predicted) - len(pairs)}


def paired_record_interval(results, control, treatment, seed=20261009):
    cases = sorted(results[control]["per_case"])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(2000):
        indices = rng.integers(0, len(cases), len(cases))
        scores = {}
        for name in (control, treatment):
            scores[name] = aggregate_scores([results[name]["per_case"][cases[index]]
                                            for index in indices])["daily"]["f1"]
        if all(value is not None for value in scores.values()):
            differences.append(scores[treatment] - scores[control])
    return {"record_count": len(cases), "draws_with_defined_f1": len(differences),
            "difference": results[treatment]["aggregate"]["daily"]["f1"]
                          - results[control]["aggregate"]["daily"]["f1"],
            "exploratory_95_percentile_interval": np.quantile(differences, [.025, .975]).tolist(),
            "scope": "Paired whole-record descriptive bootstrap; no confirmatory significance, "
                     "no pooling of repetitions into independent samples"}


def analyze(run, out, additional_baselines=None):
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Analyze only a completed run")
    predictions = read_json(run / "predictions.json")
    baseline = read_json(run / "numeric-predictions.json")
    case_ids = read_json(run / "protocol.json")["case_ids"]
    for name in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage"):
        predictions[name] = {cid: baseline[name][cid] for cid in case_ids}
    results = read_json(run / "results.json")
    rows = {cid: [] for cid in case_ids}
    with (run / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["case_id"]].append(row)
    if additional_baselines is not None:
        for method, cases in read_json(additional_baselines / "predictions.json").items():
            if method in predictions or set(cases) != set(case_ids):
                raise ValueError("Additional baselines must be new methods on the same test records")
            predictions[method] = cases
            per_case = {cid: evaluate_case(rows[cid], ActivityPrediction(
                np.array(value["activity"]), np.array(value["stage"])))
                for cid, value in cases.items()}
            results[method] = {"aggregate": aggregate_scores(list(per_case.values())),
                               "per_case": per_case}
    out.mkdir(parents=True, exist_ok=False)
    metrics, records = {}, []
    mechanism = "mechanism" in read_json(run / "protocol.json")["scope"].lower()
    for method, cases in predictions.items():
        method_boundaries = {}
        weak_total, weak_detected = 0, 0
        confusion = np.zeros((3, 5), dtype=int)
        for cid, serialized in cases.items():
            activity = np.array(serialized["activity"])
            stage = np.array(serialized["stage"])
            reference = rows[cid]
            if mechanism:
                weak = np.array([int(row["weak_signal"]) == 1 for row in reference])
            else:
                weak = np.array([row["feature"] == "slow_displacement" for row in reference])
            weak_total += int(weak.sum())
            weak_detected += int(np.count_nonzero(weak & (activity == 1)))
            for target_code in (1, 2, 3):
                mask = np.array([STAGES.get(row["feature"], -1) == target_code
                                 for row in reference])
                for column, prediction_code in enumerate((-1, 0, 1, 2, 3)):
                    confusion[target_code - 1, column] += np.count_nonzero(
                        mask & (stage == prediction_code))
            method_boundaries[cid] = {}
            for threshold in (.3, .5):
                detail = boundary_errors(reference, activity, threshold)
                method_boundaries[cid][str(threshold)] = detail
            record = results[method]["per_case"][cid]
            records.append({"method": method, "case_id": cid,
                            "family": ("quiet", "instrument_step", "weak_constant", "asd",
                                       "reactivation", "weak_after_strong")[(int(cid[-4:]) - 1) % 6]
                                      if mechanism else "previously_viewed_development",
                            "tp": record["daily"]["tp"], "fp": record["daily"]["fp"],
                            "fn": record["daily"]["fn"],
                            "weak_days": int(weak.sum()),
                            "weak_detected_days": int(np.count_nonzero(weak & (activity == 1)))})
        matched = [pair for value in method_boundaries.values()
                   for pair in value["0.3"]["matched_pairs"]]
        onset = [abs(pair["onset_error_days"]) for pair in matched]
        offset = [abs(pair["offset_error_days"]) for pair in matched]
        metrics[method] = {"weak_days": weak_total, "weak_detected_days": weak_detected,
                           "weak_recall": weak_detected / weak_total if weak_total else None,
                           "stage_confusion_rows_ASD_columns_unknown_none_ASD": confusion.tolist(),
                           "boundaries": method_boundaries,
                           "onset_mae_matched_days": float(np.mean(onset)) if onset else None,
                           "offset_mae_matched_days": float(np.mean(offset)) if offset else None,
                           "matched_event_count": len(matched),
                           "boundary_reference": "Exact controlled latent endpoints" if mechanism
                           else "Conservative confirmed internal reference endpoints; "
                                "not exact physical boundaries"}
        matched05 = sum(len(value["0.5"]["matched_pairs"])
                        for value in method_boundaries.values())
        missed05 = sum(value["0.5"]["missed_events"] for value in method_boundaries.values())
        unmatched05 = sum(value["0.5"]["unmatched_prediction_events"]
                          for value in method_boundaries.values())
        metrics[method]["events_iou05"] = score_counts(matched05, unmatched05, missed05)
        if len(matched) != results[method]["aggregate"]["events"]["tp"]:
            raise ValueError("Boundary matching disagrees with common event scoring")
    intervals = {}
    for rep in range(read_json(run / "protocol.json")["repetitions"]):
        treatment = f"adaptive_numeric_feedback_r{rep}"
        for control in (f"direct_visual_r{rep}", f"fixed_image_review_r{rep}",
                        f"adaptive_no_feedback_r{rep}", "robust_xgboost_stage", "xgboost"):
            intervals[f"{treatment}-minus-{control}"] = paired_record_interval(
                results, control, treatment)
        if additional_baselines is not None:
            for control in ("raw_multiscale_rule", "xgboost_raw_persistence"):
                intervals[f"{treatment}-minus-{control}"] = paired_record_interval(
                    results, control, treatment)
    save_json(out / "combined-results.json", results)
    save_json(out / "analysis.json", {"metrics": metrics, "paired_intervals": intervals,
                                      "source_hashes": {name: sha(run / name) for name in (
                                          "predictions.json", "results.json", "reference-subset.csv",
                                          "numeric-predictions.json", "protocol.json")}})
    with (out / "per-record.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    costs = read_json(run / "costs.json")
    for method, result in results.items():
        row = result["aggregate"]
        summary.append({"method": method, **row["daily"],
                        "event_f1_iou03": row["events"]["f1"],
                        "event_f1_iou05": metrics[method]["events_iou05"]["f1"],
                        "stage_macro_f1": row["stage_macro_f1"],
                        "weak_recall": metrics[method]["weak_recall"],
                        "coverage": row["coverage"],
                        "false_acceleration_days": row["false_acceleration_days"],
                        "onset_mae_matched_days": metrics[method]["onset_mae_matched_days"],
                        "offset_mae_matched_days": metrics[method]["offset_mae_matched_days"],
                        "requests": costs.get(method, {}).get("requests", 0),
                        "tokens": costs.get(method, {}).get("tokens", 0)})
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fig, panels = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    labels = [row["method"].replace("adaptive_numeric_feedback", "adaptive + numeric")
              .replace("adaptive_no_feedback", "adaptive image")
              .replace("fixed_image_review", "fixed image") for row in summary]
    positions = np.arange(len(summary))
    for panel, key, title in zip(panels, ("f1", "weak_recall", "fp"),
                                 ("Activity daily F1", "Weak-motion recall", "False-positive days")):
        panel.barh(positions, [row[key] or 0 for row in summary], color="#237c78", alpha=.8)
        panel.set_title(title)
        panel.set_yticks(positions, labels)
        panel.grid(axis="x", alpha=.2)
        if key != "fp":
            panel.set_xlim(0, 1.05)
    panels[0].invert_yaxis()
    fig.suptitle("Controlled mechanism recovery" if mechanism else "Non-independent development pilot")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)
    print("Analysis completed", len(summary), "methods; weak support", metrics[summary[0]["method"]]["weak_days"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-baselines", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.out, args.additional_baselines)


if __name__ == "__main__":
    main()
