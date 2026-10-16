"""Score paired context observations without changing their registered decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_review_utility_v2 import verify_receipts
from analyze_landslide_utility_prototype import strong_additional_controls
from matplotlib.colors import ListedColormap
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_context_observation import verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity


def analyze(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Only terminal frozen predictions can be scored")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Context predictions changed")
    config = read_json(run / "protocol.json")
    data, original, complete = [Path(config[name]) for name in ("data", "original", "complete")]
    ids = config["case_ids"]
    records = {cid: load_case_features(run, cid) for cid in ids}
    receipts = {**verify_receipts(run / "cached"), **verify_receipts(run)}
    initial = read_json(run / "initial-predictions.json")
    predictions = read_json(run / "predictions.json")
    locals_saved = read_json(run / "local-predictions.json")
    cached_local = read_json(run / "cached-locals.json")
    selections = read_json(run / "selections.json")
    permissions = read_json(run / "permissions.json")
    used = read_json(run / "used-requests.json")
    replay_checks = 0
    for cid, (case, feature) in records.items():
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            baseline = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"])
            priorities, selected = review_priorities(baseline, feature, long,
                                                     regular_windows(len(baseline), 180, 90), ReviewPolicy())
            choice = selections[f"{cid}-r{rep}"]
            if priorities != choice["priorities"] or selected != choice["raw"]:
                raise ValueError("Original raw selection differs")
            neighbourhood = conflict_neighbourhood(baseline, feature, long, ReviewPolicy())[1]
            if np.flatnonzero(neighbourhood).tolist() != choice["neighbourhood"]:
                raise ValueError("Original raw revision neighbourhood differs")
            for stem in ("adaptive_local", "fixed_local", "adaptive_context", "fixed_context", "adaptive_duplicate"):
                method = f"{stem}_r{rep}"
                targets = choice["fixed" if stem.startswith("fixed") else "raw"]
                authority = target_scopes(neighbourhood, feature.observed, targets)[0]
                if np.flatnonzero(authority).tolist() != permissions[method][cid]:
                    raise ValueError("Revision authority differs")
                after = np.array(predictions[method][cid]["activity"])
                if np.any(after[~authority] != baseline[~authority]):
                    raise ValueError("Review modified dates outside authority")
                replayed = baseline.copy()
                for target in targets:
                    variant = "image" if stem.endswith("local") else (
                        "duplicate180" if stem.endswith("duplicate") else "context360")
                    key = f"{cid}-window-{target['index']:02d}-{variant}-r{rep}"
                    receipt = receipts[key]
                    if variant == "image":
                        status = cached_local[key]["status"]
                    else:
                        status = locals_saved[key]["status"]
                    reviewed = np.full(len(baseline), -1)
                    if status == "success":
                        reviewed = parse_window_activity(json.loads(receipt["output"]), feature.observed,
                                                         target["start"], target["stop"])
                    permission = target_scopes(neighbourhood, feature.observed, [target])[0]
                    replayed[permission] = reviewed[permission]
                np.testing.assert_array_equal(after, replayed)
                replay_checks += 1
    numeric = recompute_numerical(data, records)
    saved_numeric = read_json(original / "numeric-predictions.json")
    for method, values in numeric.items():
        if values != {cid: saved_numeric[method][cid] for cid in ids}:
            raise ValueError("Existing numeric baseline changed")
        if method != "ungated_rule":
            predictions[method] = values
    head = XGBClassifier()
    head.load_model(data / "stage-xgboost.json")
    all_records = {cid: load_case_features(complete, cid) for cid in read_json(complete / "protocol.json")["case_ids"]}
    additional = strong_additional_controls(data, all_records, head)
    for method, values in additional.items():
        predictions[method] = {cid: values[cid] for cid in ids}
    out.mkdir(parents=True, exist_ok=False)
    with (data / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reference = [row for row in csv.DictReader(handle) if row["case_id"] in ids]
    write_rows(out / "reference-subset.csv", reference)
    labels = load_labels(out / "reference-subset.csv", {cid: records[cid][0] for cid in ids})
    results, summaries, changes, daily, event_rows, groups = {}, [], {}, [], [], []
    truth = {cid: read_json(data / "mechanism-reference" / cid / "truth.json") for cid in ids}
    for method, values in predictions.items():
        per_case, boundaries = {}, []
        weak_total, weak_detected = 0, 0
        event05 = np.zeros(3, dtype=int)
        for cid, value in values.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (prediction.activity == 1)).sum())
            for threshold in (.3, .5):
                detail = boundary_errors(labels[cid], prediction.activity, threshold)
                if threshold == .3:
                    boundaries.extend(detail["matched_pairs"])
                else:
                    event05 += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
                event_rows.append({"method": method, "case_id": cid, "iou": threshold,
                    "matched_events": len(detail["matched_pairs"]),
                    "unmatched_prediction_events": detail["unmatched_prediction_events"], "missed_events": detail["missed_events"]})
            if method.endswith(("_r0", "_r1")):
                for comparator in ("direct_visual", "adaptive_local", "adaptive_duplicate"):
                    before = np.array(predictions[f"{comparator}_r{method[-1]}"][cid]["activity"])
                    after = prediction.activity
                    changes.setdefault(method, {}).setdefault(comparator, {})[cid] = {
                        **correction_utility(actual, before, after, actual >= 0),
                        "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                        "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                        "added_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                        "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum()),
                    }
            for day, row in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day, "date": row["date"],
                    "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                    "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        flow = [receipts[key] for key in used.get(method, [])]
        summaries.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_total,
            "weak_days": weak_total, "coverage": aggregate["coverage"], "event_f1_iou03": aggregate["events"]["f1"],
            "event_f1_iou05": score_counts(*event05)["f1"],
            "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in boundaries])) if boundaries else None,
            "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in boundaries])) if boundaries else None,
            "stage_macro_f1": aggregate["stage_macro_f1"], "requests": len(flow),
            "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in flow),
            "usage_unknown_receipts": sum("usage" not in row for row in flow)})
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                group_ids = [cid for cid in ids if truth[cid][field] == group]
                measured = aggregate_scores([per_case[cid] for cid in group_ids])
                groups.append({"method": method, "field": field, "group": group, "records": len(group_ids), **measured["daily"]})
    stages = verify_metrics(predictions, results, labels, records)
    for row in summaries:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        main = f"adaptive_context_r{rep}"
        main_row, local_row, duplicate = [indexed[f"{stem}_r{rep}"] for stem in
                                        ("adaptive_context", "adaptive_local", "adaptive_duplicate")]
        local_change = changes[main]["adaptive_local"]
        criteria = {"f1_gain_over_local": main_row["f1"] > local_row["f1"],
            "positive_net_definite_correction": sum(row["net_correct"] for row in local_change.values()) > 0,
            "multiple_records_helped": sum(row["net_correct"] > 0 for row in local_change.values()) >= 2,
            "f1_nonloss_over_duplicate": main_row["f1"] >= duplicate["f1"]}
        decisions.append({"repeat": rep, "criteria": criteria, "passed": all(criteria.values()),
                          "strict_gain_over_duplicate": main_row["f1"] > duplicate["f1"]})
    allows = all(row["passed"] for row in decisions) and any(row["strict_gain_over_duplicate"] for row in decisions)
    new = verify_receipts(run)
    costs = {"new_started": len(list((run / "requests").glob("*.started.json"))), "new_local_receipts": len(new),
        "new_returned_json": sum(row["status"] == "returned_json" for row in new.values()),
        "new_reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in new.values()),
        "usage_unknown_receipts": sum("usage" not in row for row in new.values()),
        "schema_or_transport_failures": len(read_json(run / "failures.json")),
        "cached_receipts_verified": len(receipts) - len(new), "new_records": 0,
        "accounting": "Collection costs count union of distinct real calls once; deployment arms count every "
        "selected view including failures. Context/duplicate views have more image tokens, no call saving claim"}
    for cid, (case, _) in records.items():
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "adaptive_local", "adaptive_context", "adaptive_duplicate", "fixed_context")]
        methods += ["xgboost_84", "xgboost_direction_information"]
        actual = [int(row["activity_label"]) for row in labels[cid]]
        matrix = np.array([actual] + [predictions[method][cid]["activity"] for method in methods])
        fig, panels = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for axis, name in enumerate(("N", "E", "U")):
            panels[axis].plot(np.arange(len(values)), values[:, axis], linewidth=.7)
            panels[axis].set_ylabel(name + " (mm)")
            panels[axis].grid(alpha=.2)
        panels[3].imshow(matrix, aspect="auto", interpolation="nearest", extent=[-.5, len(values)-.5, len(matrix)-.5, -.5],
                         cmap=ListedColormap(["#e4e6e9", "#f7f7f7", "#238b72"]), vmin=-1, vmax=1)
        panels[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=7)
        panels[3].set_xlabel("Original day index | grey=unknown; green=active")
        fig.suptitle(cid + " | Exposed development; shared residual background, not field truth")
        fig.tight_layout()
        fig.savefig(out / f"{cid}-comparison.png", dpi=150)
        plt.close(fig)
    for name, value in (("predictions-with-controls.json", predictions), ("results.json", results), ("changes.json", changes),
                        ("internal-stages.json", stages), ("actual-costs.json", costs),
                        ("decision.json", {"allows_full_prototype": allows, "repeats": decisions,
                                           "scope": "Development continuation only, not independent confirmation"})):
        save_json(out / name, value)
    write_rows(out / "summary.csv", summaries)
    write_rows(out / "groups.csv", groups)
    write_rows(out / "daily.csv", daily)
    write_rows(out / "events.csv", event_rows)
    save_json(out / "checks.json", {"prediction_and_scope_replays": replay_checks,
        "actual_payloads_verified": len(receipts), "methods_independently_scored": len(stages),
        "common_numeric_record_predictions_recomputed": len(numeric) * len(ids),
        "additional_numeric_record_predictions_recomputed": len(additional) * len(all_records),
        "daily_rows": len(daily), "group_rows": len(groups), "full_record_figures": len(ids),
        "registered_gate_used": True, "new_inference_in_analysis": 0})
    print({"allows_full_prototype": allows, "actual_costs": costs, "methods": len(predictions)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.out.resolve())
