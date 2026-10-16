"""Score frozen grouped proposal calibration, including numerical-vote controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_review_utility_v2 import verify_receipts
from matplotlib.colors import ListedColormap
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_grouped_proposal import verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_proposal_utility import fit_proposal_model
from gnss_sim.landslide_review_utility import correction_utility


def analyze(run, out):
    verify(run)
    state = read_json(run / "state.json")
    if state["status"] != "predictions_complete":
        raise ValueError("Only terminal grouped predictions may be scored")
    for filename in ("prediction-freeze.json", "model-freeze.json", "training-selection-freeze.json", "validation-selection-freeze.json"):
        for name, expected in read_json(run / filename).items():
            if sha(run / name) != expected:
                raise ValueError("Grouped models, choices or predictions changed")
    config = read_json(run / "protocol.json")
    data = Path(config["data"])
    train, validation = config["split"]["train"], config["split"]["validation"]
    records = {cid: load_case_features(run, cid) for cid in train + validation}
    predictions = read_json(run / "predictions.json")
    training = read_json(run / "training-utilities.json")
    models = read_json(run / "models.json")
    matrix = np.array([row["features"] for row in training])
    target = [row["utility_fraction"] for row in training]
    for name, indices in (("grouped_proposal_guard", list(range(5))), ("visual_guard", [0, 1, 2])):
        if fit_proposal_model(matrix, target, indices) != models[name]:
            raise ValueError("Grouped acceptance model does not refit exactly")
    numerical = recompute_numerical(data, {cid: records[cid] for cid in validation})
    for name, values in numerical.items():
        if name != "ungated_rule" and predictions[name] != values:
            raise ValueError("Common numerical controls no longer reproduce")
    new_receipts = verify_receipts(run)
    cached_receipts = verify_receipts(Path(config["pilot"]))
    receipts = {**cached_receipts, **new_receipts}
    used = read_json(run / "used-requests.json")
    for keys in used.values():
        for key in keys:
            if key not in receipts:
                raise ValueError("A deployed flow lacks its actual cached/new receipt")
    exclusions = read_json(run / "prior-exclusions.json")
    for group, ids in exclusions.items():
        if len(ids) != 18 or any(config["training_groups"][cid] == int(group) for cid in ids):
            raise ValueError("A numerical prior includes its target training background")
    out.mkdir(parents=True, exist_ok=False)
    # Model/initial/local predictions have already frozen; references are for scoring only.
    with (data / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reference = [row for row in csv.DictReader(handle) if row["case_id"] in validation]
    write_rows(out / "reference-subset.csv", reference)
    labels = load_labels(out / "reference-subset.csv", {cid: records[cid][0] for cid in validation})
    initial_saved = read_json(run / "initial-predictions.json")
    results, summaries, groups, changes, timelines = {}, [], [], {}, []
    truth = {cid: read_json(data / "mechanism-reference" / cid / "truth.json") for cid in validation}
    for method, values in predictions.items():
        per_case, per_change, boundaries = {}, {}, []
        weak_total, weak_detected = 0, 0
        events05 = np.zeros(3, dtype=int)
        for cid, value in values.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                                            status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (prediction.activity == 1)).sum())
            boundaries.extend(boundary_errors(labels[cid], prediction.activity, .3)["matched_pairs"])
            detail = boundary_errors(labels[cid], prediction.activity, .5)
            events05 += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
            if method.endswith(("_r0", "_r1")):
                initial = np.array(initial_saved[f"{cid}-direct-r{method[-1]}"]["activity"])
                after = prediction.activity
                per_change[cid] = {
                    **correction_utility(actual, initial, after, actual >= 0),
                    "gained_tp": int(((actual == 1) & (initial != 1) & (after == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (initial == 1) & (after != 1)).sum()),
                    "added_fp": int(((actual == 0) & (initial != 1) & (after == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (initial == 1) & (after != 1)).sum()),
                }
            for day, row in enumerate(labels[cid]):
                timelines.append({"method": method, "case_id": cid, "day_index": day, "date": row["date"],
                                  "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                                  "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        changes[method] = per_change
        flow = [receipts[key] for key in used.get(method, [])]
        summaries.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_total,
                          "weak_days": weak_total, "coverage": aggregate["coverage"],
                          "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": score_counts(*events05)["f1"],
                          "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in boundaries])) if boundaries else None,
                          "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in boundaries])) if boundaries else None,
                          "stage_macro_f1": aggregate["stage_macro_f1"], "requests": len(flow),
                          "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in flow)})
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                ids = [cid for cid in validation if truth[cid][field] == group]
                measured = aggregate_scores([per_case[cid] for cid in ids])
                groups.append({"method": method, "field": field, "group": group, "records": len(ids), **measured["daily"]})
    stages = verify_metrics(predictions, results, labels, {cid: records[cid] for cid in validation})
    for row in summaries:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    index = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        main, direct, vote, visual = [f"{stem}_r{rep}" for stem in
                                     ("grouped_proposal_guard", "direct_visual", "numeric_vote", "visual_guard")]
        a, d = index[main], index[direct]
        criteria = {"daily_gain": a["f1"] > d["f1"], "weak_nonloss": a["weak_recall"] >= d["weak_recall"],
                    "numeric_vote_nonloss": a["f1"] >= index[vote]["f1"],
                    "visual_nonloss": a["f1"] >= index[visual]["f1"],
                    "multiple_actual_gains": sum(row["net_correct"] > 0 for row in changes[main].values()) >= 2}
        decisions.append({"repeat": rep, "criteria": criteria, "passed": all(criteria.values()),
                          "strict_gain_over_vote": a["f1"] > index[vote]["f1"]})
    allows = all(row["passed"] for row in decisions) and any(row["strict_gain_over_vote"] for row in decisions)
    decision = {"allows_full_prototype": allows, "repeats": decisions,
                "scope": "Development continuation only; no independent confirmation authorized by this gate"}
    cost = {"new_started": len(list((run / "requests").glob("*.started.json"))),
            "new_local_receipts": len(new_receipts), "new_returned_json": sum(row["status"] == "returned_json" for row in new_receipts.values()),
            "new_reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in new_receipts.values()),
            "training_requests": sum(row["request_id"].startswith(tuple(train)) for row in new_receipts.values()),
            "validation_requests": sum(row["request_id"].startswith(tuple(validation)) for row in new_receipts.values()),
            "usage_unknown_receipts": sum("usage" not in row for row in new_receipts.values()),
            "schema_or_transport_failures": len(read_json(run / "failures.json")), "new_records": 0,
            "cached_pilot_receipts_verified": len(cached_receipts), "deployment_costs": "Charge all selected proposals even when rejected; do not sum arms"}
    for name, value in (("results.json", results), ("decision.json", decision), ("changes.json", changes),
                        ("internal-stages.json", stages), ("actual-costs.json", cost)):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("groups.csv", groups), ("daily-predictions.csv", timelines)):
        write_rows(out / name, rows)
    for cid in validation:
        case = records[cid][0]
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "grouped_proposal_guard", "visual_guard", "numeric_vote", "raw_conflict", "fixed_grouped_guard")]
        methods.append("xgboost_84")
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in methods)
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for axis in range(3):
            axes[axis].plot(raw[:, axis], linewidth=.8, color="#24887b")
            axes[axis].set_ylabel(f"{'NEU'[axis]} (mm)")
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                       cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                       extent=[-.5, len(raw)-.5, len(matrix)-.5, -.5])
        axes[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=7)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | Grouped deployment-proposal calibration | Development only")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=120)
        plt.close(fig)
    save_json(out / "state.json", {"status": "complete", "configs": len(results), "training_proposals": len(training),
                                  "numeric_validation_predictions_recomputed": len(numerical) * len(validation),
                                  "daily_rows": len(timelines), "group_rows": len(groups), "full_record_figures": len(validation),
                                  "cached_payloads_verified": len(cached_receipts), "new_payloads_verified": len(new_receipts)})
    print({"decision": decision, "cost": cost, "main": [index[f"grouped_proposal_guard_r{rep}"] for rep in (0, 1)]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)
