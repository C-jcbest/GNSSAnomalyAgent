"""Replay calibration and choices, then score frozen validation correction utility."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from matplotlib.colors import ListedColormap
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_review_utility import training_labels, verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import (
    correction_utility,
    fit_utility_model,
    predict_utility,
    select_utility_windows,
    window_review_features,
)


def verify_receipts(run):
    """Check actual payloads while retaining transport and schema failures in cost."""
    receipts = {}
    for path in sorted((run / "requests").glob("*.response.json")):
        receipt = read_json(path)
        request = read_json(path.with_name(path.name.replace(".response.json", ".started.json")))
        content = [{"type": "text", "text": request["prompt"]}]
        for name, expected in request["image_sha256"].items():
            image = run / "images" / name
            if sha(image) != expected:
                raise ValueError("Actual utility request image changed")
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {"model": request["model"], "messages": [{"role": "user", "content": content}],
                   "temperature": .1, "top_p": .3, "max_tokens": 2048, "enable_thinking": False,
                   "response_format": {"type": "json_object"}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if digest != request["request_sha256"] or digest != receipt["request_sha256"]:
            raise ValueError("Actual utility payload binding differs")
        receipts[receipt["request_id"]] = receipt
    if len(receipts) != len(list((run / "requests").glob("*.started.json"))):
        raise ValueError("Started request lacks a terminal local receipt")
    return receipts


def analyze(run, out):
    verify(run)
    config = read_json(run / "protocol.json")
    state = read_json(run / "state.json")
    if state["status"] != "predictions_complete":
        raise ValueError("Full validation comparison needs a completed, frozen inference run")
    for filename in ("prediction-freeze.json", "calibration-freeze.json", "selection-freeze.json"):
        for name, expected in read_json(run / filename).items():
            if sha(run / name) != expected:
                raise ValueError("Utility calibration or predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    receipts = verify_receipts(run)
    initial_saved = read_json(run / "initial-predictions.json")
    local_saved = read_json(run / "local-predictions.json")
    selection = read_json(run / "selections.json")
    models = read_json(run / "utility-models.json")
    source = Path(config["source"])
    train = config["split"]["train"]
    validation = config["split"]["validation"]
    records = {cid: load_case_features(run, cid) for cid in train + validation}
    probabilities = read_json(run / "numeric-probabilities.json")
    actual_train = training_labels(source, train)
    matrices, targets, train_utilities = [], [], []
    train_snapshot = read_json(run / "training-prediction-snapshot.json")
    for rep in (0, 1):
        for cid in train:
            case, feature = records[cid]
            windows = regular_windows(len(case.dates), 180, 90)
            initial = np.asarray(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
            long = raw_motion_evidence(case, feature)
            matrix = window_review_features(initial, feature, long, probabilities[cid], windows, ReviewPolicy())
            neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())[1]
            for index, (start, stop) in enumerate(windows):
                key = f"{cid}-window-{index:02d}-image-r{rep}"
                local = local_saved[key]
                if local != train_snapshot["local"][key] or initial_saved[f"{cid}-direct-r{rep}"] != train_snapshot["initial"][f"{cid}-direct-r{rep}"]:
                    raise ValueError("Training answer changed after model fitting")
                if local["status"] == "failed":
                    continue
                permission = np.zeros(len(initial), dtype=bool)
                permission[start:stop] = neighbourhood[start:stop]
                utility = correction_utility(actual_train[cid], initial, np.asarray(local["activity"]), permission)
                matrices.append(matrix[index])
                targets.append(utility["utility_fraction"])
                train_utilities.append({"case_id": cid, "repeat": rep, "index": index, **utility})
    if train_utilities != read_json(run / "training-utilities.json"):
        raise ValueError("Actual training correction utility differs")
    for name, indices in (("calibrated_utility", list(range(5))), ("visual_only_utility", [0, 1])):
        if fit_utility_model(matrices, targets, indices) != models[name]:
            raise ValueError("Saved correction model does not reproduce from train-only labels")
    # Local validation requests must follow every frozen selection and all model fits.
    train_ids = {f"{cid}-window-{index:02d}-image-r{rep}" for cid in train for rep in (0, 1) for index in range(8)}
    validation_local = [receipt for key, receipt in receipts.items() if "-window-" in key and key not in train_ids]
    validation_global_ids = [f"{cid}-direct-r{rep}" for cid in validation for rep in (0, 1)]
    first_global = min((run / "requests" / f"{key}.started.json").stat().st_mtime_ns for key in validation_global_ids)
    first_local = min((run / "requests" / f"{row['request_id']}.started.json").stat().st_mtime_ns
                      for row in validation_local)
    if (run / "calibration-freeze.json").stat().st_mtime_ns > first_global:
        raise ValueError("Model calibration freeze does not precede validation initial observations")
    if (run / "selection-freeze.json").stat().st_mtime_ns > first_local:
        raise ValueError("Validation selections do not precede validation local observations")
    predictions, costs, permissions = {}, {}, {}
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    source_reference = source / "reference-labels.csv"
    with source_reference.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["case_id"] in validation]
        columns = reader.fieldnames
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    labels = load_labels(out / "reference-subset.csv", {cid: records[cid][0] for cid in validation})
    candidate_rows = []
    authority_checks = 0
    for rep in (0, 1):
        for cid in validation:
            case, feature = records[cid]
            windows = regular_windows(len(case.dates), 180, 90)
            direct_key = f"{cid}-direct-r{rep}"
            initial = np.asarray(initial_saved[direct_key]["activity"])
            codes = head.predict(feature.matrix).astype(int) + 1
            direct = f"direct_visual_r{rep}"
            prediction = learned_stages(codes, initial, feature.supported)
            if initial_saved[direct_key]["status"] == "failed":
                prediction = ActivityPrediction(initial, np.full(len(initial), -1), status="failed")
            predictions.setdefault(direct, {})[cid] = prediction.to_dict()
            costs.setdefault(direct, []).append(receipts[direct_key])
            long = raw_motion_evidence(case, feature)
            matrix = window_review_features(initial, feature, long, probabilities[cid], windows, ReviewPolicy())
            if matrix.tolist() != read_json(run / "selection-features.json")[f"{cid}-r{rep}"]:
                raise ValueError("Validation selection features differ")
            arms = selection[f"{cid}-r{rep}"]
            neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())[1]
            editable = [int(neighbourhood[start:stop].sum()) for start, stop in windows]
            for name, model in models.items():
                priorities, chosen = select_utility_windows(windows, predict_utility(model, matrix), editable)
                if priorities != arms[name]["priorities"] or chosen != arms[name]["selected"]:
                    raise ValueError("Validation choices were not made by the frozen utility model")
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            fractions = predict_utility(models["calibrated_utility"], matrix)
            for index, (start, stop) in enumerate(windows):
                scope = np.zeros(len(initial), dtype=bool)
                scope[start:stop] = neighbourhood[start:stop]
                local = local_saved[f"{cid}-window-{index:02d}-image-r{rep}"]
                utility = correction_utility(actual, initial, np.asarray(local["activity"]), scope)
                candidate_rows.append({"case_id": cid, "repeat": rep, "index": index, "start": start, "stop": stop,
                                       "predicted_utility_fraction": float(fractions[index]), **utility})
            for stem, arm in arms.items():
                targets_selected = arm["selected"]
                local_reviews = [local_saved[f"{cid}-window-{row['index']:02d}-image-r{rep}"] for row in targets_selected]
                combined = combine_reviews(initial, local_reviews, feature.observed)
                scope = target_scopes(neighbourhood, feature.observed, targets_selected)[0]
                if np.flatnonzero(scope).tolist() != arm["editable_indices"]:
                    raise ValueError("Validation authority differs from pre-observation selection")
                activity = scoped_activity(initial, combined.activity, scope, feature.observed)
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                permissions[f"{method}-{cid}"] = np.flatnonzero(scope).tolist()
                if np.any(activity[~scope] != initial[~scope]) or np.any(activity[~feature.observed] != -1):
                    raise ValueError("A selected observation revised unauthorized dates")
                costs.setdefault(method, []).append(receipts[direct_key])
                costs[method].extend(receipts[f"{cid}-window-{row['index']:02d}-image-r{rep}"] for row in local_reviews)
                authority_checks += 1
    numerical = read_json(run / "numeric-predictions.json")
    predictions.update({name: {cid: values[cid] for cid in validation} for name, values in numerical.items()})
    save_json(out / "predictions.json", predictions)
    save_json(out / "permissions.json", permissions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("predictions.json", "permissions.json")})
    results, changes, summary, timelines = {}, {}, [], []
    for method, values in predictions.items():
        per_case, per_changes, corrected = {}, {}, {}
        weak_days, weak_detected, matched = 0, 0, []
        event05 = [0, 0, 0]
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]), np.asarray(value["stage"]),
                                            status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            weak = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_days += int(weak.sum())
            weak_detected += int((weak & (prediction.activity == 1)).sum())
            matched.extend(boundary_errors(labels[cid], prediction.activity, .3)["matched_pairs"])
            detail = boundary_errors(labels[cid], prediction.activity, .5)
            event05[0] += len(detail["matched_pairs"])
            event05[1] += detail["unmatched_prediction_events"]
            event05[2] += detail["missed_events"]
            if method.endswith(("_r0", "_r1")):
                initial = np.asarray(initial_saved[f"{cid}-direct-r{method[-1]}"]["activity"])
                after = prediction.activity
                per_changes[cid] = {
                    "gained_tp": int(((actual == 1) & (initial != 1) & (after == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (initial == 1) & (after != 1)).sum()),
                    "added_fp": int(((actual == 0) & (initial != 1) & (after == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (initial == 1) & (after != 1)).sum()),
                }
                corrected[cid] = correction_utility(actual, initial, after, actual >= 0)
            for day, row in enumerate(labels[cid]):
                timelines.append({"method": method, "case_id": cid, "day_index": day, "date": row["date"],
                                  "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                                  "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        if per_changes:
            changes[method] = {"per_case": per_changes, "correct_decisions": corrected,
                               "total": {key: sum(row[key] for row in per_changes.values()) for key in
                                         ("gained_tp", "lost_tp", "added_fp", "removed_fp")}}
        usage = costs.get(method, [])
        summary.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_days,
                        "coverage": aggregate["coverage"], "event_f1_iou03": aggregate["events"]["f1"],
                        "event_f1_iou05": score_counts(*event05)["f1"],
                        "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in matched])) if matched else None,
                        "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in matched])) if matched else None,
                        "stage_macro_f1": aggregate["stage_macro_f1"], "weak_days": weak_days,
                        "requests": len(usage), "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in usage)})
    stages = verify_metrics(predictions, results, labels, {cid: records[cid] for cid in validation})
    for row in summary:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summary}
    decisions = []
    for rep in (0, 1):
        main, direct, matched, prior = [f"{stem}_r{rep}" for stem in
                                       ("calibrated_utility", "direct_visual", "same_count_fixed", "uncalibrated_prior")]
        a, d = indexed[main], indexed[direct]
        net = changes[main]["correct_decisions"]
        gained_records = sum(row["net_correct"] > 0 for row in net.values())
        criteria = {"daily_gain": a["f1"] > d["f1"], "weak_nonloss": a["weak_recall"] >= d["weak_recall"],
                    "event_nonloss": a["event_f1_iou03"] >= d["event_f1_iou03"],
                    "same_count_nonloss": a["f1"] >= indexed[matched]["f1"],
                    "calibration_nonloss": a["f1"] >= indexed[prior]["f1"],
                    "multiple_actual_gains": gained_records >= 2}
        decisions.append({"repeat": rep, "passed": all(criteria.values()), "criteria": criteria,
                          "corrected_records": gained_records, "summary": a, **changes[main]["total"]})
    decision = {"allows_full_prototype": all(row["passed"] for row in decisions), "repeats": decisions}
    actual_usage = {
        "new_started": len(list((run / "requests").glob("*.started.json"))), "new_local_receipts": len(receipts),
        "new_returned_model_json": sum(row["status"] == "returned_json" for row in receipts.values()),
        "new_reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
        "usage_unknown_receipts": sum("usage" not in row for row in receipts.values()),
        "training_requests": sum(key.startswith(tuple(train)) for key in receipts),
        "validation_requests": sum(key.startswith(tuple(validation)) for key in receipts),
        "validation_local_counterfactual_requests": len(validation_local),
        "schema_or_transport_failures": len(read_json(run / "failures.json")), "new_records": 0,
    }
    for name, value in (("results.json", results), ("changes.json", changes), ("internal-stages.json", stages),
                        ("decision.json", decision), ("actual-costs.json", actual_usage)):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summary), ("candidate-utilities.csv", candidate_rows), ("daily-predictions.csv", timelines)):
        write_rows(out / name, rows)
    for cid in validation:
        case = records[cid][0]
        raw = np.asarray([row if row is not None else [np.nan]*3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "calibrated_utility", "uncalibrated_prior", "raw_conflict", "fixed_review")]
        methods += ["raw_multiscale_rule", "xgboost_84"]
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in methods)
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for axis, name in enumerate(("N", "E", "U")):
            axes[axis].plot(raw[:, axis], linewidth=.8, color="#24887b")
            axes[axis].set_ylabel(f"{name} (mm)")
            axes[axis].grid(alpha=.2)
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                       cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                       extent=[-.5, len(raw)-.5, len(matrix)-.5, -.5])
        axes[3].set_yticks(range(len(matrix)), ["Known injection"]+methods, fontsize=7)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        axes[3].set_xlim(0, len(raw)-1)
        fig.suptitle(f"{cid} | Utility model frozen before validation observations\n"
                     "One shared development background; not independent field truth")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=120)
        plt.close(fig)
    save_json(out / "state.json", {"status": "complete", "configs": len(results), "authority_checks": authority_checks,
                                  "training_model_replayed": 2, "training_window_utilities": len(train_utilities),
                                  "validation_candidate_utilities": len(candidate_rows), "payload_receipts_verified": len(receipts),
                                  "daily_rows": len(timelines), "full_record_figures": len(validation),
                                  "filesystem_order_model_before_initial_and_selection_before_local": True,
                                  "order_scope": "Observed filesystem metadata and frozen execution logic, not an external timestamp attestation"})
    print({"decision": decision, "actual_costs": actual_usage}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)
