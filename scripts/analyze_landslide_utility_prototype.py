"""Verify complete utility-selected reviews and compare all strong existing controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_review_utility_v2 import verify_receipts
from finalize_landslide_adaptive_evidence import verify_request_receipts
from fit_landslide_causal_direction_controls import direction_features, direction_rule
from fit_landslide_scale_controls import scale_information
from matplotlib.colors import ListedColormap
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_utility_prototype import verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
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
    predict_utility,
    select_utility_windows,
    window_review_features,
)


def strong_additional_controls(data, records, head):
    base = data.parent
    direction = base / "causal-direction-information-controls-v1"
    scale = base / "scale-information-controls-v1"
    for folder in (direction, scale):
        for name, expected in read_json(folder / "fitting-freeze.json").items():
            if sha(folder / name) != expected:
                raise ValueError("An additional strong control was modified")
        if sha(folder / "predictions.json") != read_json(folder / "manifest.json")["prediction_sha256"]:
            raise ValueError("Additional control prediction freeze changed")
    selected_direction = read_json(direction / "frozen-selection.json")
    selected_scale = read_json(scale / "frozen-selection.json")
    direction_head, scale_head = XGBClassifier(), XGBClassifier()
    direction_head.load_model(direction / "direction-xgboost.json")
    scale_head.load_model(scale / "scale-xgboost.json")
    calibration = read_json(data / "long-selection.json")["raw_multiscale_rule"]
    saved = {**read_json(direction / "predictions.json"), **read_json(scale / "predictions.json")}
    reproduced = {name: {} for name in saved}
    for cid, (case, feature) in records.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        matrix, support, directional, _, _, long = direction_features(case, feature, calibration)
        gate = direction_rule(feature, long, directional, calibration,
                              selected_direction["rule"]["rate"], selected_direction["rule"]["snr"])
        reproduced["direction_rule"][cid] = learned_stages(codes, gate, feature.supported).to_dict()
        gate = activity_from_mask(direction_head.predict_proba(matrix)[:, 1] >= selected_direction["xgboost"]["threshold"], support)
        reproduced["xgboost_direction_information"][cid] = learned_stages(codes, gate, feature.supported).to_dict()
        matrix, support = scale_information(case, feature)
        gate = activity_from_mask(scale_head.predict_proba(matrix)[:, 1] >= selected_scale["threshold"], support)
        reproduced["xgboost_scale_information"][cid] = learned_stages(codes, gate, feature.supported).to_dict()
    if reproduced != saved:
        raise ValueError("Additional strong controls do not reproduce")
    return reproduced


def analyze(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only terminal, frozen prototype inference")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Complete prototype predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "protocol.json")
    source = Path(config["source"])
    data = ROOT / read_json(source / "protocol.json")["source_bundle"]
    original = read_json(source / "predictions.json")
    models = read_json(run / "utility-models.json")
    if models != read_json(Path(config["pilot"]) / "utility-models.json"):
        raise ValueError("The six-record calibrator was changed for full development")
    selections = read_json(run / "selections.json")
    locals_saved = {**read_json(run / "cached-locals.json"), **read_json(run / "new-locals.json")}
    original_receipts = verify_request_receipts(source)
    new_receipts = verify_receipts(run)
    receipts = {row["request_id"]: row for row in original_receipts}
    receipts.update(new_receipts)
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    numerical = recompute_numerical(data, records)
    if numerical != {key: value for key, value in read_json(run / "numeric-predictions.json").items() if key != "ungated_rule"}:
        raise ValueError("Existing common numerical predictions changed")
    head, prior = XGBClassifier(), XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    prior.load_model(data / "xgboost_84.json")
    predictions, costs, permissions = {}, {}, {}
    checks = 0
    for cid, (case, feature) in records.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        windows = regular_windows(len(case.dates), 180, 90)
        evidence = {scale: raw_motion_evidence(case, feature, scale) for scale in (42, 84)}
        matrix = np.column_stack((feature.matrix,
            *[evidence[scale][key] for scale in (42, 84) for key in ("rate", "signal_to_noise", "direction")]))
        probability = prior.predict_proba(matrix)[:, 1]
        for rep in (0, 1):
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            direct = f"direct_visual_r{rep}"
            predictions.setdefault(direct, {})[cid] = original[direct][cid]
            direct_receipt = receipts[f"{cid}-direct-r{rep}"]
            costs.setdefault(direct, []).append(direct_receipt)
            neighbourhood = conflict_neighbourhood(initial, feature, evidence[42], ReviewPolicy())[1]
            matrix = window_review_features(initial, feature, evidence[42], probability, windows, ReviewPolicy())
            if matrix.tolist() != read_json(run / "selection-features.json")[f"{cid}-r{rep}"]:
                raise ValueError("Prototype pre-observation features differ")
            editable = [int(neighbourhood[start:stop].sum()) for start, stop in windows]
            arms = selections[f"{cid}-r{rep}"]
            for stem, selector in models.items():
                priorities, selected = select_utility_windows(windows, predict_utility(selector, matrix), editable)
                if priorities != arms[stem]["priorities"] or selected != arms[stem]["selected"]:
                    raise ValueError("Frozen utility choices differ")
            for stem, arm in arms.items():
                selected = arm["selected"]
                reviews = [locals_saved[f"{cid}-window-{row['index']:02d}-image-r{rep}"] for row in selected]
                combined = combine_reviews(initial, reviews, feature.observed)
                scope = target_scopes(neighbourhood, feature.observed, selected)[0]
                if np.flatnonzero(scope).tolist() != arm["editable_indices"]:
                    raise ValueError("Complete prototype revision permission changed")
                activity = scoped_activity(initial, combined.activity, scope, feature.observed)
                if np.any(activity[~scope] != initial[~scope]) or np.any(activity[~feature.observed] != -1):
                    raise ValueError("Prototype revised unauthorized or missing dates")
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                permissions[f"{method}-{cid}"] = np.flatnonzero(scope).tolist()
                costs.setdefault(method, []).append(direct_receipt)
                costs[method].extend(receipts[row["request_id"]] for row in reviews)
                checks += 1
            # Preserve the prior numeric-text workflow as an additional existing comparator.
            raw_arm = arms["raw_conflict"]
            scope = np.zeros(len(initial), dtype=bool)
            scope[raw_arm["editable_indices"]] = True
            reviewed = np.asarray(original[f"adaptive_numeric_feedback_r{rep}"][cid]["activity"])
            activity = scoped_activity(initial, reviewed, scope, feature.observed)
            method = f"old_numeric_text_r{rep}"
            predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
            costs.setdefault(method, []).append(direct_receipt)
            costs[method].extend(receipts[f"{cid}-window-{row['index']:02d}-numeric-r{rep}"] for row in raw_arm["selected"])
    predictions.update(numerical)
    predictions.update(strong_additional_controls(data, records, head))
    save_json(out / "predictions.json", predictions)
    save_json(out / "permissions.json", permissions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("predictions.json", "permissions.json")})
    labels = load_labels(source / "analysis-v1/reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    results, changes, summaries, groups, timelines = {}, {}, [], [], []
    truth = {cid: read_json(data / "mechanism-reference" / cid / "truth.json") for cid in records}
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
                initial = np.asarray(original[f"direct_visual_r{method[-1]}"][cid]["activity"])
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
        summaries.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected/weak_days,
                          "weak_days": weak_days, "coverage": aggregate["coverage"],
                          "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": score_counts(*event05)["f1"],
                          "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in matched])) if matched else None,
                          "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in matched])) if matched else None,
                          "stage_macro_f1": aggregate["stage_macro_f1"], "requests": len(usage),
                          "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in usage)})
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                ids = [cid for cid in labels if truth[cid][field] == group]
                score = aggregate_scores([per_case[cid] for cid in ids])
                groups.append({"method": method, "field": field, "group": group, "records": len(ids), **score["daily"]})
    stages = verify_metrics(predictions, results, labels, records)
    for row in summaries:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        main, direct, matched, prior_name = [f"{stem}_r{rep}" for stem in
                                           ("calibrated_utility", "direct_visual", "same_count_fixed", "uncalibrated_prior")]
        a, d = indexed[main], indexed[direct]
        gained_records = sum(row["net_correct"] > 0 for row in changes[main]["correct_decisions"].values())
        criteria = {"daily_gain": a["f1"] > d["f1"], "weak_nonloss": a["weak_recall"] >= d["weak_recall"],
                    "event_nonloss": a["event_f1_iou03"] >= d["event_f1_iou03"],
                    "same_count_nonloss": a["f1"] >= indexed[matched]["f1"],
                    "calibration_nonloss": a["f1"] >= indexed[prior_name]["f1"], "multiple_actual_gains": gained_records >= 2}
        decisions.append({"repeat": rep, "passed": all(criteria.values()), "criteria": criteria,
                          "corrected_records": gained_records, "summary": a, **changes[main]["total"]})
    decision = {"allows_independent_confirmation": all(row["passed"] for row in decisions), "repeats": decisions}
    actual_usage = {"new_started": len(new_receipts), "new_local_receipts": len(new_receipts),
                    "new_returned_model_json": sum(row["status"] == "returned_json" for row in new_receipts.values()),
                    "new_reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in new_receipts.values()),
                    "usage_unknown_receipts": sum("usage" not in row for row in new_receipts.values()),
                    "schema_or_transport_failures": len(read_json(run / "failures.json")), "new_records": 0,
                    "cached_local_views": len(read_json(run / "cached-locals.json")), "original_payloads_verified": len(original_receipts),
                    "pilot_collection": read_json(Path(config["pilot"]) / "analysis-v2/actual-costs.json")}
    for name, value in (("results.json", results), ("changes.json", changes), ("internal-stages.json", stages),
                        ("decision.json", decision), ("actual-costs.json", actual_usage)):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("groups.csv", groups), ("daily-predictions.csv", timelines)):
        write_rows(out / name, rows)
    for cid, (case, _) in records.items():
        raw = np.asarray([row if row is not None else [np.nan]*3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "calibrated_utility", "visual_only_utility", "raw_conflict", "same_count_fixed")]
        methods += ["raw_multiscale_rule", "xgboost_84", "xgboost_direction_information"]
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
        axes[3].set_xlim(0, len(raw)-1)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | Frozen six-record correction utility, complete development\n"
                     "Four shared exposed backgrounds; not independent field truth")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=120)
        plt.close(fig)
    save_json(out / "state.json", {"status": "complete", "configs": len(results), "authority_checks": checks,
                                  "numerical_predictions_recomputed": 12*len(records), "daily_rows": len(timelines),
                                  "group_rows": len(groups), "full_record_figures": len(records), "new_payloads_verified": len(new_receipts)})
    print({"decision": decision, "actual_costs": actual_usage}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)
