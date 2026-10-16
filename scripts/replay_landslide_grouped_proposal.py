"""Apply the frozen four-background acceptor to complete exposed development."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_review_utility_v2 import verify_receipts
from analyze_landslide_utility_prototype import strong_additional_controls
from finalize_landslide_adaptive_evidence import verify_request_receipts
from matplotlib.colors import ListedColormap
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_grouped_proposal import verify as verify_calibration
from run_landslide_proposal_utility_v2 import raw_prior_matrix
from run_landslide_utility_prototype import verify as verify_complete
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_proposal_utility import (
    apply_proposal,
    expected_proposal_utility,
    proposal_features,
)
from gnss_sim.landslide_review_utility import correction_utility


def replay(calibration, complete, out):
    verify_calibration(calibration)
    verify_complete(complete)
    for run in (calibration, complete):
        for name, expected in read_json(run / "prediction-freeze.json").items():
            if sha(run / name) != expected:
                raise ValueError("Original grouped or complete predictions changed")
    if not read_json(calibration / "analysis-v1/decision.json")["allows_full_prototype"]:
        raise ValueError("Grouped calibration did not pass its full-prototype gate")
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(complete / "protocol.json")
    source = Path(config["source"])
    data = ROOT / read_json(source / "protocol.json")["source_bundle"]
    ids = config["case_ids"]
    protocol = {
        "date": "2026-10-10", "calibration": str(calibration.resolve()), "complete_cache": str(complete.resolve()),
        "scope": "Complete24 exposed development on four shared backgrounds; no new observations or independent claims",
        "case_ids": ids, "primary": "grouped_proposal_guard", "model": "Frozen common VLM cached responses",
        "mechanism": "Exact four-background train24 models and features, originalraw two views and49-day permission",
        "gate": "Both repeats mainF1>direct, weakR>=direct, mainF1>=numeric vote, >=2 net-corrected records each; "
                "strictF1 gain over vote in at least1 repeat AND main-vote net definite correct gain>0 in BOTH repeats. "
                "Unknown changes cannot satisfy the additional actual-correction condition; no model or threshold search",
        "tradeoffs": "Event/boundary/stage/cost all reported, not required universally noninferior",
        "additional_gate_rationale": "Validation passes F1 but extra numeric-vote difference may be abstention only. "
                                     "A new confirmation requires actual definite correction beyond the numerical control",
        "new_requests": 0, "new_records": 0,
    }
    save_json(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "execution-bindings.json", {str(path.resolve()): sha(path) for path in
        (Path(__file__), calibration / "models.json", complete / "cached-locals.json", complete / "new-locals.json",
         complete / "analysis-v1/predictions.json", source / "analysis-v1/reference-subset.csv")})
    shutil.copyfile(calibration / "models.json", out / "models.json")
    save_json(out / "model-freeze.json", {"models.json": sha(out / "models.json")})
    models = read_json(out / "models.json")
    records = {cid: load_case_features(complete, cid) for cid in ids}
    old = read_json(complete / "analysis-v1/predictions.json")
    predictions = {name: values for name, values in old.items() if name.startswith(
        ("direct_visual", "raw_conflict", "fixed_review", "old_numeric_text")) or not name.endswith(("_r0", "_r1"))}
    local_saved = {**read_json(complete / "cached-locals.json"), **read_json(complete / "new-locals.json")}
    selection = read_json(complete / "selections.json")
    head, prior = XGBClassifier(), XGBClassifier()
    head.load_model(complete / "stage-xgboost.json")
    prior.load_model(data / "xgboost_84.json")
    numerical = recompute_numerical(data, records)
    numerical.update(strong_additional_controls(data, records, head))
    for name, values in numerical.items():
        if predictions[name] != values:
            raise ValueError("Strong numerical control differs from its frozen original")
    original_receipts = verify_request_receipts(source)
    extra_receipts = verify_receipts(complete)
    receipts = {row["request_id"]: row for row in original_receipts}
    receipts.update(extra_receipts)
    proposals, permissions, costs = [], {}, {}
    for cid, (case, features) in records.items():
        codes = head.predict(features.matrix).astype(int) + 1
        matrix, long = raw_prior_matrix(case, features)
        probability = prior.predict_proba(matrix)[:, 1]
        for rep in (0, 1):
            initial = np.array(old[f"direct_visual_r{rep}"][cid]["activity"])
            neighbourhood = conflict_neighbourhood(initial, features, long, ReviewPolicy())[1]
            for stem in ("grouped_proposal_guard", "visual_guard", "numeric_vote", "fixed_grouped_guard"):
                choices = selection[f"{cid}-r{rep}"]["fixed_review" if stem.startswith("fixed") else "raw_conflict"]["selected"]
                activity = initial.copy()
                authority = np.zeros(len(initial), dtype=bool)
                method = f"{stem}_r{rep}"
                keys = [f"{cid}-direct-r{rep}"]
                for target in choices:
                    key = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                    reviewed = np.array(local_saved[key]["activity"])
                    scope = target_scopes(neighbourhood, features.observed, [target])[0]
                    row, changed = proposal_features(initial, reviewed, scope, features, long, probability, ReviewPolicy())
                    if stem == "numeric_vote":
                        expected = float(row[4])
                    else:
                        selector = models["grouped_proposal_guard" if stem.startswith("fixed") else stem]
                        expected = expected_proposal_utility(selector, row)
                    accepted = expected > 0 and changed.any()
                    if accepted:
                        activity = apply_proposal(activity, reviewed, scope, expected)
                        authority |= scope
                    keys.append(key)
                    proposals.append({"method": method, "case_id": cid, "index": target["index"],
                                      "features": row.tolist(), "expected_utility": expected,
                                      "changed_days": int(changed.sum()), "accepted": bool(accepted)})
                if np.any(activity[~authority] != initial[~authority]) or np.any(activity[~features.observed] != -1):
                    raise ValueError("Complete grouped acceptor changed unauthorized dates")
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, features.supported).to_dict()
                permissions[f"{method}-{cid}"] = np.flatnonzero(authority).tolist()
                costs.setdefault(method, []).extend(receipts[key] for key in keys)
    for name, value in (("predictions.json", predictions), ("proposals.json", proposals), ("permissions.json", permissions)):
        save_json(out / name, value)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("models.json", "predictions.json", "proposals.json", "permissions.json")})
    labels = load_labels(source / "analysis-v1/reference-subset.csv", {cid: records[cid][0] for cid in ids})
    results, summaries, groups, changes, timelines = {}, [], [], {}, []
    with (complete / "analysis-v1/summary.csv").open(encoding="utf-8", newline="") as handle:
        old_summary = {row["method"]: row for row in csv.DictReader(handle)}
    truth = {cid: read_json(data / "mechanism-reference" / cid / "truth.json") for cid in ids}
    for method, values in predictions.items():
        per_case, per_change, boundaries = {}, {}, []
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
            boundaries.extend(boundary_errors(labels[cid], prediction.activity, .3)["matched_pairs"])
            detail = boundary_errors(labels[cid], prediction.activity, .5)
            event05 += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
            if method.endswith(("_r0", "_r1")):
                initial = np.array(old[f"direct_visual_r{method[-1]}"][cid]["activity"])
                per_change[cid] = correction_utility(actual, initial, prediction.activity, actual >= 0)
            for day, row in enumerate(labels[cid]):
                timelines.append({"method": method, "case_id": cid, "day_index": day, "date": row["date"],
                                  "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                                  "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        changes[method] = per_change
        flow = costs.get(method, [])
        cost_count = len(flow) if flow else int(old_summary[method]["requests"])
        tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in flow) if flow else int(old_summary[method]["reported_tokens"])
        summaries.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_total,
                          "weak_days": weak_total, "coverage": aggregate["coverage"],
                          "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": score_counts(*event05)["f1"],
                          "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in boundaries])) if boundaries else None,
                          "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in boundaries])) if boundaries else None,
                          "stage_macro_f1": aggregate["stage_macro_f1"], "requests": cost_count, "reported_tokens": tokens})
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                cases = [cid for cid in ids if truth[cid][field] == group]
                measured = aggregate_scores([per_case[cid] for cid in cases])
                groups.append({"method": method, "field": field, "group": group, "records": len(cases), **measured["daily"]})
    stages = verify_metrics(predictions, results, labels, records)
    for row in summaries:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    index = {row["method"]: row for row in summaries}
    vote_changes, decisions = {}, []
    for rep in (0, 1):
        main, direct, vote = [f"{stem}_r{rep}" for stem in ("grouped_proposal_guard", "direct_visual", "numeric_vote")]
        by_case = {}
        for cid in ids:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            by_case[cid] = correction_utility(actual, np.array(predictions[vote][cid]["activity"]),
                                             np.array(predictions[main][cid]["activity"]), actual >= 0)
        vote_changes[str(rep)] = by_case
        definite_gain = sum(row["net_correct"] for row in by_case.values())
        a, d = index[main], index[direct]
        criteria = {"daily_gain": a["f1"] > d["f1"], "weak_nonloss": a["weak_recall"] >= d["weak_recall"],
                    "vote_f1_nonloss": a["f1"] >= index[vote]["f1"], "actual_gain_over_vote": definite_gain > 0,
                    "multiple_actual_gains": sum(row["net_correct"] > 0 for row in changes[main].values()) >= 2}
        decisions.append({"repeat": rep, "criteria": criteria, "passed": all(criteria.values()),
                          "strict_f1_gain_over_vote": a["f1"] > index[vote]["f1"], "net_definite_gain_over_vote": definite_gain})
    decision = {"allows_independent_confirmation": all(row["passed"] for row in decisions)
                and any(row["strict_f1_gain_over_vote"] for row in decisions), "repeats": decisions}
    for name, value in (("results.json", results), ("changes.json", changes), ("vote-relative-corrections.json", vote_changes),
                        ("decision.json", decision), ("internal-stages.json", stages)):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("groups.csv", groups), ("daily-predictions.csv", timelines)):
        write_rows(out / name, rows)
    for cid, (case, _) in records.items():
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        methods = [f"{stem}_r{rep}" for rep in (0, 1) for stem in
                   ("direct_visual", "grouped_proposal_guard", "visual_guard", "numeric_vote", "fixed_grouped_guard")]
        methods += ["xgboost_84", "xgboost_direction_information"]
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in methods)
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for axis in range(3):
            axes[axis].plot(raw[:, axis], linewidth=.8, color="#24887b")
            axes[axis].set_ylabel(f"{'NEU'[axis]} (mm)")
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                       cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                       extent=[-.5, len(raw)-.5, len(matrix)-.5, -.5])
        axes[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=7)
        axes[3].set_xlabel("Original day | blue active; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | Frozen grouped proposal acceptance | Complete exposed development")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=120)
        plt.close(fig)
    save_json(out / "state.json", {"status": "complete", "configs": len(results), "proposals_replayed": len(proposals),
                                  "numerical_predictions_recomputed": len(numerical) * len(ids),
                                  "old_payloads_verified": len(original_receipts), "complete_extra_payloads_verified": len(extra_receipts),
                                  "group_rows": len(groups), "daily_rows": len(timelines), "full_record_figures": len(ids),
                                  "new_requests": 0, "new_records": 0})
    print({"decision": decision, "main": [index[f"grouped_proposal_guard_r{rep}"] for rep in (0, 1)]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.calibration, args.complete, args.out)
