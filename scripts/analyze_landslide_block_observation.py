"""Evaluate the complete uniform block-view candidate on the common development set."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from probe_landslide_longer_motion_evidence import longer_conflict_scope
from probe_landslide_revision_scope import scoped_activity, target_scopes
from run_landslide_block_observation import verify
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_local_experiment import learned_stages


def analyze(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all uniform display answers before scoring")
    for file, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / file) != expected:
            raise ValueError("Uniform answers changed")
    config = read_json(run / "protocol.json")
    new_receipts = verify_request_receipts(run)
    screen = Path(config["source_screen"])
    screen_receipts = {row["request_id"]: row for row in verify_request_receipts(screen)}
    plan = read_json(run / "query-plan.json")
    if len(new_receipts) != config["maximum_new_requests"] or len(plan) != config["total_local_receipts"]:
        raise ValueError("Frozen request population differs")
    new_by_id = {row["request_id"]: row for row in new_receipts}
    receipts = {}
    for row in plan:
        if "cached_receipt" in row:
            path = Path(row["cached_receipt"])
            if sha(path) != row["cached_sha256"]:
                raise ValueError("Cached screen request changed")
            receipt = read_json(path)
            if receipt != screen_receipts[receipt["request_id"]]:
                raise ValueError("Cached receipt lacks an actual screen payload binding")
        else:
            receipt = new_by_id[row["request_id"]]
        receipts[row["request_id"]] = receipt
    records = {cid: load_case_features(run, cid) for cid in config["all_case_ids"]}
    labels = load_labels(run / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    codes = {cid: head.predict(features.matrix).astype(int) + 1 for cid, (_, features) in records.items()}
    initial = read_json(run / "initial-predictions.json")
    selections = read_json(run / "selections.json")
    reviews = read_json(run / "reviews.json")
    predictions = read_json(run / "prior-comparison-predictions.json")
    changes = []
    policy = ReviewPolicy()
    for rep in range(2):
        for arm in config["arms"]:
            method = f"uniform_{arm}_r{rep}"
            predictions[method] = {}
            for cid, (case, features) in records.items():
                first = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"], dtype=int)
                chosen = selections[f"{cid}-r{rep}"]["selected"]
                local = reviews.get(method, {}).get(cid, [])
                if sorted((row["start"], row["stop"]) for row in local) != sorted(
                        (row["start"], row["stop"]) for row in chosen):
                    raise ValueError("Uniform arm omitted or added a selected review")
                review_values = [{**row, "activity": np.array(row["activity"], dtype=int)} for row in local]
                combined = combine_reviews(first, review_values, features.observed).activity
                _, _, neighbourhood, _ = longer_conflict_scope(first, features,
                    raw_motion_evidence(case, features), raw_motion_evidence(case, features, 84), policy)
                scope, _, _ = target_scopes(neighbourhood, features.observed, chosen)
                activity = scoped_activity(first, combined, scope, features.observed)
                predictions[method][cid] = learned_stages(codes[cid], activity, features.supported).to_dict()
                original = np.array(predictions[f"adaptive_no_feedback_r{rep}__conflict_scope"][cid]["activity"])
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                removed = (original == 1) & (activity != 1)
                added = (original != 1) & (activity == 1)
                changes.append({"method": method, "case_id": cid,
                    "lost_tp_vs_base42": int(np.count_nonzero(removed & (actual == 1))),
                    "gained_tp_vs_base42": int(np.count_nonzero(added & (actual == 1))),
                    "removed_fp_vs_base42": int(np.count_nonzero(removed & (actual == 0))),
                    "gained_fp_vs_base42": int(np.count_nonzero(added & (actual == 0)))})
    if len(predictions) != 43:
        raise ValueError("Uniform comparison must retain all39 prior configurations and four new ones")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {"predictions.json": sha(out / "predictions.json")})
    results, summary, per_record = evaluate_population(predictions, labels)
    internal = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
    save_json(out / "results.json", results)
    save_json(out / "true-activity-stage-scores.json", internal)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-record.csv", per_record)
    csv_rows(out / "changes-vs-base42.csv", changes)
    indexed = {row["method"]: row for row in summary}
    decisions = []
    for rep in range(2):
        candidate, control = f"uniform_narrow_blocks_r{rep}", f"uniform_narrow_raw_r{rep}"
        base = f"adaptive_no_feedback_r{rep}__conflict_scope"
        a, b, c = indexed[candidate], indexed[control], indexed[base]
        steps = ("case_0074", "case_0080")
        step_fp = {name: sum(results[name]["per_case"][cid]["daily"]["fp"] for cid in steps)
                   for name in (candidate, control)}
        tp_difference = results[candidate]["per_case"]["case_0082"]["daily"]["tp"] - \
                        results[control]["per_case"]["case_0082"]["daily"]["tp"]
        row = {"repeat": rep + 1, "f1_difference_vs_uniform_raw": a["f1"] - b["f1"],
               "weak_recall_difference_vs_uniform_raw": a["weak_recall"] - b["weak_recall"],
               "f1_difference_vs_base42": a["f1"] - c["f1"],
               "step_fp_difference": step_fp[candidate] - step_fp[control],
               "case0082_tp_difference": tp_difference}
        row["passes"] = (row["f1_difference_vs_uniform_raw"] > 0 and row["weak_recall_difference_vs_uniform_raw"] > 0
                         and row["f1_difference_vs_base42"] >= 0 and row["step_fp_difference"] <= 0
                         and row["case0082_tp_difference"] >= 0)
        decisions.append(row)
    decision = {"pre_inference_stop": config["primary_stop"], "repetitions": decisions,
                "proceed_to_new_confirmation": all(row["passes"] for row in decisions),
                "scope": config["scope"], "screen_wide_primary_remains_rejected": True}
    save_json(out / "candidate-decision.json", decision)
    longer = Path(config["source_longer"])
    original = Path(read_json(longer / "protocol.json")["source_bundle"])
    method_costs = {}
    for rep in range(2):
        global_receipts = [read_json(original / "requests" / f"{cid}-direct-r{rep}.response.json")
                           for cid in config["all_case_ids"]]
        for arm in config["arms"]:
            selected = [receipts[row["request_id"]] for row in plan if row["repeat"] == rep and row["arm"] == arm]
            all_receipts = global_receipts + selected
            method_costs[f"uniform_{arm}_r{rep}"] = {
                "requests": len(all_receipts), "tokens": sum(row["usage"]["total_tokens"] for row in all_receipts),
                "seconds_model_receipts": sum(row["seconds"] for row in all_receipts),
                "scope": "Complete hypothetical method with shared real initial/cached local receipts, "
                         "never additive across methods or to the new unique ledger"}
    save_json(out / "costs.json", {"new_unique_requests": len(new_receipts),
        "new_unique_tokens": sum(row["usage"]["total_tokens"] for row in new_receipts),
        "cached_local_receipts": config["cached_identical_receipts"], "local_receipts": len(plan),
        "per_method": method_costs})
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.5))
    for rep in range(2):
        names = [f"adaptive_no_feedback_r{rep}__conflict_scope", f"extended84_evidence_scope_r{rep}",
                 f"uniform_narrow_raw_r{rep}", f"uniform_narrow_blocks_r{rep}",
                 "xgboost_42_matched_support", "xgboost_84"]
        for panel, metric in zip(panels, ("f1", "weak_recall")):
            panel.bar(np.arange(6) + (rep - .5) * .36, [indexed[name][metric] for name in names],
                      width=.34, label=f"repeat {rep + 1}")
            panel.set_xticks(range(6), ["base42 scope", "old84 scope", "uniform raw", "uniform blocks", "XGB42 same support", "XGB84"],
                             rotation=25, ha="right")
            panel.set_ylim(0, 1.02)
            panel.set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
            panel.grid(axis="y", alpha=.2)
            panel.legend()
    fig.suptitle("Viewed twelve-record development: uniform raw/block views; identical numerical bars")
    fig.tight_layout()
    fig.savefig(out / "uniform-block-development.png", dpi=180)
    plt.close(fig)
    save_json(out / "verification.json", {"same_inputs_and_stage_head": 12, "independent_metric_groups": 43,
        "new_actual_payloads": len(new_receipts), "cached_actual_payloads_used": config["cached_identical_receipts"],
        "all58_local_receipts_bound": True, "new_records": 0, "failed_requests": len(read_json(run / "failures.json")),
        "fitting_or_calibration_on_viewed_records": False})
    save_json(out / "manifest.json", {"source_bindings": {str(run / file): sha(run / file) for file in (
        "freeze.json", "prediction-freeze.json")},
        "files": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}})
    print("Uniform block candidate:", decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
