"""Replay one evidence-specific display rule on the complete paired view cache."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from finalize_landslide_adaptive_evidence import verify_metrics
from probe_landslide_longer_motion_evidence import longer_conflict_scope
from probe_landslide_revision_scope import scoped_activity, target_scopes
from run_landslide_block_observation import verify
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_evidence_observation import evidence_display_choices
from gnss_sim.landslide_local_experiment import learned_stages


def probe(run, out):
    verify(run)
    analysis = run / "analysis-v1"
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Uniform view analysis changed")
    config = read_json(run / "protocol.json")
    records = {cid: load_case_features(run, cid) for cid in config["all_case_ids"]}
    labels = load_labels(run / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    first = read_json(run / "initial-predictions.json")
    selected = read_json(run / "selections.json")
    reviews = read_json(run / "reviews.json")
    predictions = read_json(analysis / "predictions.json")
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    choices = {}
    policy = ReviewPolicy()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    shutil.copyfile(ROOT / "src/gnss_sim/landslide_evidence_observation.py", out / "display-policy-source.py")
    save_json(out / "protocol.json", {
        "scope": "Post-outcome developer-selected cache mechanism, not independent or pre-inference confirmation",
        "rule": "On frozen84 selected targets, use narrow_blocks only if base42 conflict days <14; "
                "otherwise narrow_raw. Reuse existing conflict threshold, never labels or local answers",
        "basis": "Universal blocks recover0081 but suppress133 weak days in0078 and add platform FP; "
                 "retain raw display on targets already supported by original42 evidence",
        "new_requests": 0, "new_parameters": 0, "repetitions": 2,
        "selection_independence": "Rule chosen after seeing complete uniform answers and labels; "
                                  "scoring this cache cannot independently confirm it"})
    for rep in range(2):
        method = f"evidence_display_r{rep}"
        predictions[method] = {}
        for cid, (case, features) in records.items():
            initial = np.array(first[f"direct_visual_r{rep}"][cid]["activity"], dtype=int)
            targets = selected[f"{cid}-r{rep}"]["selected"]
            h42 = raw_motion_evidence(case, features)
            choice = evidence_display_choices(initial, features, h42, targets, policy)
            choices[f"{cid}-r{rep}"] = choice
            picked = []
            for row in choice:
                matches = [value for value in reviews[f"uniform_{row['arm']}_r{rep}"][cid]
                           if value["start"] == row["start"] and value["stop"] == row["stop"]]
                if len(matches) != 1:
                    raise ValueError("Evidence-specific view lacks an exact matched cached answer")
                picked.append({**matches[0], "activity": np.array(matches[0]["activity"], dtype=int)})
            combined = combine_reviews(initial, picked, features.observed).activity
            _, _, neighbourhood, _ = longer_conflict_scope(initial, features, h42,
                raw_motion_evidence(case, features, 84), policy)
            scope, _, _ = target_scopes(neighbourhood, features.observed, targets)
            activity = scoped_activity(initial, combined, scope, features.observed)
            codes = head.predict(features.matrix).astype(int) + 1
            predictions[method][cid] = learned_stages(codes, activity, features.supported).to_dict()
    save_json(out / "choices.json", choices)
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
    indexed = {row["method"]: row for row in summary}
    comparisons, costs = [], {}
    actual_receipts = read_json(run / "query-plan.json")
    source = Path(config["source_longer"])
    original = Path(read_json(source / "protocol.json")["source_bundle"])
    for rep in range(2):
        name = f"evidence_display_r{rep}"
        for control in (f"uniform_narrow_raw_r{rep}", f"uniform_narrow_blocks_r{rep}",
                        f"adaptive_no_feedback_r{rep}__conflict_scope"):
            a, b = indexed[name], indexed[control]
            comparisons.append({"repeat": rep + 1, "control": control,
                "f1_difference": a["f1"] - b["f1"], "weak_recall_difference": a["weak_recall"] - b["weak_recall"],
                "tp_difference": a["tp"] - b["tp"], "fp_difference": a["fp"] - b["fp"]})
        receipts = [read_json(original / "requests" / f"{cid}-direct-r{rep}.response.json") for cid in records]
        block_count = 0
        for cid in records:
            for choice in choices[f"{cid}-r{rep}"]:
                block_count += choice["arm"] == "narrow_blocks"
                rows = [row for row in actual_receipts if row["case_id"] == cid and row["repeat"] == rep
                        and row["index"] == choice["index"] and row["arm"] == choice["arm"]]
                row = rows[0]
                path = Path(row["cached_receipt"]) if "cached_receipt" in row else run / "requests" / f"{row['request_id']}.response.json"
                receipts.append(read_json(path))
        costs[name] = {"requests": len(receipts), "tokens": sum(row["usage"]["total_tokens"] for row in receipts),
                       "seconds_model_receipts": sum(row["seconds"] for row in receipts), "block_views": block_count,
                       "scope": "Hypothetical full method using actual selected receipts, 0 new requests in this replay"}
    save_json(out / "comparisons.json", comparisons)
    save_json(out / "costs.json", {"new_unique_requests": 0, "new_unique_tokens": 0, "per_method": costs})
    fig, panels = plt.subplots(1, 2, figsize=(12, 4.5))
    for rep in range(2):
        names = [f"uniform_narrow_raw_r{rep}", f"uniform_narrow_blocks_r{rep}", f"evidence_display_r{rep}",
                 f"adaptive_no_feedback_r{rep}__conflict_scope", "xgboost_84"]
        for panel, metric in zip(panels, ("f1", "weak_recall")):
            panel.bar(np.arange(5) + (rep - .5) * .36, [indexed[name][metric] for name in names],
                      width=.34, label=f"repeat {rep + 1}")
            panel.set_xticks(range(5), ["uniform raw", "uniform blocks", "evidence-specific", "base42 scope", "XGB84"],
                             rotation=25, ha="right")
            panel.set_ylim(0, 1.02)
            panel.set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
            panel.legend()
            panel.grid(axis="y", alpha=.2)
    fig.suptitle("Post-outcome viewed cache: evidence-specific display needs new frozen confirmation")
    fig.tight_layout()
    fig.savefig(out / "evidence-display-development.png", dpi=180)
    plt.close(fig)
    save_json(out / "verification.json", {"independent_metric_groups": len(predictions),
        "new_requests": 0, "same_inputs_and_stage_head": 12, "choice_rule_reads_local_answers": False,
        "choice_rule_reads_labels": False, "rule_selection_itself_is_post_outcome": True})
    save_json(out / "manifest.json", {"source_bindings": {str(analysis / "manifest.json"): sha(analysis / "manifest.json")},
        "files": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}})
    print("Evidence-specific cached development:", comparisons, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe(args.run, args.out)


if __name__ == "__main__":
    main()
