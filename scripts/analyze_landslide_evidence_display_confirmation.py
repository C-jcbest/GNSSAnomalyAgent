"""Score every frozen unseen confirmation, including rejected candidates."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import paired_record_interval
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_evidence_display_confirmation import (
    load_model,
    numerical_predictions,
    select_workflows,
    verify,
)
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import combine_reviews
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity


def replay_predictions(run, predictions, records):
    """Reparse actual replies and reconstruct every output without labels or API."""
    head = load_model(run, "stage-xgboost.json")
    saved_selections = read_json(run / "selections.json")
    reviews = read_json(run / "reviews.json")
    failures = read_json(run / "failures.json")
    usages = read_json(run / "method-request-usage.json")
    for rep in range(2):
        for cid, (case, features) in records.items():
            direct_id = f"{cid}-direct-r{rep}"
            receipt = read_json(run / "requests" / f"{direct_id}.response.json")
            initial = np.full(len(features.observed), -1, dtype=int)
            if direct_id not in failures:
                initial = parse_activity(json.loads(receipt["output"]), features.observed)
            selected, scopes, diagnostics = select_workflows(initial, case, features)
            saved = saved_selections[f"{cid}-r{rep}"]
            if saved["initial"] != initial.tolist() or saved["methods"] != selected:
                raise ValueError("Saved selection or initial answer differs from replay")
            if saved["display_choices"] != diagnostics["display_choices"]:
                raise ValueError("Display choice changed after local answers")
            codes = head.predict(features.matrix).astype(int) + 1
            for name, targets in selected.items():
                parsed = []
                for target in targets:
                    request_id = f"{cid}-window-{target['index']:02d}-{target['arm']}-r{rep}"
                    row = reviews[request_id]
                    receipt = read_json(run / "requests" / f"{request_id}.response.json")
                    activity = np.full(len(initial), -1, dtype=int)
                    if request_id not in failures:
                        activity = parse_window_activity(json.loads(receipt["output"]), features.observed,
                                                         target["start"], target["stop"])
                    if row["activity"] != activity.tolist():
                        raise ValueError("Local activity differs from the actual response")
                    parsed.append({**row, "activity": activity})
                combined = combine_reviews(initial, parsed, features.observed).activity
                if name in scopes:
                    if saved["scope_days"][name] != np.flatnonzero(scopes[name]).tolist():
                        raise ValueError("Saved authority differs from replay")
                    combined = np.where(scopes[name], combined, initial)
                method = f"{name}_r{rep}"
                replay = learned_stages(codes, combined, features.supported)
                if name == "direct_visual" and direct_id in failures:
                    replay.status = "failed"
                if predictions[method][cid] != replay.to_dict():
                    raise ValueError(f"Frozen workflow does not replay: {method}/{cid}")
                expected_ids = [direct_id] + [row["request_id"] for row in parsed]
                actual_ids = [value for value in usages[method] if value.startswith(cid + "-")]
                if actual_ids != expected_ids or len(expected_ids) > 3:
                    raise ValueError("Per-method request budget or usage differs")
    for method, cases in numerical_predictions(run, records).items():
        if cases != predictions[method]:
            raise ValueError(f"Frozen numerical competitor does not replay: {method}")


def analyze(run, out):
    verify(run, generated=True)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("References remain closed until all predictions complete")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Frozen prediction artifact changed")
    config = read_json(run / "protocol.json")
    records = {cid: load_case_features(run, cid) for cid in config["generation"]["case_ids"]}
    predictions = read_json(run / "predictions.json")
    expected_methods = set(config["numeric_controls"]) | {
        f"{name}_r{rep}" for rep in range(2) for name in config["visual_controls"]}
    if set(predictions) != expected_methods or any(set(cases) != set(records) for cases in predictions.values()):
        raise ValueError("All 25 configurations must score on exactly the same twelve records")
    replay_predictions(run, predictions, records)
    receipts = verify_request_receipts(run)
    usage = read_json(run / "method-request-usage.json")
    by_id = {row["request_id"]: row for row in receipts}
    if set(by_id) != {value for ids in usage.values() for value in ids} or len(receipts) > 312:
        raise ValueError("Actual unique calls differ from shared workflow ledger")
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    results, summary, per_record = evaluate_population(predictions, labels)
    internal = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    families = {cid: read_json(run / "mechanism-reference" / cid / "truth.json")["family"] for cid in records}
    negative_ids = [cid for cid, family in families.items() if family in ("quiet", "instrument_step")]
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
        row["quiet_step_fp"] = sum(results[row["method"]]["per_case"][cid]["daily"]["fp"] for cid in negative_ids)
    indexed = {row["method"]: row for row in summary}
    decisions, comparisons, changes, intervals = [], [], [], {}
    for rep in range(2):
        candidate = f"adaptive84_evidence_display_r{rep}"
        raw84 = f"adaptive84_narrow_raw_r{rep}"
        raw42 = f"adaptive42_narrow_raw_r{rep}"
        a, b, c = indexed[candidate], indexed[raw84], indexed[raw42]
        decision = {"repeat": rep + 1, "f1_difference_vs_raw84": a["f1"] - b["f1"],
                    "weak_recall_difference_vs_raw84": a["weak_recall"] - b["weak_recall"],
                    "f1_difference_vs_raw42": a["f1"] - c["f1"],
                    "quiet_step_fp_difference_vs_raw84": a["quiet_step_fp"] - b["quiet_step_fp"]}
        decision["passes"] = (decision["f1_difference_vs_raw84"] > 0
                               and decision["weak_recall_difference_vs_raw84"] > 0
                               and decision["f1_difference_vs_raw42"] >= 0
                               and decision["quiet_step_fp_difference_vs_raw84"] <= 0)
        decisions.append(decision)
        controls = [name for name in predictions if name != candidate
                    and (name.endswith(f"_r{rep}") or name in config["numeric_controls"])]
        for control in controls:
            b = indexed[control]
            comparisons.append({"candidate": candidate, "control": control,
                **{f"difference_{metric}": a[metric] - b[metric] for metric in (
                    "f1", "weak_recall", "tp", "fp", "event_f1_iou03", "event_f1_iou05", "stage_macro_f1")}})
            for cid in records:
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                new = np.array(predictions[candidate][cid]["activity"])
                old = np.array(predictions[control][cid]["activity"])
                removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
                changes.append({"candidate": candidate, "control": control, "case_id": cid, "family": families[cid],
                    "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                    "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                    "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                    "gained_fp": int(np.count_nonzero(added & (actual == 0)))})
        for control in (raw84, raw42, "xgboost_84"):
            intervals[f"{candidate}-minus-{control}"] = paired_record_interval(results, control, candidate)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "results.json", results)
    save_json(out / "true-activity-stage-scores.json", internal)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-record.csv", per_record)
    csv_rows(out / "comparisons.csv", comparisons)
    csv_rows(out / "changes.csv", changes)
    save_json(out / "exploratory-record-intervals.json", intervals)
    decision = {"registered_stop": config["stop"], "repetitions": decisions,
        "promote": all(row["passes"] for row in decisions),
        "scope": "Frozen new random-stream controlled mechanism confirmation, not field/expert validation",
        "retuning": False}
    save_json(out / "candidate-decision.json", decision)
    method_costs = {}
    for method, ids in usage.items():
        selected_receipts = [by_id[value] for value in ids]
        method_costs[method] = {"requests": len(ids),
            "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in selected_receipts),
            "seconds_model_receipts": sum(row["seconds"] for row in selected_receipts)}
    costs = {"new_unique_requests": len(receipts),
        "new_unique_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
        "sum_response_seconds": sum(row["seconds"] for row in receipts), "per_method": method_costs,
        "numeric": read_json(run / "numeric-cost.json"),
        "scope": "Actual unique ledger; hypothetical per-method shared receipt sums are never additive"}
    save_json(out / "costs.json", costs)
    fig, panels = plt.subplots(1, 2, figsize=(15, 5))
    names = config["visual_controls"] + ["raw_rule_84", "xgboost_42_matched_support", "xgboost_84"]
    for rep in range(2):
        for panel, metric in zip(panels, ("f1", "weak_recall")):
            values = [indexed[f"{name}_r{rep}" if name in config["visual_controls"] else name][metric] for name in names]
            panel.bar(np.arange(len(names)) + (rep - .5) * .36, values, width=.34, label=f"repeat {rep + 1}")
            panel.set_xticks(range(len(names)), [name.replace("_narrow_", " ").replace("_budget2", "")
                            .replace("adaptive84_evidence_display", "evidence display") for name in names],
                            rotation=52, ha="right", fontsize=8)
            panel.set_ylim(0, 1.02)
            panel.set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
            panel.legend()
            panel.grid(axis="y", alpha=.2)
    fig.suptitle("Unseen controlled case0085-0096: frozen protocol, all repetitions retained")
    fig.tight_layout()
    fig.savefig(out / "confirmation-comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# 证据显示未见受控确认", "", "新0085–0096，25配置；潜在机制参考，不是领域专家观测标签。", "",
        f"登记判定：{'通过' if decision['promote'] else '不通过，不升级候选'}；不在本批救分。", "",
        "| 方法 | P | R | F1 | 弱召回 | FP | 事件.3 | 事件.5 | 阶段macro | 真活动阶段 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                     f"{row['weak_recall']:.4f} | {row['fp']} | {row['event_f1_iou03']:.4f} | "
                     f"{row['event_f1_iou05']:.4f} | {row['stage_macro_f1']:.4f} | "
                     f"{row['true_activity_stage_macro_f1']:.4f} |")
    lines += ["", f"新唯一请求{costs['new_unique_requests']}，token {costs['new_unique_tokens']}。",
              "所有视觉活动配置共用学习阶段头，不将阶段成绩称纯VLM成绩。", "",
              "停止条件、全部TP损失、边界、分记录及探索性整记录bootstrap见同目录JSON/CSV。"]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(out / "verification.json", {"replayed_visual_configurations": 16, "replayed_numeric_configurations": 9,
        "independent_sklearn_activity_stage_groups": 25, "event_boundary_groups": 25,
        "actual_payload_bindings": len(receipts), "failed_requests": len(read_json(run / "failures.json")),
        "unseen_before_execution_freeze": True, "expert_observable_reference": False})
    save_json(out / "manifest.json", {"source_bindings": {str(run / name): sha(run / name) for name in (
        "pre-generation-freeze.json", "data-freeze.json", "prediction-freeze.json")},
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Unseen confirmation decision:", decision, flush=True)
    print("Actual unique cost:", costs["new_unique_requests"], costs["new_unique_tokens"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
