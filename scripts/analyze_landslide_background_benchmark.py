"""Evaluate frozen full-record reviews on grouped observed-background injections."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from finalize_landslide_adaptive_evidence import verify_request_receipts
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_local_experiment import learned_stages


def verify_data(source):
    manifest = read_json(source / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Background input or fitted artifact changed: {name}")
    for name, expected in read_json(source / "pre-generation-freeze.json").items():
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError(f"Executing source changed since background preparation: {name}")
    return manifest


def analyze(run, out):
    verify_freeze(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only after both full repetitions freeze")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Visual outputs changed after inference")
    config = read_json(run / "protocol.json")
    source = ROOT / config["source_bundle"]
    data_manifest = verify_data(source)
    if config["case_ids"] != data_manifest["split"]["mechanism_test"]:
        raise ValueError("Visual and numerical test calendars differ")
    receipts = verify_request_receipts(run)
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    # Exact injected labels become scoreable only after the entire model run.
    with (source / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        references = [row for row in csv.DictReader(handle) if row["case_id"] in records]
    out.mkdir(parents=True, exist_ok=False)
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    labels = load_labels(out / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    cached = read_json(run / "predictions.json")
    selections = read_json(run / "selections.json")
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    codes = {cid: head.predict(row[1].matrix).astype(int) + 1 for cid, row in records.items()}
    predictions, scope_rows, usage = {}, [], {}
    for rep in (0, 1):
        for cid, (case, feature) in records.items():
            initial = np.array(cached[f"direct_visual_r{rep}"][cid]["activity"])
            long = raw_motion_evidence(case, feature)
            _, neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())
            selection = selections[f"{cid}-r{rep}"]
            all_windows = regular_windows(len(initial), 180, 90)
            # Fixed target indices are taken from the actual frozen run, not
            # reconstructed by assuming a centre or selecting a nicer target.
            for arm in ("fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"):
                if arm == "fixed_image_review":
                    targets = [{"index": index, "start": all_windows[index][0], "stop": all_windows[index][1]}
                               for index in selection["fixed_indices"]]
                else:
                    targets = selection["selected"]
                conflict, center, detail = target_scopes(neighbourhood, feature.observed, targets)
                reviewed = np.array(cached[f"{arm}_r{rep}"][cid]["activity"])
                requests = [f"{cid}-direct-r{rep}"]
                suffix = "numeric" if arm == "adaptive_numeric_feedback" else "image"
                requests += [f"{cid}-window-{row['index']:02d}-{suffix}-r{rep}" for row in targets]
                for variant, scope in (("whole_target", None), ("conflict_scope", conflict), ("matched_center_scope", center)):
                    method = f"{arm}_r{rep}__{variant}"
                    gate = reviewed if scope is None else scoped_activity(initial, reviewed, scope, feature.observed)
                    predictions.setdefault(method, {})[cid] = learned_stages(codes[cid], gate, feature.supported).to_dict()
                    usage.setdefault(method, []).extend(requests)
                scope_rows.extend({"repeat": rep + 1, "case_id": cid, "arm": arm, **row} for row in detail)
            method = f"direct_visual_r{rep}"
            predictions.setdefault(method, {})[cid] = cached[method][cid]
            usage.setdefault(method, []).append(f"{cid}-direct-r{rep}")
    predictions.update(read_json(run / "numeric-predictions.json"))
    results, summary, per_record = evaluate_population(predictions, labels)
    stages = verify_metrics(predictions, results, labels, records)
    for row in summary:
        row["true_activity_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summary}
    changes, decisions = [], []
    for rep in (0, 1):
        candidate = f"adaptive_no_feedback_r{rep}__conflict_scope"
        control = f"adaptive_no_feedback_r{rep}__whole_target"
        lost, removed = 0, 0
        for cid in records:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            new = np.array(predictions[candidate][cid]["activity"])
            old = np.array(predictions[control][cid]["activity"])
            deleted = (old == 1) & (new != 1)
            added = (old != 1) & (new == 1)
            counts = {"lost_tp": int(np.count_nonzero(deleted & (actual == 1))),
                "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                "removed_fp": int(np.count_nonzero(deleted & (actual == 0))),
                "gained_fp": int(np.count_nonzero(added & (actual == 0)))}
            lost += counts["lost_tp"]
            removed += counts["removed_fp"]
            changes.append({"repeat": rep + 1, "case_id": cid, **counts})
        new, old = indexed[candidate], indexed[control]
        decision = {"repeat": rep + 1, "lost_tp": lost, "removed_fp": removed,
            "fp_difference": new["fp"] - old["fp"], "f1_difference": new["f1"] - old["f1"],
            "weak_recall_difference": new["weak_recall"] - old["weak_recall"]}
        decision["replicated"] = (lost == 0 and decision["fp_difference"] < 0
            and decision["f1_difference"] >= 0 and decision["weak_recall_difference"] >= 0)
        decisions.append(decision)
    truth_map = {cid: read_json(source / "mechanism-reference" / cid / "truth.json") for cid in records}
    groups = {f"background{group}": [cid for cid, truth in truth_map.items() if truth["background_group"] == group]
              for group in sorted({truth["background_group"] for truth in truth_map.values()})}
    groups.update({family: [cid for cid, truth in truth_map.items() if truth["family"] == family]
                  for family in sorted({truth["family"] for truth in truth_map.values()})})
    group_rows = []
    for group, ids in groups.items():
        _, rows, _ = evaluate_population({method: {cid: cases[cid] for cid in ids}
                                        for method, cases in predictions.items()}, {cid: labels[cid] for cid in ids})
        group_rows.extend({"group": group, **row} for row in rows)
    receipt_map = {row["request_id"]: row for row in receipts}
    costs = {"new_unique_requests": len(receipts),
        "new_unique_tokens": sum(row["usage"]["total_tokens"] for row in receipts),
        "scope": "Method costs share actual global/local requests; do not sum across arms",
        "per_method": {method: {"requests": len(ids),
            "tokens": sum(receipt_map[rid]["usage"]["total_tokens"] for rid in ids),
            "seconds": sum(receipt_map[rid]["seconds"] for rid in ids)} for method, ids in usage.items()}}
    save_json(out / "predictions.json", predictions)
    save_json(out / "results.json", results)
    save_json(out / "stage-metrics.json", stages)
    save_json(out / "replication-decision.json", {"repetitions": decisions,
        "both_replicated": all(row["replicated"] for row in decisions), "no_method_switch": True})
    save_json(out / "costs.json", costs)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-record.csv", per_record)
    csv_rows(out / "group-summary.csv", group_rows)
    csv_rows(out / "changes.csv", changes)
    csv_rows(out / "revision-scopes.csv", scope_rows)
    chosen = [f"{name}_r{rep}__conflict_scope" for name in ("fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback")
              for rep in (0, 1)] + ["direct_visual_r0", "direct_visual_r1", "xgboost_42_matched_support", "xgboost_84"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    for ax, metric in zip(axes, ("f1", "weak_recall", "fp")):
        ax.barh(range(len(chosen)), [indexed[name][metric] for name in chosen], color="#367b9c")
        ax.set_yticks(range(len(chosen)), chosen if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Held-out observed-background injections | full730-day comparison | no field-accuracy claim")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    save_json(out / "verification.json", {"complete_record_metric_groups": len(predictions),
        "group_metric_rows": len(group_rows), "background_blocks": len(groups) - 6,
        "actual_payloads_bound": len(receipts), "unknown_and_missing_retained": True,
        "no_test_tuning": True, "expert_field_reference": False})
    save_json(out / "manifest.json", {"run_freeze_sha256": sha(run / "freeze.json"),
        "data_manifest_sha256": sha(source / "manifest.json"), "prediction_freeze_sha256": sha(run / "prediction-freeze.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Scope replication:", decisions, flush=True)
    print("New requests/tokens:", costs["new_unique_requests"], costs["new_unique_tokens"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
