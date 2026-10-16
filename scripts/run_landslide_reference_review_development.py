"""Run a bounded full-record reference-observation prototype on two viewed groups."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from finalize_landslide_adaptive_evidence import verify_request_receipts
from fit_landslide_reference_controls import DEVELOPMENT_IDS, reference_information, residual_rule
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_reference_observation_probe_v2 import BASE, DATA, NOTICE, render_supplement
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
)
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import WINDOW_TASK, parse_window_activity

CONTROLS = BASE / "annual-reference-controls-v1"
PROBE = BASE / "reference-observation-development-v2"


def prepare(out):
    if not read_json(PROBE / "analysis-v1/candidate-decision.json")["proceed_to_complete_window_development"]:
        raise ValueError("The paired observation must pass before extending")
    for name, expected in read_json(CONTROLS / "manifest.json")["files"].items():
        if sha(CONTROLS / name) != expected:
            raise ValueError("Same-information controls changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    shutil.copyfile(CONTROLS / "annual-reference.json", out / "annual-reference.json")
    shutil.copyfile(CONTROLS / "stage-xgboost.json", out / "stage-xgboost.json")
    cached = read_json(DATA / "predictions.json")
    initial = {f"direct_visual_r{rep}": {cid: cached[f"direct_visual_r{rep}"][cid] for cid in DEVELOPMENT_IDS}
               for rep in (0, 1)}
    save_json(out / "initial-predictions.json", initial)
    protocol = {"scope": "Full730-day12-record development on viewed P227 background4/5, all six families per group",
        "case_ids": DEVELOPMENT_IDS, "repetitions": 2, "model": "qwen3.8-flash", "retries": 0,
        "initial": "Reuse identical frozen direct predictions from prior background run for all arms; no new global inference",
        "arms": ["raw", "reference_residual"], "flows": ["adaptive", "fixed"], "maximum_reviews": 2,
        "selection": "Original raw21/42 conflicts,14 count,.03mm/day,2MAD,49-day raw revision neighbourhood unchanged",
        "primary": "adaptive_reference_residual__conflict_scope", "ablation": "Matched raw removes reference/residual; fixed retains same information but replaces conflict selection",
        "reference": "Four frozen train-only annual references, no fitting current displacement or hard residual gate; offline group split",
        "stage": "Same learned raw background stage head, supported only in activity",
        "decision": "Both repetitions must improve full-record F1 and not lower weak recall vs matched adaptive raw. "
                    "Combined quiet/step FP must not increase per repetition. Report every record TP loss, all stages/events/costs. "
                    "Failure stops this extension without prompt or threshold rescue; passing requires a new frozen confirmation next.",
        "controls": "Original9 plus annual residual rule9 trials and augmented XGBoost6 trials, shared train24/validation12",
        "not_independent": "Backgrounds and paired probes already viewed; this evaluates implementation generalization over full records only"}
    reference = read_json(out / "annual-reference.json")
    old_selections = read_json(DATA / "selections.json")
    selections, plan, bindings = {}, [], {}
    order = np.random.default_rng(20261015)
    for cid in DEVELOPMENT_IDS:
        shutil.copytree(DATA / "inputs" / cid, out / "inputs" / cid)
        shutil.copyfile(DATA / "images" / f"{cid}-raw.png", out / "images" / f"{cid}-raw.png")
        case, feature = load_case_features(out, cid)
        windows = regular_windows(len(case.dates), 180, 90)
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            priorities, selected = review_priorities(np.array(initial[f"direct_visual_r{rep}"][cid]["activity"]),
                                                     feature, long, windows, ReviewPolicy())
            previous = old_selections[f"{cid}-r{rep}"]
            if selected != previous["selected"] or priorities != previous["priorities"]:
                raise ValueError("Original raw selection did not reproduce")
            adaptive = [row["index"] for row in selected]
            fixed = previous["fixed_indices"]
            selections[f"{cid}-r{rep}"] = {"selected": selected, "adaptive_indices": adaptive, "fixed_indices": fixed}
            for index in sorted(set(adaptive) | set(fixed)):
                left, right = windows[index]
                target = {"start": left, "stop": right}
                local = f"{cid}-window-{index:02d}.png"
                if not (out / "images" / local).exists():
                    render_observation_view(case, feature.noise, left, right, 0, False, out / "images" / local)
                for arm in order.permutation(protocol["arms"]):
                    supplement = f"{cid}-window-{index:02d}-supplement-{arm}.png"
                    if not (out / "images" / supplement).exists():
                        render_supplement(case, reference, target, str(arm), out / "images" / supplement)
                    images = [f"{cid}-raw.png", local, supplement]
                    if len(set(images)) != 3:
                        raise ValueError("Distinct images required")
                    plan.append({"case_id": cid, "repeat": rep, "index": index, "arm": str(arm),
                                 "start": left, "stop": right, "images": images,
                                 "request_id": f"{cid}-window-{index:02d}-{arm}-r{rep}",
                                 "prompt": WINDOW_TASK.format(start=left, end=right - 1) + NOTICE})
        if sha(out / "images" / f"{cid}-raw.png") != sha(DATA / "images" / f"{cid}-raw.png"):
            raise ValueError("Original global raw image changed")
        for name in (f"inputs/{cid}/input.json", f"images/{cid}-raw.png"):
            bindings[str(DATA / name)] = sha(DATA / name)
    protocol["maximum_requests"] = len(plan)
    save_json(out / "protocol.json", protocol)
    save_json(out / "selections.json", selections)
    save_json(out / "query-plan.json", plan)
    for source in (DATA / "prediction-freeze.json", CONTROLS / "manifest.json", PROBE / "analysis-v1/manifest.json"):
        bindings[str(source)] = sha(source)
    save_json(out / "source-bindings.json", bindings)
    for folder in (ROOT / "src/gnss_sim", ROOT / "scripts"):
        for source in folder.glob("*.py"):
            destination = out / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    save_json(out / "freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                    for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared_not_inferred"})
    print(f"Full12-record reference prototype frozen; {len(plan)} unique paired requests", flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen prototype changed: {name}")
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError(f"Executing source changed: {name}")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Source experiment changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared_not_inferred":
        raise ValueError("Run once from a fresh prepared prototype")
    config = read_json(out / "protocol.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    outputs, failures = {}, {}
    save_json(out / "state.json", {"status": "running"})
    for number, row in enumerate(read_json(out / "query-plan.json")):
        status = "success"
        try:
            payload = transport.ask(row["request_id"], row["prompt"], [out / "images" / name for name in row["images"]])
            activity = parse_window_activity(payload, records[row["case_id"]][1].observed, row["start"], row["stop"])
        except (RuntimeError, ValueError) as error:
            failures[row["request_id"]] = str(error)
            status = "failed"
            activity = np.full(len(records[row["case_id"]][1].observed), -1, dtype=int)
        outputs[row["request_id"]] = {**row, "activity": activity.tolist(), "status": status}
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        print(f"Reference full review {number + 1}/{config['maximum_requests']} {row['request_id']} {status}", flush=True)
    verify(out)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("outputs.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})


def analyze(out, analysis):
    verify(out)
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("All full-record outputs must freeze before reference access")
    for name, expected in read_json(out / "prediction-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen prediction changed")
    receipts = verify_request_receipts(out)
    config = read_json(out / "protocol.json")
    if len(receipts) != config["maximum_requests"]:
        raise ValueError("Incomplete paired requests")
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    outputs = read_json(out / "outputs.json")
    for row in outputs.values():
        if row["status"] == "success":
            payload = json.loads(read_json(out / "requests" / f"{row['request_id']}.response.json")["output"])
            parsed = parse_window_activity(payload, records[row["case_id"]][1].observed, row["start"], row["stop"])
            if parsed.tolist() != row["activity"]:
                raise ValueError("Receipt does not reproduce output")
    analysis.mkdir(parents=True, exist_ok=False)
    with (DATA / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reference_rows = [row for row in csv.DictReader(handle) if row["case_id"] in records]
    csv_rows(analysis / "reference-subset.csv", reference_rows)
    labels = load_labels(analysis / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    predictions = read_json(out / "initial-predictions.json")
    selections = read_json(out / "selections.json")
    scope_details, usages = [], {}
    for rep in (0, 1):
        for cid, (case, feature) in records.items():
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            long = raw_motion_evidence(case, feature)
            _, neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())
            codes = head.predict(feature.matrix).astype(int) + 1
            chosen = selections[f"{cid}-r{rep}"]
            for flow in ("adaptive", "fixed"):
                for arm in config["arms"]:
                    reviews = [outputs[f"{cid}-window-{index:02d}-{arm}-r{rep}"] for index in chosen[f"{flow}_indices"]]
                    gate = combine_reviews(initial, reviews, feature.observed).activity
                    conflict, center, details = target_scopes(neighbourhood, feature.observed, reviews)
                    for variant, scope in (("whole_target", None), ("conflict_scope", conflict), ("matched_center_scope", center)):
                        method = f"{flow}_{arm}_r{rep}__{variant}"
                        activity = gate if scope is None else scoped_activity(initial, gate, scope, feature.observed)
                        predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                        usages.setdefault(method, []).extend(row["request_id"] for row in reviews)
                    scope_details.extend({"repeat": rep + 1, "case_id": cid, "flow": flow, "arm": arm, **row} for row in details)
    numeric = recompute_numerical(BASE / "background-benchmark-data-v1", records)
    old_numeric = read_json(DATA / "numeric-predictions.json")
    for method, cases in numeric.items():
        for cid, value in cases.items():
            if value != old_numeric[method][cid]:
                raise ValueError("Original numerical predictions changed")
    predictions.update(numeric)
    annual = read_json(CONTROLS / "predictions.json")
    annual_head = XGBClassifier()
    annual_head.load_model(CONTROLS / "annual-augmented-xgboost.json")
    choices = read_json(CONTROLS / "frozen-selection.json")
    for cid, (case, feature) in records.items():
        matrix, evidence, support = reference_information(case, feature, read_json(out / "annual-reference.json"))
        rule, learned = choices["annual_residual_rule"], choices["annual_augmented_xgboost"]
        gates = {"annual_residual_rule": residual_rule(evidence, support, rule["rate"], rule["snr"]),
                 "annual_augmented_xgboost": activity_from_mask(annual_head.predict_proba(matrix)[:, 1] >= learned["threshold"], support)}
        codes = head.predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            if learned_stages(codes, gate, feature.supported).to_dict() != annual[method][cid]:
                raise ValueError("Same-reference numerical recomputation differs")
    predictions.update(annual)
    results, summary, per_record = evaluate_population(predictions, labels)
    stages = verify_metrics(predictions, results, labels, records)
    indexed = {row["method"]: row for row in summary}
    changes, decisions = [], []
    for rep in (0, 1):
        new_method, old_method = f"adaptive_reference_residual_r{rep}__conflict_scope", f"adaptive_raw_r{rep}__conflict_scope"
        for cid in records:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            new, old = np.array(predictions[new_method][cid]["activity"]), np.array(predictions[old_method][cid]["activity"])
            removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
            changes.append({"repeat": rep + 1, "case_id": cid,
                "lost_tp": int(np.count_nonzero(removed & (actual == 1))), "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                "removed_fp": int(np.count_nonzero(removed & (actual == 0))), "gained_fp": int(np.count_nonzero(added & (actual == 0)))})
        negatives = {f"case_{number}" for number in (2025, 2026, 2031, 2032)}
        neg_change = sum(row["gained_fp"] - row["removed_fp"] for row in changes if row["repeat"] == rep + 1 and row["case_id"] in negatives)
        new_score, old_score = indexed[new_method], indexed[old_method]
        decision = {"repeat": rep + 1, "f1_difference": new_score["f1"] - old_score["f1"],
                    "weak_recall_difference": new_score["weak_recall"] - old_score["weak_recall"], "negative_fp_difference": neg_change,
                    "total_lost_tp": sum(row["lost_tp"] for row in changes if row["repeat"] == rep + 1)}
        decision["passes"] = decision["f1_difference"] > 0 and decision["weak_recall_difference"] >= 0 and neg_change <= 0
        decisions.append(decision)
    receipt_map = {row["request_id"]: row for row in receipts}
    costs = {"new_unique_requests": len(receipts), "new_unique_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
             "global_initial_reused_requests": 24, "global_cost_not_new": True, "per_method": {}}
    for method, request_ids in usages.items():
        selected_receipts = [receipt_map[rid] for rid in request_ids]
        costs["per_method"][method] = {"new_local_requests": len(selected_receipts),
            "new_local_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in selected_receipts),
            "new_local_seconds": sum(row["seconds"] for row in selected_receipts)}
    save_json(analysis / "predictions.json", predictions)
    save_json(analysis / "results.json", results)
    save_json(analysis / "stage-metrics.json", stages)
    csv_rows(analysis / "summary.csv", summary)
    csv_rows(analysis / "per-record.csv", per_record)
    csv_rows(analysis / "changes.csv", changes)
    csv_rows(analysis / "scope-details.csv", scope_details)
    save_json(analysis / "decision.json", {"repetitions": decisions, "proceed_to_new_confirmation": all(row["passes"] for row in decisions),
        "not_independent_confirmation": True, "all_tp_losses_reported": True})
    save_json(analysis / "costs.json", costs)
    save_json(analysis / "verification.json", {"actual_payloads_bound": len(receipts), "independent_metric_groups": len(predictions),
        "original_numeric_recomputed": 108, "same_reference_numeric_recomputed": 24, "contracts_failed": len(read_json(out / "failures.json")),
        "raw_selection_reproduced": 24, "all_record_population": 12})
    save_json(analysis / "manifest.json", {"run_freeze_sha256": sha(out / "freeze.json"),
        "prediction_freeze_sha256": sha(out / "prediction-freeze.json"),
        "files": {path.relative_to(analysis).as_posix(): sha(path) for path in analysis.rglob("*") if path.is_file()}})
    print("Full-record prototype decision:", decisions, flush=True)
    print("New tokens:", costs["new_unique_tokens"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "run":
        execute(args.out)
    else:
        if args.analysis is None:
            parser.error("analyze requires --analysis")
        analyze(args.out, args.analysis)


if __name__ == "__main__":
    main()
