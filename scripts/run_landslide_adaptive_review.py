"""Freeze and execute a paired bounded Agent pilot, then score frozen predictions."""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
from run_landslide_comparison import (
    ROOT,
    load_labels,
    plot_case,
    plot_stages,
    save_json,
    sha,
)
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity

PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def prepare(config_path, out):
    config = read_json(config_path)
    source = ROOT / config["source_bundle"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    shutil.copyfile(config_path, out / "protocol.json")
    for name in ("stage-xgboost.json", "numeric-predictions.json", "reference-labels.csv"):
        shutil.copyfile(source / name, out / name)
    for cid in config["case_ids"]:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(source / "inputs" / cid / name, target / name)
        case, features = load_case_features(out, cid)
        render_raw(case, out / "images" / f"{cid}-raw.png")
        windows = regular_windows(len(case.dates), config["window_days"], config["stride_days"])
        for index, (start, stop) in enumerate(windows):
            render_window(case, features.noise, start, stop,
                          out / "images" / f"{cid}-window-{index:02d}.png", PLOT)
    (out / "source").mkdir()
    for path in [Path(__file__), ROOT / "src/gnss_sim/landslide_adaptive_review.py",
                 ROOT / "src/gnss_sim/landslide_visual.py",
                 ROOT / "src/gnss_sim/landslide_evaluation.py",
                 ROOT / "src/gnss_sim/landslide_window_experiment.py",
                 ROOT / "src/gnss_sim/landslide_local_experiment.py",
                 ROOT / "src/gnss_sim/landslide_error_audit.py",
                 ROOT / "src/gnss_sim/visual.py",
                 ROOT / "scripts/run_landslide_window_optimization.py",
                 ROOT / "scripts/run_landslide_visual_comparison.py",
                 ROOT / "scripts/run_landslide_comparison.py"]:
        shutil.copyfile(path, out / "source" / path.name)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "scope": config["scope"]})


def verify_freeze(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    for name in ("run_landslide_adaptive_review.py", "landslide_adaptive_review.py",
                 "landslide_visual.py", "landslide_evaluation.py",
                 "landslide_window_experiment.py", "landslide_local_experiment.py",
                 "landslide_error_audit.py", "visual.py",
                 "run_landslide_window_optimization.py", "run_landslide_visual_comparison.py",
                 "run_landslide_comparison.py"):
        parent = "scripts" if name.startswith("run_") else "src/gnss_sim"
        if sha(ROOT / parent / name) != sha(out / "source" / name):
            raise ValueError(f"Executing source differs from snapshot: {name}")


def execute(out):
    verify_freeze(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a new prepared run; no retries or overwrite")
    config = read_json(out / "protocol.json")
    policy = ReviewPolicy(**config["policy"])
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               config["maximum_requests"])
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    predictions, selections, failures = {}, {}, {}
    save_json(out / "state.json", {"status": "running"})
    started = time.monotonic()
    for rep in range(config["repetitions"]):
        for ordinal, cid in enumerate(config["case_ids"]):
            if time.monotonic() - started > config["hard_timeout_seconds"]:
                raise RuntimeError("Frozen wall-time budget exceeded")
            case, features = load_case_features(out, cid)
            raw = out / "images" / f"{cid}-raw.png"
            request_id = f"{cid}-direct-r{rep}"
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200].")
            try:
                payload = transport.ask(request_id, prompt, [raw])
                initial = parse_activity(payload, features.observed)
                initial_status = "success"
            except (ValueError, RuntimeError) as error:
                failures[request_id] = str(error)
                initial = np.full(len(case.dates), -1)
                initial_status = "failed"
            long_evidence = raw_motion_evidence(case, features)
            windows = regular_windows(len(case.dates), config["window_days"], config["stride_days"])
            priorities, selected = review_priorities(initial, features, long_evidence,
                                                    windows, policy)
            # Equal-spacing targets are fixed before observing any signal or prediction.
            centers = np.linspace(0, len(case.dates), policy.maximum_reviews + 2)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index:
                            abs(sum(windows[index]) / 2 - center)) for center in centers})
            adaptive = {row["index"] for row in selected}
            selection_key = f"{cid}-r{rep}"
            selections[selection_key] = {"priorities": priorities, "selected": selected,
                                         "fixed_indices": fixed}
            # Save every decision before requesting any local observation.
            save_json(out / "selections.json", selections)
            review_outputs = {}
            for index in sorted(set(fixed) | adaptive):
                start, stop = windows[index]
                conditions = [False]
                if index in adaptive:
                    conditions.append(True)
                if (ordinal + rep + index) % 2:
                    conditions.reverse()
                for numeric_feedback in conditions:
                    arm = "numeric" if numeric_feedback else "image"
                    request_id = f"{cid}-window-{index:02d}-{arm}-r{rep}"
                    prompt = review_prompt(features, long_evidence, start, stop, numeric_feedback)
                    local = out / "images" / f"{cid}-window-{index:02d}.png"
                    try:
                        payload = transport.ask(request_id, prompt, [raw, local])
                        activity = parse_window_activity(payload, features.observed, start, stop)
                        status = "success"
                    except (ValueError, RuntimeError) as error:
                        failures[request_id] = str(error)
                        activity = np.full(len(case.dates), -1)
                        status = "failed"
                    review_outputs[index, numeric_feedback] = {
                        "start": start, "stop": stop, "activity": activity.tolist(),
                        "status": status, "request_id": request_id}
            codes = model.predict(features.matrix).astype(int) + 1
            gates = {"direct_visual": ActivityPrediction(
                initial, np.where(initial == 0, 0, -1), status=initial_status),
                "fixed_image_review": combine_reviews(initial, [review_outputs[index, False]
                    for index in fixed], features.observed),
                "adaptive_no_feedback": combine_reviews(initial, [review_outputs[index, False]
                    for index in sorted(adaptive)], features.observed),
                "adaptive_numeric_feedback": combine_reviews(initial, [review_outputs[index, True]
                    for index in sorted(adaptive)], features.observed)}
            for name, gate in gates.items():
                if gate.status == "failed":
                    prediction = gate
                else:
                    prediction = learned_stages(codes, gate.activity, features.supported)
                predictions.setdefault(f"{name}_r{rep}", {})[cid] = prediction.to_dict()
            save_json(out / "predictions.json", predictions)
            save_json(out / "failures.json", failures)
            print(f"{cid} repeat {rep + 1}: adaptive {sorted(adaptive)}, fixed {fixed}, "
                  f"requests {len(list((out / 'requests').glob('*.response.json')))}", flush=True)
    verify_freeze(out)
    save_json(out / "prediction-freeze.json", {
        name: sha(out / name) for name in ("predictions.json", "failures.json", "selections.json")})
    save_json(out / "state.json", {"status": "predictions_complete",
                                   "seconds": time.monotonic() - started})


def score(out):
    verify_freeze(out)
    if read_json(out / "state.json")["status"] not in {"predictions_complete", "complete"}:
        raise ValueError("Score only after all inference outputs have been frozen")
    for name, expected in read_json(out / "prediction-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Inference changed after freezing")
    config = read_json(out / "protocol.json")
    cases = {cid: load_case_features(out, cid)[0] for cid in config["case_ids"]}
    # Reference access is confined to scoring, after the entire run is frozen.
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["case_id"] in cases]
        columns = reader.fieldnames
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    labels = load_labels(out / "reference-subset.csv", cases)
    serialized = read_json(out / "predictions.json")
    numeric = read_json(out / "numeric-predictions.json")
    for name in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage"):
        serialized[name] = {cid: numeric[name][cid] for cid in cases}
    methods, results = {}, {}
    for name, records in serialized.items():
        methods[name] = {}
        for cid, value in records.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                status=value.get("status", "success"), gated=value.get("gated", True),
                stage_status=value.get("stage_status", "success"))
            methods[name][cid] = prediction
        per_case = {cid: evaluate_case(labels[cid], prediction)
                    for cid, prediction in methods[name].items()}
        results[name] = {"aggregate": aggregate_scores(list(per_case.values())),
                         "per_case": per_case}
    save_json(out / "results.json", results)
    costs = {}
    for rep in range(config["repetitions"]):
        for method in ("direct_visual", "fixed_image_review", "adaptive_no_feedback",
                       "adaptive_numeric_feedback"):
            receipts = []
            for cid in cases:
                receipts.append(read_json(out / "requests" / f"{cid}-direct-r{rep}.response.json"))
                selection = read_json(out / "selections.json")[f"{cid}-r{rep}"]
                if method == "direct_visual":
                    continue
                indices = selection["fixed_indices"] if method == "fixed_image_review" else [
                    row["index"] for row in selection["selected"]]
                arm = "numeric" if method == "adaptive_numeric_feedback" else "image"
                for index in indices:
                    receipts.append(read_json(out / "requests" /
                        f"{cid}-window-{index:02d}-{arm}-r{rep}.response.json"))
            costs[f"{method}_r{rep}"] = {
                "requests": len(receipts),
                "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
                "seconds": sum(row["seconds"] for row in receipts),
                "shared_requests": "Flows share the initial and matching pure-image reviews"}
    save_json(out / "costs.json", costs)
    for cid in cases:
        plot_case(cases[cid], labels[cid], {name: records[cid] for name, records in methods.items()},
                  out / f"{cid}-activity.png")
        plot_stages(cases[cid], labels[cid], {name: records[cid] for name, records in methods.items()},
                    out / f"{cid}-stages.png")
    lines = ["# 有限自适应复核实际结果", "", config["scope"], "",
             "所有视觉活动配置使用同一个保存的 XGBoost 阶段头；不能称纯 VLM 阶段成绩。", "",
             "| 方法 | P | R | F1 | 事件F1 | 阶段macroF1 | FP | FN | 弱检出日 | 调用 | token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        row = result["aggregate"]
        cost = costs.get(name, {})
        daily = row["daily"]
        lines.append(f"| {name} | {daily['precision']} | {daily['recall']} | {daily['f1']} | "
                     f"{row['events']['f1']} | {row['stage_macro_f1']} | {daily['fp']} | "
                     f"{daily['fn']} | {row['weak_detected_days']} | {cost.get('requests', '—')} | "
                     f"{cost.get('tokens', '—')} |")
        print(name, daily, "stage", row["stage_macro_f1"], flush=True)
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    state = read_json(out / "state.json")
    state["status"] = "complete"
    save_json(out / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.config is None:
            parser.error("prepare requires --config")
        prepare(args.config, args.out)
    elif args.action == "run":
        execute(args.out)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
