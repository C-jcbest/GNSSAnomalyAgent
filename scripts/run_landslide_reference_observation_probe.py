"""Test a frozen training reference as additional observation, never a trigger."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population, verify_target_metrics
from finalize_landslide_adaptive_evidence import verify_request_receipts
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_seasonal_reference import apply_annual_reference, observed_values
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import WINDOW_TASK, parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "background-review-v1"
REFERENCE = BASE / "annual-reference-evidence-v2"
ARMS = ("raw", "reference_curve", "reference_residual")
TARGETS = [
    {"case_id": f"case_{number}", "start": 450 if family == "weak_after_strong" else 180,
     "stop": 630 if family == "weak_after_strong" else 360, "family": family}
    for number, family in ((2025, "quiet"), (2026, "instrument_step"), (2027, "weak_constant"),
                           (2030, "weak_after_strong"), (2031, "quiet"), (2032, "instrument_step"),
                           (2033, "weak_constant"), (2036, "weak_after_strong"))
]
NOTICE = (
    "\nImages are a global raw view, a local raw view, and a two-column supplementary view. "
    "The supplement repeats raw observations or overlays a frozen annual training reference; "
    "its right column repeats raw observations or shows observed minus reference. "
    "The reference was fitted only on four separate training records assumed to contain no "
    "target injection. This is offline grouped training: some training dates are later than "
    "these observations. It is not a strictly past-only forecast, independent measurement, "
    "physical causal model, activity label or decision threshold. There is no fitting of "
    "current displacement. Seasonal variation may itself be real motion, and a reference "
    "can mismatch. Subtraction preserves an added signal algebraically but does not guarantee "
    "its detectability. Always judge sustained activity from the preserved raw observations; "
    "use the supplement only to examine competing explanations, and retain uncertainty "
    "where evidence is insufficient. Do not copy reference extrema as activity boundaries.\n"
)


def render_supplement(case, model, target, arm, path):
    corrected, predicted = apply_annual_reference(case, model)
    raw, _ = observed_values(case)
    residual, _ = observed_values(corrected)
    left, right = target["start"], target["stop"]
    days = np.arange(left, right)
    fig, panels = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    for component, name in enumerate(("N", "E", "U")):
        raw_series = raw[left:right, component]
        residual_series = residual[left:right, component]
        reference_series = predicted[left:right, component]
        # All arms have identical limits, dimensions and preserved raw curves.
        combined = np.concatenate((raw_series, residual_series, reference_series))
        finite = combined[np.isfinite(combined)]
        lower, upper = float(finite.min()), float(finite.max())
        margin = max((upper - lower) * .15, 2)
        panels[component, 0].plot(days, raw_series, color="#24887b", linewidth=.8, label="Raw observations")
        if arm != "raw":
            panels[component, 0].plot(days, reference_series, color="#c78635", linewidth=1.2,
                                      label="Frozen training reference")
        second = residual_series if arm == "reference_residual" else raw_series
        panels[component, 1].plot(days, second, color="#24887b", linewidth=.8)
        panels[component, 0].set_ylabel(name + " (mm)")
        for panel in panels[component]:
            panel.set_ylim(lower - margin, upper + margin)
            panel.set_xlim(left, right - 1)
            panel.grid(alpha=.2)
    panels[0, 0].legend(fontsize=8)
    panels[0, 1].set_title("Observed minus reference" if arm == "reference_residual" else "Repeated raw observations")
    for panel in panels[-1]:
        panel.set_xlabel("Original daily calendar index")
    fig.suptitle(case.case_id + " | Supplementary observation; no activity labels")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def prepare(out):
    for name, expected in read_json(REFERENCE / "manifest.json")["files"].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError("Frozen reference evidence changed")
    if read_json(REFERENCE / "decision.json")["allows_visual_probe"]:
        raise ValueError("This separately registered observation branch expects the failed trigger branch")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    for name in ("annual-reference.json",):
        shutil.copyfile(REFERENCE / name, out / name)
    shutil.copyfile(DATA / "stage-xgboost.json", out / "stage-xgboost.json")
    sources = list((ROOT / "src/gnss_sim").glob("*.py"))
    sources += list((ROOT / "scripts").glob("*.py"))
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    protocol = {
        "scope": "Eight previously viewed developer-selected targets on two shared backgrounds; development only",
        "targets": TARGETS, "arms": ARMS, "primary": "reference_residual", "repetitions": 2,
        "maximum_requests": 48, "model": "qwen3.8-flash", "retries": 0,
        "mechanism": "Additional frozen training reference observation; original raw images always retained",
        "separate_registration": "Residual-trigger evidence branch failed and remains stopped; no hard residual screening here",
        "fitting": "Reuse four train-only annual coefficients; no current displacement fitting, thresholds or prompt search",
        "chronology": "Offline group split, not strictly past-only training or an online forecasting experiment",
        "controls": "Three images each, same prompt, same raw global/local images, matched supplement dimensions and axis limits",
        "seed": 20261014, "stage": "Same background-benchmark learned stage head; absent reference classes disclosed",
        "pass_rule": "In BOTH repetitions: primary target F1 strictly improves over raw, weak recall does not decrease, "
                     "no target loses TP, and FP does not increase on any quiet or step target. Any contract failure stops. "
                     "Passing justifies full-window development only; failure stops without rescue tuning.",
        "fairness_next": "A full-method comparison must provide the same reference information to numeric and learned controls",
    }
    save_json(out / "protocol.json", protocol)
    model = read_json(out / "annual-reference.json")
    target_map = {row["case_id"]: row for row in TARGETS}
    with (DATA / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        references = []
        for row in csv.DictReader(handle):
            if row["case_id"] not in target_map:
                continue
            target = target_map[row["case_id"]]
            if not target["start"] <= int(row["day_index"]) < target["stop"]:
                row = {**row, "activity_label": "-1", "feature": "", "episode_id": "", "weak_signal": "0"}
            references.append(row)
    csv_rows(out / "reference-subset.csv", references)
    plan, bindings = [], {}
    order = np.random.default_rng(protocol["seed"])
    for target in TARGETS:
        cid = target["case_id"]
        shutil.copytree(DATA / "inputs" / cid, out / "inputs" / cid)
        case, features = load_case_features(out, cid)
        raw_name = f"{cid}-raw.png"
        local_name = f"{cid}-target-raw.png"
        shutil.copyfile(DATA / "images" / raw_name, out / "images" / raw_name)
        render_observation_view(case, features.noise, target["start"], target["stop"], 0, False,
                                out / "images" / local_name)
        for arm in ARMS:
            render_supplement(case, model, target, arm, out / "images" / f"{cid}-{arm}.png")
        for rep in range(2):
            for arm in order.permutation(ARMS):
                plan.append({"case_id": cid, "repeat": rep, "arm": str(arm),
                             "start": target["start"], "stop": target["stop"],
                             "request_id": f"{cid}-{arm}-r{rep}",
                             "images": [raw_name, local_name, f"{cid}-{arm}.png"],
                             "prompt": WINDOW_TASK.format(start=target["start"], end=target["stop"] - 1) + NOTICE})
        for name in (f"inputs/{cid}/input.json", f"images/{raw_name}"):
            bindings[str(DATA / name)] = sha(DATA / name)
    for source in (DATA / "stage-xgboost.json", DATA / "prediction-freeze.json", REFERENCE / "manifest.json"):
        bindings[str(source)] = sha(source)
    save_json(out / "source-bindings.json", bindings)
    save_json(out / "query-plan.json", plan)
    save_json(out / "freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                    for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared_not_inferred"})
    print("Prepared separate observation branch: 8 targets, 3 arms, 2 repetitions, 48 requests", flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen observation probe changed: {name}")
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError(f"Live source changed: {name}")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Bound previous experiment changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared_not_inferred":
        raise ValueError("Run a fresh prepared probe once; no retry or overwrite")
    transport = VisualRequests(out / "requests", ROOT / ".env", "qwen3.8-flash", 48)
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    records = {target["case_id"]: load_case_features(out, target["case_id"]) for target in TARGETS}
    predictions, outputs, failures = {}, {}, {}
    save_json(out / "state.json", {"status": "running"})
    for number, row in enumerate(read_json(out / "query-plan.json")):
        _, features = records[row["case_id"]]
        status = "success"
        try:
            payload = transport.ask(row["request_id"], row["prompt"], [out / "images" / name for name in row["images"]])
            activity = parse_window_activity(payload, features.observed, row["start"], row["stop"])
        except (ValueError, RuntimeError) as error:
            failures[row["request_id"]] = str(error)
            status = "failed"
            activity = np.full(len(features.observed), -1, dtype=int)
        codes = head.predict(features.matrix).astype(int) + 1
        prediction = learned_stages(codes, activity, features.supported)
        prediction.status = status
        method = f"{row['arm']}_r{row['repeat']}"
        predictions.setdefault(method, {})[row["case_id"]] = prediction.to_dict()
        outputs[row["request_id"]] = {**row, "status": status, "activity": activity.tolist()}
        save_json(out / "predictions.json", predictions)
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        print(f"Reference observation {number + 1}/48 {row['request_id']} {status}", flush=True)
    verify(out)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("predictions.json", "outputs.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})


def analyze(out, analysis):
    verify(out)
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all predictions before scoring")
    for name, expected in read_json(out / "prediction-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Prediction changed after inference")
    records = {target["case_id"]: load_case_features(out, target["case_id"]) for target in TARGETS}
    labels = load_labels(out / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    predictions = read_json(out / "predictions.json")
    receipts = verify_request_receipts(out)
    plan = read_json(out / "query-plan.json")
    if len(receipts) != 48 or len(predictions) != 6 or any(len(rows) != 8 for rows in predictions.values()):
        raise ValueError("Incomplete paired population")
    for row in plan:
        output = read_json(out / "outputs.json")[row["request_id"]]
        if output["status"] == "success":
            payload = json.loads(read_json(out / "requests" / f"{row['request_id']}.response.json")["output"])
            activity = parse_window_activity(payload, records[row["case_id"]][1].observed, row["start"], row["stop"])
            if activity.tolist() != output["activity"]:
                raise ValueError("Actual receipt differs from scored output")
    results, summary, per_record = evaluate_population(predictions, labels)
    internal = verify_target_metrics(predictions, results, labels, records)
    indexed = {row["method"]: row for row in summary}
    decisions, changes = [], []
    for rep in range(2):
        new_method, old_method = f"reference_residual_r{rep}", f"raw_r{rep}"
        new_score, old_score = indexed[new_method], indexed[old_method]
        rep_changes = []
        for target in TARGETS:
            cid = target["case_id"]
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            new = np.array(predictions[new_method][cid]["activity"])
            old = np.array(predictions[old_method][cid]["activity"])
            removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
            change = {"repeat": rep + 1, "case_id": cid, "family": target["family"],
                      "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                      "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                      "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                      "gained_fp": int(np.count_nonzero(added & (actual == 0)))}
            rep_changes.append(change)
        decision = {"repeat": rep + 1, "f1_difference": new_score["f1"] - old_score["f1"],
                    "weak_recall_difference": new_score["weak_recall"] - old_score["weak_recall"],
                    "maximum_target_lost_tp": max(row["lost_tp"] for row in rep_changes),
                    "maximum_negative_fp_increase": max(row["gained_fp"] - row["removed_fp"]
                        for row in rep_changes if row["family"] in ("quiet", "instrument_step"))}
        decision["passes"] = (decision["f1_difference"] > 0 and decision["weak_recall_difference"] >= 0
                              and decision["maximum_target_lost_tp"] == 0
                              and decision["maximum_negative_fp_increase"] <= 0
                              and not read_json(out / "failures.json"))
        decisions.append(decision)
        changes.extend(rep_changes)
    analysis.mkdir(parents=True, exist_ok=False)
    save_json(analysis / "results.json", results)
    save_json(analysis / "stage-denominators-and-macros.json", internal)
    csv_rows(analysis / "summary.csv", summary)
    csv_rows(analysis / "per-target.csv", per_record)
    csv_rows(analysis / "changes.csv", changes)
    save_json(analysis / "candidate-decision.json", {"repetitions": decisions,
        "proceed_to_complete_window_development": all(row["passes"] for row in decisions),
        "scope": "Viewed selected-target development; distinct from the stopped residual-trigger branch"})
    receipt_map = {row["request_id"]: row for row in receipts}
    costs = {"new_unique_requests": len(receipts),
             "new_unique_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
             "per_method": {}}
    for method in predictions:
        selected = [receipt_map[row["request_id"]] for row in plan if f"{row['arm']}_r{row['repeat']}" == method]
        costs["per_method"][method] = {"requests": len(selected),
            "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in selected),
            "seconds": sum(row["seconds"] for row in selected)}
    save_json(analysis / "costs.json", costs)
    save_json(analysis / "verification.json", {"actual_payloads_bound": 48,
        "independent_metric_groups": 6, "failed_contracts": len(read_json(out / "failures.json")),
        "same_raw_inputs_and_stage_head": True, "target_only_not_full_method": True})
    save_json(analysis / "manifest.json", {"run_freeze_sha256": sha(out / "freeze.json"),
        "prediction_freeze_sha256": sha(out / "prediction-freeze.json"),
        "files": {path.relative_to(analysis).as_posix(): sha(path) for path in analysis.rglob("*") if path.is_file()}})
    print("Reference observation decision:", decisions, flush=True)
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
