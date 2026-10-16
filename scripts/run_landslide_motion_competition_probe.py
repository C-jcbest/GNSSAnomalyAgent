"""Run a bounded rival-motion observation experiment on eight viewed targets."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population, verify_target_metrics
from finalize_landslide_adaptive_evidence import verify_request_receipts
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_observation_probe import display_prompt
from run_landslide_window_optimization import load_case_features, read_json
from verify_landslide_evidence_display_confirmation import verify_live_sources
from xgboost import XGBClassifier

from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_motion_competition import motion_competition, render_competition
from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
TARGETS = [
    {"case_id": "case_0078", "start": 450, "stop": 630, "source": "block-observation-development-v2", "role": "previous block-suppressed weak tail"},
    {"case_id": "case_0080", "start": 360, "stop": 540, "source": "block-observation-development-v2", "role": "instrument-step negative"},
    {"case_id": "case_0081", "start": 180, "stop": 360, "source": "block-observation-development-v2", "role": "block-recovered weak-constant preservation"},
    {"case_id": "case_0083", "start": 270, "stop": 450, "source": "block-observation-development-v2", "role": "post-motion plateau"},
    {"case_id": "case_0084", "start": 450, "stop": 630, "source": "block-observation-development-v2", "role": "unrecovered weak tail"},
    {"case_id": "case_0090", "start": 450, "stop": 630, "source": "evidence-display-confirmation-v2", "role": "weak tail and daily-noise interpretation conflict"},
    {"case_id": "case_0091", "start": 540, "stop": 720, "source": "evidence-display-confirmation-v2", "role": "quiet negative with previous visual FP"},
    {"case_id": "case_0096", "start": 450, "stop": 630, "source": "evidence-display-confirmation-v2", "role": "partially recovered weak tail"},
]
ARMS = ("raw", "rival_curves", "rival_feedback")
NOTICE = (
    "\nAny additional rival-description image compares a stationary level, a single abrupt step, "
    "and a continuous ramp fitted to complete 14-day raw median blocks (at least8 observed days). "
    "These are models of the SAME observations, not independent evidence or activity labels. "
    "Dashed nodes are descriptive, never activity boundaries. If an error table is present, "
    "it gives all-block and alternating-block prediction errors divided by the raw daily-difference "
    "MAD noise proxy. Hiding blocks can make a step location ambiguous: held-out error alone "
    "cannot identify a physical cause or rule out a step. Blocks remain correlated, and errors "
    "are not probabilities, uncertainties or significance tests. Judge sustained cumulative "
    "change from the raw observations, distinguish isolated steps and stationary plateaus, "
    "and keep uncertain where observations cannot distinguish them. Do not copy model nodes "
    "as output interval endpoints or report model-fit numbers as activity decisions.\n"
)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    for folder in sorted({row["source"] for row in TARGETS}):
        source = BASE / folder
        freeze_file = "pre-generation-freeze.json" if folder.endswith("confirmation-v2") else "freeze.json"
        for name, expected in read_json(source / freeze_file).items():
            if sha(source / name) != expected:
                raise ValueError("Bound viewed experiment changed")
        if folder.endswith("confirmation-v2"):
            verify_live_sources(source)
    models = [BASE / folder / "stage-xgboost.json" for folder in {row["source"] for row in TARGETS}]
    if len({sha(path) for path in models}) != 1:
        raise ValueError("Target groups must use the same frozen learned stage head")
    shutil.copyfile(models[0], out / "stage-xgboost.json")
    source_files = [Path(__file__), ROOT / "scripts/analyze_landslide_observation_probe.py",
                    ROOT / "scripts/finalize_landslide_adaptive_evidence.py",
                    ROOT / "scripts/run_landslide_observation_probe.py",
                    ROOT / "scripts/run_landslide_comparison.py",
                    ROOT / "scripts/run_landslide_window_optimization.py"]
    source_files += list((ROOT / "src/gnss_sim").glob("landslide*.py"))
    source_files += [ROOT / "src/gnss_sim/visual.py"]
    for source in source_files:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config = {"scope": "Eight developer-selected previously viewed targets; hypothesis development, not independent confirmation",
        "targets": TARGETS, "arms": ARMS, "primary": "rival_feedback", "repetitions": 2,
        "maximum_requests": 48, "model": "qwen3.8-flash", "retries": 0,
        "mechanism": "Stationary/step/finite ramp competitors from14-day raw blocks; descriptive model errors only, no assigned activity",
        "ablations": "raw removes the numerical tool; rival_curves retains fits but removes the explicit error table; identical neutral prompt",
        "stage": "Same frozen learned stage head. Target population may lack A/D; report denominators and defined/fixed macros",
        "stop": "Both repetitions must improve target F1 and weak recall vs same-prompt raw, not increase total FP on any known negatives, "
                "and not lose TP on any target. Otherwise stop this mechanism without prompt/threshold rescue",
        "full_method": "Passing this selected-target screen would only justify a complete-window development prototype, not a final method claim",
        "model_settings": "temperature.1/top_p.3/max_tokens2048/no_thinking/no_retries; seed20261012 arm permutation",
        "numeric_limit": "Alternating-block step locations can be ambiguous even in an ideal step; error ranking is not cause identification"}
    save_json(out / "protocol.json", config)
    diagnostics, numerical_rows, references, plan = {}, [], [], []
    order = np.random.default_rng(20261012)
    for target in TARGETS:
        cid, source = target["case_id"], BASE / target["source"]
        shutil.copytree(source / "inputs" / cid, out / "inputs" / cid)
        case, features = load_case_features(out, cid)
        shutil.copyfile(source / "images" / f"{cid}-raw.png", out / "images" / f"{cid}-raw.png")
        local = f"{cid}-target-raw.png"
        render_observation_view(case, features.noise, target["start"], target["stop"], 0, False, out / "images" / local)
        diagnostic = motion_competition(case, features.noise, target["start"], target["stop"])
        diagnostics[cid] = diagnostic
        if diagnostic["status"] != "available":
            raise ValueError("The diagnostic screen contains an unsupported competitor target")
        for name, fitted in diagnostic["models"].items():
            numerical_rows.append({"case_id": cid, "role": target["role"], "model": name,
                "blocks": len(diagnostic["blocks"]), "full_rmse": float(np.sqrt(fitted["training_mse_noise_units"])),
                "held_out_rmse": fitted["alternating_block_rmse_noise_units"],
                "fitted_start": fitted["start"], "fitted_stop": fitted["stop"]})
        for arm in ("rival_curves", "rival_feedback"):
            render_competition(case, diagnostic, arm == "rival_feedback", out / "images" / f"{cid}-{arm}.png")
        # Keep the exact observation calendar; only these selected targets score.
        with (source / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["case_id"] != cid:
                    continue
                if not target["start"] <= int(row["day_index"]) < target["stop"]:
                    row = {**row, "activity_label": "-1", "feature": "", "episode_id": "", "weak_signal": "0"}
                references.append(row)
        for rep in range(2):
            for arm in order.permutation(ARMS):
                images = [f"{cid}-raw.png", local]
                if arm != "raw":
                    images.append(f"{cid}-{arm}.png")
                plan.append({"case_id": cid, "repeat": rep, "arm": str(arm),
                    "start": target["start"], "stop": target["stop"],
                    "request_id": f"{cid}-{arm}-r{rep}", "images": images,
                    "prompt": display_prompt(target["start"], target["stop"]) + NOTICE})
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    csv_rows(out / "numerical-screen.csv", numerical_rows)
    save_json(out / "diagnostics.json", diagnostics)
    save_json(out / "query-plan.json", plan)
    save_json(out / "source-bindings.json", {str(BASE / folder / "prediction-freeze.json"): sha(BASE / folder / "prediction-freeze.json")
                                             for folder in {row["source"] for row in TARGETS}})
    save_json(out / "freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared_not_inferred"})
    print("Eight viewed targets: numerical screen and 48-request paired probe frozen", flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen probe changed: {name}")
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError("Executing probe dependency changed")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Previous viewed prediction changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared_not_inferred":
        raise ValueError("Execute a fresh probe once; no retry or overwrite")
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
        print(f"Competition probe {number + 1}/48 {row['request_id']} {status}", flush=True)
    verify(out)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in (
        "predictions.json", "outputs.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})


def analyze(out, analysis):
    verify(out)
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only after all48 predictions freeze")
    for name, expected in read_json(out / "prediction-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Prediction changed after inference")
    records = {target["case_id"]: load_case_features(out, target["case_id"]) for target in TARGETS}
    labels = load_labels(out / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    predictions = read_json(out / "predictions.json")
    receipts = verify_request_receipts(out)
    if len(receipts) != 48:
        raise ValueError("Probe omitted an arm or repetition")
    for request_id, row in read_json(out / "outputs.json").items():
        if row["status"] == "success":
            payload = json.loads(read_json(out / "requests" / f"{request_id}.response.json")["output"])
            activity = parse_window_activity(payload, records[row["case_id"]][1].observed, row["start"], row["stop"])
            if row["activity"] != activity.tolist():
                raise ValueError("Local output differs from actual receipt")
    results, summary, per_record = evaluate_population(predictions, labels)
    internal = verify_target_metrics(predictions, results, labels, records)
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summary}
    decisions, changes = [], []
    for rep in range(2):
        candidate, control = f"rival_feedback_r{rep}", f"raw_r{rep}"
        a, b = indexed[candidate], indexed[control]
        losses = []
        for cid in records:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            new, old = np.array(predictions[candidate][cid]["activity"]), np.array(predictions[control][cid]["activity"])
            removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
            lost = int(np.count_nonzero(removed & (actual == 1)))
            losses.append(lost)
            changes.append({"repeat": rep + 1, "case_id": cid, "lost_tp": lost,
                "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                "gained_fp": int(np.count_nonzero(added & (actual == 0)))})
        decision = {"repeat": rep + 1, "f1_difference": a["f1"] - b["f1"],
                    "weak_recall_difference": a["weak_recall"] - b["weak_recall"],
                    "fp_difference": a["fp"] - b["fp"], "maximum_target_lost_tp": max(losses)}
        decision["passes"] = (decision["f1_difference"] > 0 and decision["weak_recall_difference"] > 0
                               and decision["fp_difference"] <= 0 and decision["maximum_target_lost_tp"] == 0)
        decisions.append(decision)
    analysis.mkdir(parents=True, exist_ok=False)
    save_json(analysis / "results.json", results)
    save_json(analysis / "stage-denominators-and-macros.json", internal)
    csv_rows(analysis / "summary.csv", summary)
    csv_rows(analysis / "per-target.csv", per_record)
    csv_rows(analysis / "changes.csv", changes)
    save_json(analysis / "candidate-decision.json", {"repetitions": decisions,
        "proceed_to_complete_window_development": all(row["passes"] for row in decisions),
        "independence": "Viewed selected-target development; even passing would not confirm a full method"})
    costs = {"new_unique_requests": len(receipts),
        "new_unique_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
        "per_method": {method: {"requests": 8, "tokens": sum(row.get("usage", {}).get("total_tokens", 0)
            for row in receipts if row["request_id"].split("-", 1)[1] == method.replace("_r", "-r"))}
            for method in predictions}}
    save_json(analysis / "costs.json", costs)
    save_json(analysis / "verification.json", {"actual_payloads_bound": 48,
        "independent_target_metric_groups": 6, "failed_contracts": len(read_json(out / "failures.json")),
        "no_new_fitting_or_tuning": True, "target_population_only": True})
    save_json(analysis / "manifest.json", {"run_freeze_sha256": sha(out / "freeze.json"),
        "prediction_freeze_sha256": sha(out / "prediction-freeze.json"),
        "files": {path.relative_to(analysis).as_posix(): sha(path) for path in analysis.rglob("*") if path.is_file()}})
    print("Competition development decision:", decisions, flush=True)
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
