"""Compare direct VLM and bounded Agent activity with the same native VLM stage head."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, LandslideDiagnostics, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_diagnostics import render_diagnostics
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_visual import (
    ACTIVITY_TASK,
    EVOLUTION_TASK,
    VisualRequests,
    parse_stages,
)

HEADS = ("direct_visual", "adaptive_numeric_feedback")


def prepare(activity_run, out, heads):
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(activity_run / "protocol.json")
    save_json(out / "protocol.json", {
        "date": "2026-10-09", "scope": config["scope"],
        "purpose": "Native VLM stage comparison supplement, registered before test scoring",
        "activity_run": str(activity_run.resolve()), "activity_protocol_sha256": sha(
            activity_run / "protocol.json"),
        "heads": heads, "repetitions": config["repetitions"],
        "case_ids": config["case_ids"], "model": config["model"],
        "maximum_requests": len(config["case_ids"]) * config["repetitions"] * len(heads),
        "retries": 0, "hard_timeout_seconds": 1800,
        "stage_evidence": "Same full-record raw image and 61-day auxiliary image",
        "stage_support": "Same feature-supported mask used for XGBoost stage heads",
        "task_prefix": ACTIVITY_TASK, "evolution_task": EVOLUTION_TASK,
        "contract": "Same strict inclusive stage parser; failed stages unknown, activity retained",
        "selection": "No choice of repeats, no method changes after these test scores"})
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "state.json", {"status": "prepared"})


def run(out):
    config = read_json(out / "protocol.json")
    source = Path(config["activity_run"])
    if read_json(source / "state.json")["status"] not in {"predictions_complete", "complete"}:
        raise ValueError("Wait for every activity prediction to be frozen")
    if sha(Path(__file__)) != sha(out / "source.py"):
        raise ValueError("Native stage runner differs from registered source")
    activity_predictions = read_json(source / "predictions.json")
    save_json(out / "activity-prediction-binding.json", {"sha256": sha(source / "predictions.json")})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               config["maximum_requests"])
    (out / "images").mkdir()
    predictions, failures, skipped, costs = {}, {}, [], {}
    started = time.monotonic()
    for rep in range(config["repetitions"]):
        for ordinal, cid in enumerate(config["case_ids"]):
            if time.monotonic() - started > config["hard_timeout_seconds"]:
                raise RuntimeError("Native stage budget exceeded")
            case, features = load_case_features(source, cid)
            raw = out / "images" / f"{cid}-raw.png"
            auxiliary = out / "images" / f"{cid}-auxiliary.png"
            if not raw.exists():
                shutil.copyfile(source / "images" / raw.name, raw)
                motion = LandslideDiagnostics.model_validate_json(
                    (source / "inputs" / cid / "motion-61.json").read_text(encoding="utf-8"))
                auxiliary.write_bytes(render_diagnostics(case, motion))
            registered_heads = tuple(config["heads"])
            heads = registered_heads if (ordinal + rep) % 2 == 0 else registered_heads[::-1]
            for head in heads:
                source_method = f"{head}_r{rep}"
                method = f"{head}_native_stage_r{rep}"
                gate = np.array(activity_predictions[source_method][cid]["activity"])
                request_id = f"{cid}-{head}-stage-r{rep}"
                if not np.any(gate == 1):
                    prediction = ActivityPrediction(gate, np.where(gate == 0, 0, -1))
                    skipped.append(request_id)
                else:
                    spans = [[left, right - 1] for left, right in runs(gate == 1)]
                    prompt = (ACTIVITY_TASK.format(last=len(gate) - 1) + EVOLUTION_TASK
                              + "Only label stages inside the supplied confirmed activity "
                              "intervals. Do not extend, revise or invent activity.\n"
                              "Confirmed inclusive activity intervals: " + json.dumps(spans)
                              + '\nReturn exactly {"stages":[[start,end,"A"],[start,end,"S"],'
                              '[start,end,"D"]]} with actual intervals; empty lists are valid. '
                              "No other keys.")
                    try:
                        payload = transport.ask(request_id, prompt, [raw, auxiliary])
                        prediction = parse_stages(payload, gate, features.observed)
                        prediction.stage[(gate == 1) & ~features.supported] = -1
                    except (ValueError, RuntimeError) as error:
                        failures[request_id] = str(error)
                        prediction = ActivityPrediction(gate, np.full(len(gate), -1),
                                                        stage_status="failed")
                predictions.setdefault(method, {})[cid] = prediction.to_dict()
                costs.setdefault(method, []).append(request_id)
            save_json(out / "predictions.json", predictions)
            save_json(out / "failures.json", failures)
            print(f"Native stages {cid} repeat {rep + 1}", flush=True)
    save_json(out / "skipped.json", skipped)
    save_json(out / "cost-bindings.json", costs)
    save_json(out / "prediction-freeze.json", {"sha256": sha(out / "predictions.json")})
    save_json(out / "state.json", {"status": "predictions_complete",
                                   "seconds": time.monotonic() - started})


def score(out):
    config = read_json(out / "protocol.json")
    source = Path(config["activity_run"])
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Native stage predictions must be frozen before scoring")
    if sha(out / "predictions.json") != read_json(out / "prediction-freeze.json")["sha256"]:
        raise ValueError("Native stage predictions changed")
    cases = {cid: load_case_features(source, cid)[0] for cid in config["case_ids"]}
    with (source / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["case_id"] in cases]
        columns = reader.fieldnames
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    labels = load_labels(out / "reference-subset.csv", cases)
    results, costs = {}, {}
    activity_costs = read_json(source / "costs.json")
    for method, cases in read_json(out / "predictions.json").items():
        per_case = {cid: evaluate_case(labels[cid], ActivityPrediction(
            np.array(value["activity"]), np.array(value["stage"]),
            stage_status=value["stage_status"])) for cid, value in cases.items()}
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())),
                           "per_case": per_case}
        source_method = method.replace("_native_stage", "")
        costs[method] = dict(activity_costs[source_method])
        receipts = [read_json(out / "requests" / f"{request_id}.response.json")
                    for request_id in read_json(out / "cost-bindings.json")[method]
                    if (out / "requests" / f"{request_id}.response.json").exists()]
        costs[method]["stage_requests"] = len(receipts)
        costs[method]["requests"] += len(receipts)
        costs[method]["tokens"] += sum(row.get("usage", {}).get("total_tokens", 0)
                                        for row in receipts)
        costs[method]["seconds"] += sum(row["seconds"] for row in receipts)
        print(method, "activity F1", results[method]["aggregate"]["daily"]["f1"],
              "native stage F1", results[method]["aggregate"]["stage_macro_f1"], flush=True)
    save_json(out / "results.json", results)
    save_json(out / "costs.json", costs)
    state = read_json(out / "state.json")
    state["status"] = "complete"
    save_json(out / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--activity-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", default=list(HEADS),
                        choices=("direct_visual", "adaptive_no_feedback", "adaptive_numeric_feedback"))
    args = parser.parse_args()
    if args.action == "prepare":
        if args.activity_run is None:
            parser.error("prepare requires --activity-run")
        prepare(args.activity_run, args.out, args.heads)
    elif args.action == "run":
        run(args.out)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
