"""Create the unchanged direct-visual activity entry for new stage-confirmation records."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity


def prepare(data, out):
    registered = read_json(data / "protocol.json")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "scope": "Frozen direct raw VLM activity on twelve new analytic records for phase confirmation",
        "source_bundle": str(data.resolve()), "case_ids": registered["case_ids"],
        "model": "qwen3.8-flash", "repetitions": 2, "requests_maximum": 24, "retries": 0,
        "activity_policy": "Identical full-record raw image, ACTIVITY_TASK, output schema and settings",
        "phase_policy": "Compare native VLM and separately frozen bounded phase repair; activity never revised",
        "data_manifest_sha256": sha(data / "manifest.json"),
    })
    for name in ("numeric-predictions.json", "stage-xgboost.json", "reference-labels.csv"):
        shutil.copyfile(data / name, out / name)
    shutil.copyfile(data / "reference-labels.csv", out / "reference-subset.csv")
    shutil.copytree(data / "inputs", out / "inputs")
    (out / "images").mkdir()
    for cid in registered["case_ids"]:
        case, _ = load_case_features(out, cid)
        render_raw(case, out / "images" / f"{cid}-raw.png")
    save_json(out / "dependency-freeze.json", {
        str(path.resolve()): sha(path) for path in (
            Path(__file__), ROOT / "src/gnss_sim/landslide_visual.py",
            ROOT / "src/gnss_sim/landslide_local_experiment.py",
            ROOT / "scripts/run_landslide_visual_comparison.py",
        )})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})


def run(out):
    config = read_json(out / "protocol.json")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("A fresh prepared run is required")
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen activity inputs changed")
    for name, expected in read_json(out / "dependency-freeze.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen activity source changed")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               config["requests_maximum"])
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    predictions, costs, failures = {}, {}, {}
    save_json(out / "state.json", {"status": "running"})
    for rep in range(2):
        method = f"direct_visual_r{rep}"
        costs[method] = {"requests": 0, "tokens": 0, "seconds": 0}
        for cid in config["case_ids"]:
            case, features = load_case_features(out, cid)
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200].")
            request_id = f"{cid}-direct-r{rep}"
            try:
                payload = transport.ask(request_id, prompt, [out / "images" / f"{cid}-raw.png"])
                gate = parse_activity(payload, features.observed)
            except (ValueError, RuntimeError) as error:
                failures[request_id] = str(error)
                gate = np.full(len(case.dates), -1)
            stage_codes = model.predict(features.matrix).astype(int) + 1
            prediction = learned_stages(stage_codes, gate, features.supported)
            predictions.setdefault(method, {})[cid] = prediction.to_dict()
            receipt = read_json(out / "requests" / f"{request_id}.response.json")
            costs[method]["requests"] += 1
            costs[method]["tokens"] += receipt.get("usage", {}).get("total_tokens", 0)
            costs[method]["seconds"] += receipt["seconds"]
            save_json(out / "predictions.json", predictions)
            save_json(out / "failures.json", failures)
            save_json(out / "costs.json", costs)
            print("Activity", cid, "repeat", rep + 1, receipt["status"], flush=True)
            if receipt.get("http_status") in (400, 401, 403, 404):
                save_json(out / "state.json", {"status": "failed_provider_contract"})
                raise RuntimeError("Stop after provider contract rejection")
    save_json(out / "prediction-freeze.json", {"sha256": sha(out / "predictions.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
