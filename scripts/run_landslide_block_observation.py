"""Uniformly compare raw and block views on every frozen84 review target."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_observation_probe import display_prompt
from run_landslide_observation_probe import verify as verify_probe
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

SOURCE_FILES = [
    "scripts/run_landslide_block_observation.py", "scripts/run_landslide_observation_probe.py",
    "scripts/analyze_landslide_observation_probe.py",
    "scripts/probe_landslide_longer_motion_evidence.py", "scripts/probe_landslide_revision_scope.py",
    "src/gnss_sim/landslide_observation_view.py", "src/gnss_sim/landslide_visual.py",
    "src/gnss_sim/landslide_window_experiment.py", "src/gnss_sim/landslide_adaptive_review.py",
    "src/gnss_sim/landslide_local_experiment.py",
]


def prepare(out):
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    source = base / "observation-probe-v2"
    verify_probe(source)
    if read_json(source / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete the bounded display screen first")
    for file, expected in read_json(source / "analysis-v3/manifest.json")["files"].items():
        if sha(source / "analysis-v3" / file) != expected:
            raise ValueError("Display screen scoring changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    for file in SOURCE_FILES:
        destination = out / "source" / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, destination)
    config = read_json(source / "protocol.json")
    for cid in config["all_case_ids"]:
        shutil.copytree(source / "inputs" / cid, out / "inputs" / cid)
    for file in ("reference-subset.csv", "stage-xgboost.json", "selections.json",
                 "prior-comparison-predictions.json", "prior-comparison-summary.csv"):
        shutil.copyfile(source / file, out / file)
    longer = Path(config["source_run"])
    shutil.copyfile(longer / "original-predictions.json", out / "initial-predictions.json")
    selections = read_json(source / "selections.json")
    cached_plan = {(row["case_id"], row["start"], row["stop"], row["arm"], row["repeat"]): row
                   for row in read_json(source / "query-plan.json") if row["arm"] in ("narrow_raw", "narrow_blocks")}
    plan, displays = [], {}
    order = np.random.default_rng(20261010)
    for cid in config["all_case_ids"]:
        case, features = load_case_features(out, cid)
        shutil.copyfile(longer / "images" / f"{cid}-raw.png", out / "images" / f"{cid}-raw.png")
        for rep in range(2):
            for target in selections[f"{cid}-r{rep}"]["selected"]:
                for arm in order.permutation(["narrow_raw", "narrow_blocks"]):
                    key = (cid, target["start"], target["stop"], str(arm), rep)
                    prompt = display_prompt(target["start"], target["stop"])
                    local_name = f"{cid}-window-{target['index']:02d}-{arm}.png"
                    local = out / "images" / local_name
                    if not local.exists():
                        displays[local_name] = render_observation_view(case, features.noise,
                            target["start"], target["stop"], 0, arm == "narrow_blocks", local)
                    row = {"request_id": f"{cid}-window-{target['index']:02d}-{arm}-r{rep}",
                           "case_id": cid, "repeat": rep, "arm": str(arm), **target,
                           "prompt": prompt, "images": [f"{cid}-raw.png", local_name]}
                    if key in cached_plan:
                        cached = cached_plan[key]
                        receipt_path = source / "requests" / f"{cached['request_id']}.response.json"
                        receipt = read_json(receipt_path)
                        if receipt["prompt"] != prompt or receipt["status"] != "returned_json":
                            raise ValueError("Cached screen answer is not an identical successful request")
                        for old_name, new_name in zip(cached["images"], row["images"]):
                            if sha(source / "images" / old_name) != sha(out / "images" / new_name):
                                raise ValueError("Cached display pixels differ from the uniform view")
                        row["cached_receipt"] = str(receipt_path)
                        row["cached_sha256"] = sha(receipt_path)
                    plan.append(row)
    new_count = sum("cached_receipt" not in row for row in plan)
    save_json(out / "query-plan.json", plan)
    save_json(out / "displays.json", displays)
    save_json(out / "protocol.json", {"scope": "Post-screen uniform development comparison on viewed case0073-0084",
        "all_case_ids": config["all_case_ids"], "source_screen": str(source), "source_longer": str(longer),
        "primary": "narrow_blocks", "arms": ["narrow_raw", "narrow_blocks"], "repetitions": 2,
        "model": config["model"], "maximum_new_requests": new_count,
        "total_local_receipts": len(plan), "cached_identical_receipts": len(plan) - new_count,
        "fixed": "Every frozen84 selected window receives the same narrow display policy; unchanged "
                 "initials, short/42/84 conflicts, budget2, physical49/91 revision scope, stage head",
        "primary_stop": "Both repetitions must improve whole-record F1 and weak recall versus uniform "
                        "same-prompt narrow_raw, have F1 no lower than original base42 scoped method, "
                        "not increase summed FP on step negatives0074/0080, and not lose case0082 TP. "
                        "If it fails, no new confirmation; no threshold or repetition rescue",
        "independence": "Selected after five-target screen. These twelve records remain viewed development, "
                        "not new independent confirmation; wide_blocks primary screen remains rejected",
        "request_policy": "temperature.1/top_p.3/max_tokens2048/no_thinking/no_retries; random arm order seed20261010"})
    save_json(out / "source-bindings.json", {str(source / name): sha(source / name) for name in (
        "freeze.json", "prediction-freeze.json", "analysis-v3/manifest.json")})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print(f"Uniform narrow-display probe frozen: {len(plan)} targets/arms, "
          f"{len(plan) - new_count} identical cached, {new_count} new requests", flush=True)


def verify(out):
    for file, expected in read_json(out / "freeze.json").items():
        if sha(out / file) != expected:
            raise ValueError(f"Uniform display freeze changed: {file}")
    for file, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(file)) != expected:
            raise ValueError("Uniform display's bound screen changed")
    for file in SOURCE_FILES:
        if sha(ROOT / file) != sha(out / "source" / file):
            raise ValueError("Executing uniform display source changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only a fresh prepared experiment, no repeats")
    config = read_json(out / "protocol.json")
    save_json(out / "state.json", {"status": "running"})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_new_requests"])
    reviews, failures = {}, {}
    for index, row in enumerate(read_json(out / "query-plan.json")):
        _, features = load_case_features(out, row["case_id"])
        status = "success"
        values = np.full(len(features.observed), -1, dtype=int)
        try:
            if "cached_receipt" in row:
                path = Path(row["cached_receipt"])
                if sha(path) != row["cached_sha256"]:
                    raise ValueError("Cached actual receipt changed")
                payload = json.loads(read_json(path)["output"])
            else:
                payload = transport.ask(row["request_id"], row["prompt"], [out / "images" / f for f in row["images"]])
            values = parse_window_activity(payload, features.observed, row["start"], row["stop"])
        except (ValueError, RuntimeError) as error:
            failures[row["request_id"]] = str(error)
            status = "failed"
        method = f"uniform_{row['arm']}_r{row['repeat']}"
        reviews.setdefault(method, {}).setdefault(row["case_id"], []).append({
            "start": row["start"], "stop": row["stop"], "activity": values.tolist(), "status": status,
            "request_id": row["request_id"], "cached": "cached_receipt" in row})
        save_json(out / "reviews.json", reviews)
        save_json(out / "failures.json", failures)
        print(f"Uniform receipt {index + 1}/{config['total_local_receipts']}: {row['request_id']} {status}", flush=True)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("reviews.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
