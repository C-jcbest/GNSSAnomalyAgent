"""Freeze and run a small context-by-block-display GNSS mechanism experiment."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_longer_motion_review import verify as verify_longer
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import WINDOW_TASK, parse_window_activity

TARGETS = [
    {"case_id": "case_0074", "index": 4, "start": 360, "stop": 540, "role": "viewed step negative"},
    {"case_id": "case_0080", "index": 4, "start": 360, "stop": 540, "role": "viewed step negative"},
    {"case_id": "case_0081", "index": 2, "start": 180, "stop": 360, "role": "viewed weak constant motion"},
    {"case_id": "case_0082", "index": 3, "start": 270, "stop": 450, "role": "viewed TP-loss control"},
    {"case_id": "case_0084", "index": 5, "start": 450, "stop": 630, "role": "viewed weak tail"},
]
ARMS = {
    "narrow_raw": (0, False), "wide_raw": (91, False),
    "narrow_blocks": (0, True), "wide_blocks": (91, True),
}
SOURCE_FILES = [
    "scripts/run_landslide_observation_probe.py",
    "scripts/run_landslide_longer_motion_review.py",
    "scripts/probe_landslide_longer_motion_evidence.py",
    "scripts/probe_landslide_revision_scope.py",
    "src/gnss_sim/landslide_observation_view.py",
    "src/gnss_sim/landslide_visual.py",
    "src/gnss_sim/landslide_window_experiment.py",
    "src/gnss_sim/landslide_local_experiment.py",
    "src/gnss_sim/landslide_adaptive_review.py",
    "tests/test_landslide_observation_view.py",
]


def display_prompt(start, stop):
    notice = ("Any black horizontal bars summarize complete 14-day calendar blocks with at least "
              "8 observed days; they are medians of the same raw observations, not new measurements "
              "or activity labels. Use the raw observations to judge continuity and isolated steps.\n")
    return WINDOW_TASK.format(start=start, end=stop - 1).replace("Judge ONLY", notice + "Judge ONLY", 1)


def prepare(out):
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    source = base / "longer-evidence-review-v1"
    verify_longer(source)
    for name, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / name) != expected:
            raise ValueError("Prior longer-evidence predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    images = out / "images"
    images.mkdir()
    (out / "source").mkdir()
    for file in SOURCE_FILES:
        destination = out / "source" / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, destination)
    config = read_json(source / "protocol.json")
    all_cases = config["case_ids"]
    for cid in all_cases:
        shutil.copytree(source / "inputs" / cid, out / "inputs" / cid)
    for file in ("reference-subset.csv", "stage-xgboost.json", "selections.json"):
        shutil.copyfile(source / file, out / file)
    shutil.copyfile(source / "predictions.json", out / "base84-predictions.json")
    shutil.copyfile(source / "analysis-v3/predictions.json", out / "prior-comparison-predictions.json")
    shutil.copyfile(source / "analysis-v3/summary.csv", out / "prior-comparison-summary.csv")
    targets = {target["case_id"]: target for target in TARGETS}
    with (source / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = [row for row in reader if row["case_id"] in targets]
    with (out / "target-reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            target = targets[row["case_id"]]
            if not target["start"] <= int(row["day_index"]) < target["stop"]:
                row = {**row, "activity_label": "-1", "feature": "unknown", "weak_signal": "0"}
            writer.writerow(row)
    displays, cached, plan = {}, {}, []
    order = np.random.default_rng(20261009)
    for target in TARGETS:
        cid = target["case_id"]
        case, features = load_case_features(out, cid)
        shutil.copyfile(source / "images" / f"{cid}-raw.png", images / f"{cid}-raw.png")
        for arm, (context, blocks) in ARMS.items():
            path = images / f"{cid}-{arm}.png"
            displays[f"{cid}-{arm}"] = render_observation_view(
                case, features.noise, target["start"], target["stop"], context, blocks, path)
        for rep in range(2):
            selected = read_json(source / "selections.json")[f"{cid}-r{rep}"]["selected"]
            if not any(row["index"] == target["index"] for row in selected):
                raise ValueError("Probe must not silently change an existing target")
            request_id = f"{cid}-window-{target['index']:02d}-image-r{rep}"
            receipt = source / "cached-responses" / f"{request_id}.response.json"
            if not receipt.exists():
                receipt = source / "requests" / receipt.name
            cached[f"{cid}-r{rep}"] = read_json(receipt)
            for arm in order.permutation(list(ARMS)):
                plan.append({"request_id": f"{cid}-{arm}-r{rep}", "case_id": cid,
                             "arm": str(arm), "repeat": rep, "start": target["start"],
                             "stop": target["stop"], "prompt": display_prompt(target["start"], target["stop"]),
                             "images": [f"{cid}-raw.png", f"{cid}-{arm}.png"]})
    save_json(out / "displays.json", displays)
    save_json(out / "cached-target-receipts.json", cached)
    save_json(out / "query-plan.json", plan)
    save_json(out / "protocol.json", {
        "scope": "Five deliberately selected viewed development targets; not independent validation",
        "source_run": str(source), "all_case_ids": all_cases, "targets": TARGETS,
        "arms": {name: {"context_days": value[0], "block_medians": value[1]} for name, value in ARMS.items()},
        "primary": "wide_blocks", "repetitions": 2, "maximum_new_requests": 40,
        "model": config["model"], "temperature": .1, "top_p": .3, "max_tokens": 2048,
        "retries": 0, "order_seed": 20261009,
        "fixed": "Same prompt, global raw image, 180-day target, initial and 84 evidence selection, "
                 "49/91 physical revision mask, shared stage head; paired raw/block axis limits identical",
        "block_policy": "Day-zero anchored disjoint 14-day blocks, >=8 complete raw observations, "
                        "horizontal median bars without interpolation or connecting adjacent blocks",
        "primary_stop": "Both repetitions must improve weak target recall, not reduce total target F1, "
                        "not increase step-negative FP, and not lose target TP in case0082, all versus "
                        "the new same-prompt narrow_raw arm. Otherwise do not proceed to new confirmation. "
                        "Also disclose whole-record F1 and all TP/FP changes versus original base42 scope",
        "interpretation": "A display probe inside the failed84 selection, not a retained full method. "
                          "No numerical text or ground-truth annotations in any model request",
    })
    bindings = {str(source / name): sha(source / name) for name in (
        "freeze.json", "prediction-freeze.json", "analysis-v3/manifest.json")}
    save_json(out / "source-bindings.json", bindings)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print("Observation probe frozen: five targets, four display arms, two repeats, 40 new requests", flush=True)


def verify(out):
    for file, expected in read_json(out / "freeze.json").items():
        if sha(out / file) != expected:
            raise ValueError(f"Frozen observation probe changed: {file}")
    for file, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(file)) != expected:
            raise ValueError("Bound previous experiment changed")
    for file in SOURCE_FILES:
        if sha(ROOT / file) != sha(out / "source" / file):
            raise ValueError(f"Executing source differs from probe freeze: {file}")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only once from prepared state; no repeats or overwrite")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    if len(plan) != config["maximum_new_requests"]:
        raise ValueError("Query population differs from frozen budget")
    save_json(out / "state.json", {"status": "running"})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], len(plan))
    failures, targets = {}, {}
    for index, query in enumerate(plan):
        case, features = load_case_features(out, query["case_id"])
        request_id = query["request_id"]
        values = np.full(len(case.dates), -1, dtype=int)
        status = "failed"
        try:
            payload = transport.ask(request_id, query["prompt"], [out / "images" / file for file in query["images"]])
            values = parse_window_activity(payload, features.observed, query["start"], query["stop"])
            status = "success"
        except (ValueError, RuntimeError) as error:
            failures[request_id] = str(error)
        method = f"obs_{query['arm']}_r{query['repeat']}"
        targets.setdefault(method, {})[query["case_id"]] = {
            "activity": values.tolist(), "status": status, "start": query["start"], "stop": query["stop"]}
        save_json(out / "failures.json", failures)
        save_json(out / "target-reviews.json", targets)
        print(f"Observation request {index + 1}/{len(plan)}: {request_id} {status}", flush=True)
    save_json(out / "prediction-freeze.json", {"target-reviews.json": sha(out / "target-reviews.json"),
                                               "failures.json": sha(out / "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "new_requests": len(plan)})


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
