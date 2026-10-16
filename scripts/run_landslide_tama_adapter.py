"""TAMA-inspired GNSS adaptation: train normal references and positive-only review.

This executes the inspected reference/controller design, not the original paper's
generic taxonomy, renderer, model transport or benchmark scores. Prior dialogue is
explicitly flattened to text and retained images in the shared VisualRequests API.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity

REFERENCES = ("case_0001", "case_0007", "case_0013")
NORMAL_PROMPT = """These three images are normal references from the common training split.
They contain observed N/E/U displacement in mm versus original calendar day index.
Describe the normal pattern, including stability, trend, fluctuations and possible
measurement artifacts. Normal here means no confirmed sustained displacement
activity, not zero-valued observations. Do not invent hidden causes, generating
parameters or test labels. Return only JSON with exactly one key:
{"normal_pattern":"a concise description of the three observed references"}.
"""
SCHEMA = '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. No other keys.'
PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def prepare(source, out):
    config = read_json(source / "protocol.json")
    out.mkdir(parents=True, exist_ok=False)
    images = out / "images"
    images.mkdir()
    training = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/motion-probe-v2"
    train_ids = read_json(training / "protocol.json")["split"]["train"]
    if not set(REFERENCES).issubset(train_ids) or set(REFERENCES) & set(config["case_ids"]):
        raise ValueError("Normal references must be disjoint common-training records")
    if read_json(training / "protocol.json")["seed"] != 2026100937:
        raise ValueError("Unexpected training source")
    reference_counts = dict.fromkeys(REFERENCES, 0)
    with (training / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in reference_counts:
                if int(row["activity_label"]) == 1:
                    raise ValueError("Normal training reference contains confirmed activity")
                reference_counts[row["case_id"]] += 1
    if any(count != 730 for count in reference_counts.values()):
        raise ValueError("Incomplete normal training references")
    config.update({
        "schema_version": "tama-gnss-adaptation-v1",
        "scope": "Executed TAMA-inspired GNSS task/controller adaptation, not published TAMA scores",
        "normal_references": list(REFERENCES), "maximum_requests": 50,
        "hard_timeout_seconds": 1800,
        "normal_pattern_calls": "One shared training-reference pattern call per repetition; "
                                "same synthetic GNSS distribution, two independent descriptions",
        "review_controller": "Only nonempty parsed initial activity triggers one full-calendar "
                             "review with first [0,2*n//3) and last [n//3,n) raw views",
        "adaptations": [
            "Three-component GNSS raw plots and continuing-displacement/unknown schema",
            "Shared Qwen model, inference settings, transport, no retries, no JSON repair",
            "Training-only references selected as first three quiet train-family records",
            "Common stage head and common evaluation, no label-dependent point adjustment",
            "Dialogue flattened to labelled prior text plus retained reference/global images",
            "One reference description per distribution/repetition, amortized across test records",
        ],
        "official_code_sha256": sha(ROOT / "artifacts/landslide-adaptive-review-2026-10-09/"
                                         "literature-v1/tama-code-v1/main_cli.py"),
    })
    save_json(out / "protocol.json", config)
    shutil.copyfile(source / "stage-xgboost.json", out / "stage-xgboost.json")
    for cid in (*REFERENCES, *config["case_ids"]):
        origin = training if cid in REFERENCES else source
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for file in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(origin / "inputs" / cid / file, target / file)
        case, feature = load_case_features(out, cid)
        render_raw(case, images / f"{cid}-raw.png")
        if cid not in REFERENCES:
            days = len(case.dates)
            for index, (start, stop) in enumerate(((0, 2 * days // 3), (days // 3, days))):
                render_window(case, feature.noise, start, stop,
                              images / f"{cid}-review-{index}.png", PLOT)
    (out / "source").mkdir()
    for path in (Path(__file__), ROOT / "src/gnss_sim/landslide_visual.py",
                 ROOT / "src/gnss_sim/landslide_evaluation.py",
                 ROOT / "src/gnss_sim/landslide_local_experiment.py",
                 ROOT / "scripts/run_landslide_window_optimization.py",
                 ROOT / "scripts/run_landslide_visual_comparison.py"):
        shutil.copyfile(path, out / "source" / path.name)
    save_json(out / "source-bindings.json", {
        str(source / "freeze.json"): sha(source / "freeze.json"),
        str(training / "manifest.json"): sha(training / "manifest.json"),
    })
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                    for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})


def verify(out):
    for file, expected in read_json(out / "freeze.json").items():
        if sha(out / file) != expected:
            raise ValueError(f"Frozen TAMA adaptation evidence changed: {file}")
    if sha(Path(__file__)) != sha(out / "source" / Path(__file__).name):
        raise ValueError("TAMA adaptation executable changed")
    for file, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(file)) != expected:
            raise ValueError("Bound source changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only newly prepared inference; no retries or overwrites")
    config = read_json(out / "protocol.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               config["maximum_requests"])
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    references = [out / "images" / f"{cid}-raw.png" for cid in REFERENCES]
    predictions, failures, selections = {}, {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running"})
    for rep in range(config["repetitions"]):
        pattern_id = f"normal-pattern-r{rep}"
        try:
            pattern = transport.ask(pattern_id, NORMAL_PROMPT, references)
            if (not isinstance(pattern, dict) or set(pattern) != {"normal_pattern"}
                    or not isinstance(pattern["normal_pattern"], str)
                    or not 1 <= len(pattern["normal_pattern"].strip()) <= 5000):
                raise ValueError("Normal pattern does not match the frozen schema")
            pattern_text = pattern["normal_pattern"]
        except (ValueError, RuntimeError) as error:
            failures[pattern_id] = str(error)
            pattern_text = None
        for cid in config["case_ids"]:
            if time.monotonic() - started > config["hard_timeout_seconds"]:
                raise RuntimeError("TAMA adaptation exceeded frozen wall-time budget")
            case, features = load_case_features(out, cid)
            raw = out / "images" / f"{cid}-raw.png"
            initial = np.full(len(case.dates), -1)
            initial_payload = None
            initial_status = "failed"
            task = ACTIVITY_TASK.format(last=len(case.dates) - 1)
            if pattern_text is not None:
                request_id = f"{cid}-detect-r{rep}"
                prompt = (task + "\nFirst three images are TRAIN normal references; last "
                          "image is the TEST record to label. Normal-reference description "
                          "from the preceding call:\n" + pattern_text + SCHEMA)
                try:
                    initial_payload = transport.ask(request_id, prompt, [*references, raw])
                    initial = parse_activity(initial_payload, features.observed)
                    initial_status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[request_id] = str(error)
            review_triggered = initial_status == "success" and bool(np.any(initial == 1))
            selections[f"{cid}-r{rep}"] = {"positive_initial_trigger": review_triggered}
            save_json(out / "selections.json", selections)
            reviewed = initial.copy()
            reviewed_status = initial_status
            if review_triggered:
                request_id = f"{cid}-review-r{rep}"
                prompt = (task + "\nFirst three images are TRAIN normal references, fourth "
                          "is global TEST context, last two are first/last two-thirds raw TEST "
                          "views. Their axes retain original day indices. Check each prior "
                          "interval; correct boundaries, remove unsupported activity and add "
                          "visibly missed activity anywhere in the test calendar. Retain "
                          "uncertainty. Prior normal description:\n" + pattern_text
                          + "\nPrior activity JSON:\n" + json.dumps(initial_payload) + SCHEMA)
                try:
                    reviewed = parse_activity(transport.ask(request_id, prompt, [
                        *references, raw, out / "images" / f"{cid}-review-0.png",
                        out / "images" / f"{cid}-review-1.png",
                    ]), features.observed)
                    reviewed_status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[request_id] = str(error)
                    reviewed = np.full(len(case.dates), -1)
                    reviewed_status = "failed"
            stages = model.predict(features.matrix).astype(int) + 1
            for method, activity, status in (("tama_gnss_no_review", initial, initial_status),
                                              ("tama_gnss_positive_review", reviewed, reviewed_status)):
                prediction = learned_stages(stages, activity, features.supported)
                if status == "failed":
                    prediction = ActivityPrediction(activity, np.full(len(activity), -1), status="failed")
                predictions.setdefault(f"{method}_r{rep}", {})[cid] = prediction.to_dict()
            save_json(out / "predictions.json", predictions)
            save_json(out / "failures.json", failures)
            print(f"TAMA GNSS {cid} repeat {rep + 1}: review={review_triggered}, "
                  f"requests={len(list((out / 'requests').glob('*.response.json')))}", flush=True)
    verify(out)
    save_json(out / "prediction-freeze.json", {
        file: sha(out / file) for file in ("predictions.json", "failures.json", "selections.json")
    })
    save_json(out / "state.json", {"status": "predictions_complete",
                                   "seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source, args.out)
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
