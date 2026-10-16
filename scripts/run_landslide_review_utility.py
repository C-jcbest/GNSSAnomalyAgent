"""Freeze, collect and fit a small paired visual-correction selection experiment."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json, render_window
from summarize_landslide_background_benchmark import recompute_numerical
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_review_utility import (
    FEATURE_NAMES,
    correction_utility,
    fit_utility_model,
    predict_utility,
    select_utility_windows,
    window_review_features,
)
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity

PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def prepare(source, out):
    verify_data(source)
    split = read_json(source / "manifest.json")["split"]
    ids = {"train": split["train"][:6], "validation": split["validation"][:6]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    save_json(out / "protocol.json", {
        "scope": "Development correction-utility probe, first six common train and validation IDs; "
                 "not independent, one correlated background per phase. Existing numerical training24/validation12",
        "source": str(source.resolve()), "split": ids, "model": "qwen3.8-flash", "repetitions": 2,
        "maximum_requests": 216, "parallel_requests": 2, "retries": 0, "hard_timeout_seconds": 21600,
        "primary": "calibrated_utility", "features": FEATURE_NAMES, "ridge_alpha": 1.,
        "calibration": "Train actual net newly correct decisions in original49-day authority, divided by eligible "
                       "known dates. Unknown outputs are incorrect, so negative abstention earns no gain",
        "selection": "Predict net correct decisions, admit only score>0, at most2 disjoint180day views, stride90; "
                     "zero is literal expected benefit, no threshold or regularization search",
        "ablations": "Learned visual-only first2 features; uncalibrated trained raw84 expected-error priority; "
                     "original raw conflict; fixed equal-spacing; same-record same-count fixed",
        "inference_order": "All training global/local collected; fit only train references; models frozen BEFORE "
                           "validation initial. All validation selections saved BEFORE any validation local answer",
        "counterfactual_collection": "Eight windows per record collected for action-value measurement and controls; "
                                     "not deployed bounded-method request count. Include all collection/training cost",
        "revision": "Same original49-day raw conflict authority; unchanged global/local plots and image-only prompt",
        "stage": "Same common background-train24 numerical head gated by final activity; no native VLM ASD claim",
        "quality_gate": "Stop before local batch if>=3 of12 direct replies fail; stop if training has<2 records "
                        "with any positive actual utility; transport/schema failures remain unknown, no retries",
        "continue_gate": "Both repeats main dailyF1>direct, weakR>=direct, eventF1>=direct, "
                         "mainF1>=same-count fixed and uncalibrated prior; positive net activity corrections in>=2 records. "
                         "Otherwise stop, do not swap best ablation",
        "new_records": 0, "references": "Known injection recovery only, not field total activity or expert truth",
    })
    files = [Path(__file__), ROOT / "src/gnss_sim/landslide_review_utility.py",
             ROOT / "scripts/probe_landslide_revision_scope.py", ROOT / "scripts/summarize_landslide_background_benchmark.py",
             ROOT / "scripts/run_landslide_comparison.py", ROOT / "scripts/run_landslide_visual_comparison.py",
             ROOT / "scripts/run_landslide_window_optimization.py"]
    files.extend(ROOT.glob("src/gnss_sim/*.py"))
    source_copy = out / "source"
    source_copy.mkdir()
    for path in dict.fromkeys(files):
        shutil.copyfile(path, source_copy / path.name)
    records, probabilities = {}, {}
    numerical = XGBClassifier()
    numerical.load_model(source / "xgboost_84.json")
    for cid in ids["train"] + ids["validation"]:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(source / "inputs" / cid / name, target / name)
        case, feature = load_case_features(out, cid)
        records[cid] = case, feature
        evidence = {scale: raw_motion_evidence(case, feature, scale) for scale in (42, 84)}
        matrix = np.column_stack((feature.matrix,
                                  *[evidence[scale][key] for scale in (42, 84)
                                    for key in ("rate", "signal_to_noise", "direction")]))
        probabilities[cid] = numerical.predict_proba(matrix)[:, 1].astype(float).tolist()
        render_raw(case, out / "images" / f"{cid}-raw.png")
        for index, (start, stop) in enumerate(regular_windows(len(case.dates), 180, 90)):
            render_window(case, feature.noise, start, stop, out / "images" / f"{cid}-window-{index:02d}.png", PLOT)
    save_json(out / "numeric-predictions.json", recompute_numerical(source, records))
    save_json(out / "numeric-probabilities.json", probabilities)
    shutil.copyfile(source / "stage-xgboost.json", out / "stage-xgboost.json")
    save_json(out / "external-bindings.json", {str(path.resolve()): sha(path) for path in
        (source / "manifest.json", source / "long-selection.json", source / "xgboost_84.json", source / "reference-labels.csv")})
    save_json(out / "execution-bindings.json", {str(path.resolve()): sha(path) for path in dict.fromkeys(files)})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "split": ids, "request_ceiling": 216}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen utility inputs changed")
    for file in ("execution-bindings.json", "external-bindings.json"):
        for name, expected in read_json(out / file).items():
            if sha(Path(name)) != expected:
                raise ValueError("Utility execution source or common control changed")


def training_labels(source, ids):
    labels = {cid: [] for cid in ids}
    with (source / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    return {cid: np.asarray([int(row["activity_label"]) for row in values]) for cid, values in labels.items()}


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a new prepared experiment, without retries")
    config = read_json(out / "protocol.json")
    source = Path(config["source"])
    policy = ReviewPolicy()
    ids = config["split"]["train"] + config["split"]["validation"]
    records = {cid: load_case_features(out, cid) for cid in ids}
    long = {cid: raw_motion_evidence(case, feature) for cid, (case, feature) in records.items()}
    probabilities = read_json(out / "numeric-probabilities.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    initial_saved, local_saved, failures, selections, feature_saved = {}, {}, {}, {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "phase": "training_global"})

    def ask(job):
        cid, rep, index = job
        case, feature = records[cid]
        raw = out / "images" / f"{cid}-raw.png"
        request = f"{cid}-direct-r{rep}"
        start, stop = 0, len(case.dates)
        prompt = ACTIVITY_TASK.format(last=stop - 1) + (
            '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
            "Adjacent intervals never share endpoints; use [100,149],[150,200].")
        images = [raw]
        if index is not None:
            start, stop = regular_windows(len(case.dates), 180, 90)[index]
            request = f"{cid}-window-{index:02d}-image-r{rep}"
            prompt = review_prompt(feature, long[cid], start, stop, False)
            images.append(out / "images" / f"{cid}-window-{index:02d}.png")
        try:
            payload = transport.ask(request, prompt, images)
            activity = parse_activity(payload, feature.observed) if index is None else parse_window_activity(
                payload, feature.observed, start, stop)
            status, error = "success", None
        except (ValueError, RuntimeError) as exception:
            activity, status, error = np.full(stop if index is None else len(case.dates), -1), "failed", str(exception)
        return request, {"case_id": cid, "repeat": rep, "index": index, "start": start, "stop": stop,
                         "activity": activity.tolist(), "status": status}, error

    def collect(jobs, phase):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Frozen wall-time budget exceeded")
        save_json(out / "state.json", {"status": "running", "phase": phase})
        with ThreadPoolExecutor(max_workers=config["parallel_requests"]) as pool:
            pending = [pool.submit(ask, job) for job in jobs]
            for task in as_completed(pending):
                request, result, error = task.result()
                if result["index"] is None:
                    initial_saved[request] = result
                else:
                    local_saved[request] = result
                if error is not None:
                    failures[request] = error
                save_json(out / "initial-predictions.json", initial_saved)
                save_json(out / "local-predictions.json", local_saved)
                save_json(out / "failures.json", failures)
                print(f"{phase}: {request} {result['status']}; completed={len(initial_saved)+len(local_saved)}", flush=True)

    def globals_for(group, phase):
        collect([(cid, rep, None) for rep in (0, 1) for cid in group], phase)
        failed = sum(initial_saved[f"{cid}-direct-r{rep}"]["status"] == "failed" for rep in (0, 1) for cid in group)
        if failed >= 3:
            raise RuntimeError("Frozen direct-response quality gate failed; no local batch")

    def locals_for(group, phase):
        collect([(cid, rep, index) for rep in (0, 1) for cid in group
                 for index in range(len(regular_windows(len(records[cid][0].dates), 180, 90)))], phase)

    try:
        train = config["split"]["train"]
        globals_for(train, "training_global")
        locals_for(train, "training_local")
        actual = training_labels(source, train)
        calibration_rows, targets, matrices, gain_records = [], [], [], set()
        for rep in (0, 1):
            for cid in train:
                case, feature = records[cid]
                windows = regular_windows(len(case.dates), 180, 90)
                initial = np.asarray(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
                matrix = window_review_features(initial, feature, long[cid], probabilities[cid], windows, policy)
                neighbourhood = conflict_neighbourhood(initial, feature, long[cid], policy)[1]
                for index, (start, stop) in enumerate(windows):
                    permission = np.zeros(len(initial), dtype=bool)
                    permission[start:stop] = neighbourhood[start:stop]
                    local = local_saved[f"{cid}-window-{index:02d}-image-r{rep}"]
                    if local["status"] == "failed":
                        continue
                    utility = correction_utility(actual[cid], initial, np.asarray(local["activity"]), permission)
                    matrices.append(matrix[index])
                    targets.append(utility["utility_fraction"])
                    calibration_rows.append({"case_id": cid, "repeat": rep, "index": index, **utility})
                    if utility["net_correct"] > 0:
                        gain_records.add(cid)
        save_json(out / "training-utilities.json", calibration_rows)
        if len(gain_records) < 2:
            raise RuntimeError("Training views lack corrective opportunity in two records; stop before validation")
        models = {"calibrated_utility": fit_utility_model(matrices, targets, list(range(5))),
                  "visual_only_utility": fit_utility_model(matrices, targets, [0, 1])}
        save_json(out / "utility-models.json", models)
        save_json(out / "training-prediction-snapshot.json", {"initial": initial_saved, "local": local_saved})
        # Combined prediction files acquire validation results; freeze the training snapshot instead.
        save_json(out / "calibration-freeze.json", {name: sha(out / name) for name in
                                                   ("utility-models.json", "training-utilities.json", "training-prediction-snapshot.json")})
        validation = config["split"]["validation"]
        globals_for(validation, "validation_global")
        for rep in (0, 1):
            for cid in validation:
                case, feature = records[cid]
                initial = np.asarray(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
                windows = regular_windows(len(case.dates), 180, 90)
                matrix = window_review_features(initial, feature, long[cid], probabilities[cid], windows, policy)
                neighbourhood = conflict_neighbourhood(initial, feature, long[cid], policy)[1]
                editable = [int(neighbourhood[start:stop].sum()) for start, stop in windows]
                arms = {}
                for name, model in models.items():
                    priorities, chosen = select_utility_windows(windows, predict_utility(model, matrix), editable)
                    arms[name] = {"priorities": priorities, "selected": chosen}
                priorities, chosen = select_utility_windows(windows, matrix[:, 4], editable)
                arms["uncalibrated_prior"] = {"priorities": priorities, "selected": chosen}
                priorities, chosen = review_priorities(initial, feature, long[cid], windows, policy)
                arms["raw_conflict"] = {"priorities": priorities, "selected": chosen}
                centers = np.linspace(0, len(case.dates), 4)[1:-1]
                fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index])/2 - center)) for center in centers})
                fixed_targets = [{"index": index, "start": windows[index][0], "stop": windows[index][1]} for index in fixed]
                arms["fixed_review"] = {"selected": fixed_targets}
                count = len(arms["calibrated_utility"]["selected"])
                arms["same_count_fixed"] = {"selected": fixed_targets[:count]}
                for name, arm in arms.items():
                    scope = target_scopes(neighbourhood, feature.observed, arm["selected"])[0]
                    arm["editable_indices"] = np.flatnonzero(scope).tolist()
                selections[f"{cid}-r{rep}"] = arms
                feature_saved[f"{cid}-r{rep}"] = matrix.tolist()
        save_json(out / "selections.json", selections)
        save_json(out / "selection-features.json", feature_saved)
        save_json(out / "selection-freeze.json", {name: sha(out / name) for name in
                                                 ("selections.json", "selection-features.json", "utility-models.json")})
        locals_for(validation, "validation_local")
        verify(out)
        save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
            ("initial-predictions.json", "local-predictions.json", "failures.json", "selections.json", "utility-models.json")})
        save_json(out / "state.json", {"status": "predictions_complete", "seconds": time.monotonic() - started})
    except Exception as error:
        save_json(out / "state.json", {"status": "stopped", "reason": str(error),
                                       "seconds": time.monotonic() - started,
                                       "completed_requests": len(initial_saved) + len(local_saved)})
        raise


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
