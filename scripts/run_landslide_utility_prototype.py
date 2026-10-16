"""Apply the frozen six-record utility calibrator to complete background development."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_review_utility import (
    predict_utility,
    select_utility_windows,
    window_review_features,
)
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity


def prepare(source, pilot, out):
    verify_freeze(source)
    for name, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / name) != expected:
            raise ValueError("Original background predictions changed")
    for name, expected in read_json(pilot / "calibration-freeze.json").items():
        if sha(pilot / name) != expected:
            raise ValueError("Pilot utility calibration changed")
    if not read_json(pilot / "analysis-v2/decision.json")["allows_full_prototype"]:
        raise ValueError("Pilot did not pass the registered complete-prototype gate")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "cached-requests").mkdir()
    source_config = read_json(source / "protocol.json")
    data = ROOT / source_config["source_bundle"]
    ids = source_config["case_ids"]
    model = XGBClassifier()
    model.load_model(data / "xgboost_84.json")
    models = read_json(pilot / "utility-models.json")
    original = read_json(source / "predictions.json")
    selections, features_saved, cached, missing = {}, {}, {}, []
    for name in ("stage-xgboost.json", "numeric-predictions.json"):
        shutil.copyfile(source / name, out / name)
    shutil.copyfile(pilot / "utility-models.json", out / "utility-models.json")
    for cid in ids:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(source / "inputs" / cid / name, target / name)
        for image in (source / "images").glob(f"{cid}-*.png"):
            shutil.copyfile(image, out / "images" / image.name)
        case, feature = load_case_features(out, cid)
        windows = regular_windows(len(case.dates), 180, 90)
        evidence = {scale: raw_motion_evidence(case, feature, scale) for scale in (42, 84)}
        numerical_matrix = np.column_stack((feature.matrix,
            *[evidence[scale][key] for scale in (42, 84) for key in ("rate", "signal_to_noise", "direction")]))
        probability = model.predict_proba(numerical_matrix)[:, 1]
        for rep in (0, 1):
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            matrix = window_review_features(initial, feature, evidence[42], probability, windows, ReviewPolicy())
            neighbourhood = conflict_neighbourhood(initial, feature, evidence[42], ReviewPolicy())[1]
            editable = [int(neighbourhood[start:stop].sum()) for start, stop in windows]
            arms = {}
            for name, selector in models.items():
                priorities, selected = select_utility_windows(windows, predict_utility(selector, matrix), editable)
                arms[name] = {"priorities": priorities, "selected": selected}
            priorities, selected = select_utility_windows(windows, matrix[:, 4], editable)
            arms["uncalibrated_prior"] = {"priorities": priorities, "selected": selected}
            priorities, selected = review_priorities(initial, feature, evidence[42], windows, ReviewPolicy())
            arms["raw_conflict"] = {"priorities": priorities, "selected": selected}
            centers = np.linspace(0, len(case.dates), 4)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index])/2-center)) for center in centers})
            fixed_targets = [{"index": index, "start": windows[index][0], "stop": windows[index][1]} for index in fixed]
            arms["fixed_review"] = {"selected": fixed_targets}
            arms["same_count_fixed"] = {"selected": fixed_targets[:len(arms["calibrated_utility"]["selected"])]}
            needed = set()
            for name, arm in arms.items():
                scope = target_scopes(neighbourhood, feature.observed, arm["selected"])[0]
                arm["editable_indices"] = np.flatnonzero(scope).tolist()
                needed.update(row["index"] for row in arm["selected"])
            selections[f"{cid}-r{rep}"] = arms
            features_saved[f"{cid}-r{rep}"] = matrix.tolist()
            for index in sorted(needed):
                key = f"{cid}-window-{index:02d}-image-r{rep}"
                response = source / "requests" / f"{key}.response.json"
                if response.exists():
                    receipt = read_json(response)
                    start, stop = windows[index]
                    payload = read_json(response)["output"]
                    activity = parse_window_activity(json.loads(payload), feature.observed, start, stop)
                    cached[key] = {"request_id": key, "case_id": cid, "repeat": rep, "index": index,
                                   "start": start, "stop": stop, "activity": activity.tolist(), "status": "success",
                                   "receipt_sha256": sha(response)}
                    for suffix in ("started", "response"):
                        shutil.copyfile(source / "requests" / f"{key}.{suffix}.json", out / "cached-requests" / f"{key}.{suffix}.json")
                    if receipt["status"] != "returned_json":
                        raise ValueError("A cached failed observation requires explicit unknown handling")
                else:
                    missing.append({"request_id": key, "case_id": cid, "repeat": rep, "index": index})
    save_json(out / "protocol.json", {
        "scope": "Complete24 exposed residual-background development, four shared backgrounds, not independent",
        "source": str(source.resolve()), "pilot": str(pilot.resolve()), "case_ids": ids,
        "primary": "calibrated_utility", "model": source_config["model"], "repetitions": 2,
        "utility_training": "Exact pilot six common-train records; no refit, threshold search or arm substitution",
        "numerical_training": "Original commontrain24/validation12. Include existing direction/scale controls where available",
        "maximum_requests": len(missing), "parallel_requests": 2, "retries": 0,
        "selection": "Same positive expected net corrections, up to2 disjoint180day windows, original49-day authority",
        "continue_gate": "Both repeats mainF1>direct, weakR>=direct, eventF1>=direct, "
                         "mainF1>=same-count fixed and uncalibrated prior, positive net correct decisions in>=2 records",
        "cache": "Global and existing local answers shared by every arm. New requests only absent selected image views",
        "cost": "Report new calls, cached deployed flow, and pilot216 calibration/counterfactual collection separately",
        "references": "Known injected recovery, not residual-no-motion or field total labels", "new_records": 0,
    })
    for name, value in (("selections.json", selections), ("selection-features.json", features_saved),
                        ("cached-locals.json", cached), ("missing-requests.json", missing)):
        save_json(out / name, value)
    files = [Path(__file__), ROOT / "src/gnss_sim/landslide_review_utility.py",
             ROOT / "scripts/probe_landslide_revision_scope.py", ROOT / "scripts/run_landslide_comparison.py",
             ROOT / "scripts/run_landslide_window_optimization.py", ROOT / "src/gnss_sim/landslide_visual.py",
             ROOT / "src/gnss_sim/landslide_adaptive_review.py", ROOT / "src/gnss_sim/landslide_window_experiment.py",
             source / "predictions.json", source / "selections.json", pilot / "utility-models.json",
             pilot / "analysis-v2/decision.json", data / "xgboost_84.json"]
    save_json(out / "execution-bindings.json", {str(path.resolve()): sha(path) for path in files})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"missing_requests": len(missing), "cached_local_requests": len(cached), "case_ids": len(ids)}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen complete utility prototype changed")
    for name, expected in read_json(out / "execution-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Complete prototype source or pilot calibration changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a newly prepared prototype may run, no retries")
    config = read_json(out / "protocol.json")
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    long = {cid: raw_motion_evidence(case, feature) for cid, (case, feature) in records.items()}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    locals_saved, failures = {}, {}
    save_json(out / "new-locals.json", locals_saved)
    save_json(out / "failures.json", failures)
    save_json(out / "state.json", {"status": "running"})
    started = time.monotonic()

    def ask(job):
        cid, index = job["case_id"], job["index"]
        case, feature = records[cid]
        start, stop = regular_windows(len(case.dates), 180, 90)[index]
        prompt = review_prompt(feature, long[cid], start, stop, False)
        images = [out / "images" / f"{cid}-raw.png", out / "images" / f"{cid}-window-{index:02d}.png"]
        try:
            payload = transport.ask(job["request_id"], prompt, images)
            activity = parse_window_activity(payload, feature.observed, start, stop)
            status, error = "success", None
        except (ValueError, RuntimeError) as exception:
            activity, status, error = np.full(len(case.dates), -1), "failed", str(exception)
        return {**job, "start": start, "stop": stop, "activity": activity.tolist(), "status": status}, error

    with ThreadPoolExecutor(max_workers=config["parallel_requests"]) as pool:
        pending = [pool.submit(ask, job) for job in read_json(out / "missing-requests.json")]
        for task in as_completed(pending):
            result, error = task.result()
            key = result["request_id"]
            locals_saved[key] = result
            if error is not None:
                failures[key] = error
            save_json(out / "new-locals.json", locals_saved)
            save_json(out / "failures.json", failures)
            print(f"{key} {result['status']}; new completed={len(locals_saved)}", flush=True)
    verify(out)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("new-locals.json", "failures.json", "selections.json", "utility-models.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "seconds": time.monotonic()-started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None or args.pilot is None:
            parser.error("prepare requires --source and --pilot")
        prepare(args.source, args.pilot, args.out)
    else:
        execute(args.out)
