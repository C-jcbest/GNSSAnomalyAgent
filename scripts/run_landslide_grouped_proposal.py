"""Calibrate deployed visual proposals on all four common training backgrounds."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_proposal_utility_v2 import raw_prior_matrix
from run_landslide_review_utility import training_labels
from run_landslide_review_utility import verify as verify_pilot
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
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_proposal_utility import (
    FEATURE_NAMES,
    apply_proposal,
    expected_proposal_utility,
    fit_proposal_model,
    proposal_features,
)
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity


def prepare(data, pilot, out):
    verify_data(data)
    verify_pilot(pilot)
    for name, expected in read_json(pilot / "prediction-freeze.json").items():
        if sha(pilot / name) != expected:
            raise ValueError("Cached pilot predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(data / "manifest.json")["split"]
    train, validation = split["train"], split["validation"]
    ids = train + validation
    source_files = [path for folder in (ROOT / "scripts", ROOT / "src/gnss_sim") for path in folder.glob("*.py")]
    bindings = {}
    for path in source_files:
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        bindings[str(path.resolve())] = sha(path)
    for folder in (out / "images", out / "cached-requests"):
        folder.mkdir()
    groups = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["background_group"] for cid in train}
    save_json(out / "protocol.json", {
        "date": "2026-10-10", "scope": "Development, commontrain24 on four backgrounds and validation12 on two, "
                                         "partly exposed. No independent confirmation or expert references",
        "data": str(data.resolve()), "pilot": str(pilot.resolve()), "split": {"train": train, "validation": validation},
        "primary": "grouped_proposal_guard", "training_groups": groups,
        "change": "Replace single-background exhaustive-view calibration by commontrain24 deployment-selected "
                  "proposals and background-excluded numerical predictions; no change to features/alpha/admission",
        "features": FEATURE_NAMES, "ridge_alpha": 1, "admission_threshold": 0,
        "training_unit": "Only raw-conflict-selected at most2 views per record/repeat, chosen before local answers; "
                         "four shared backgrounds, repeated and daily samples not independent",
        "prior": "Four leave-one-background-out raw84 XGBoost models with existing fixed selected parameters. "
                 "Validation gets original commontrain24 prior. No tuning or validation-labelled guard training",
        "selection": "Original raw21/42 conflicts,14-day trigger and two nonoverlapping180-day windows, original49 authority",
        "acceptance": "Commit the complete permitted VLM proposal iff predicted net definite correction fraction>0; "
                      "rejection preserves initial including unknown; numerical prior cannot invent activity",
        "ablations": ["direct", "raw_conflict", "fixed_review", "visual_guard", "numeric_vote", "fixed_grouped_guard"],
        "continue_gate": "Both repeats main dailyF1>direct, weakR>=direct, mainF1>=visual guard and numeric vote; "
                         "strictF1 gain over numeric vote in at least1 repeat, >=2 positive net corrected records each. "
                         "Event/boundary/stage/cost tradeoffs reported in full, not required universally noninferior",
        "gate_rationale": "Primary is binary activity recovery and evidence that learned approval adds more than "
                          "numerical endorsement; user does not require every metric superior. Frozen before new outputs",
        "stage": "Original common numerical head gated by final activity, not native VLM ASD",
        "model": "qwen3.8-flash", "repetitions": 2, "maximum_requests": 168, "parallel_requests": 2,
        "retries": 0, "new_records": 0,
        "cache": "Reuse only original pilot answers; every selected cached or new review charged even if rejected. "
                 "Models freeze before new validation global/local calls; cached validation already exposed",
    })
    for cid in ids:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(data / "inputs" / cid / name, target / name)
        case, features = load_case_features(out, cid)
        if (pilot / "images" / f"{cid}-raw.png").exists():
            for path in (pilot / "images").glob(f"{cid}-*.png"):
                shutil.copyfile(path, out / "images" / path.name)
        else:
            render_raw(case, out / "images" / f"{cid}-raw.png")
            for index, (start, stop) in enumerate(regular_windows(len(case.dates), 180, 90)):
                render_window(case, features.noise, start, stop, out / "images" / f"{cid}-window-{index:02d}.png",
                              {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150})
    for name in ("initial-predictions.json", "local-predictions.json"):
        shutil.copyfile(pilot / name, out / f"cached-{name}")
        bindings[str((pilot / name).resolve())] = sha(pilot / name)
    for path in (pilot / "requests").glob("*.json"):
        shutil.copyfile(path, out / "cached-requests" / path.name)
        bindings[str(path.resolve())] = sha(path)
    shutil.copyfile(data / "stage-xgboost.json", out / "stage-xgboost.json")
    records = {cid: load_case_features(out, cid) for cid in ids}
    numeric = recompute_numerical(data, records)
    save_json(out / "numeric-predictions.json", numeric)
    actual = training_labels(data, train)
    matrices = {cid: raw_prior_matrix(*records[cid])[0] for cid in ids}
    depth = read_json(data / "long-selection.json")["xgboost_84"]["depth"]
    probabilities = {}
    exclusions = {}
    for group in sorted(set(groups.values())):
        fitting = [cid for cid in train if groups[cid] != group]
        eligible = []
        for cid in fitting:
            case, features = records[cid]
            evidence = {scale: raw_motion_evidence(case, features, scale) for scale in (42, 84)}
            support = features.observed & (np.isfinite(features.progressive_rate)
                | np.isfinite(evidence[42]["rate"]) | np.isfinite(evidence[84]["rate"]))
            eligible.extend((support & (actual[cid] >= 0)).tolist())
        matrix = np.concatenate([matrices[cid] for cid in fitting])
        target = np.concatenate([actual[cid] for cid in fitting])
        eligible = np.array(eligible)
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                              n_jobs=2, random_state=20261007)
        model.fit(matrix[eligible], target[eligible])
        model.save_model(out / f"prior-excluding-background-{group}.json")
        exclusions[str(group)] = fitting
        for cid in train:
            if groups[cid] == group:
                probabilities[cid] = model.predict_proba(matrices[cid])[:, 1].astype(float).tolist()
    model = XGBClassifier()
    model.load_model(data / "xgboost_84.json")
    for cid in validation:
        probabilities[cid] = model.predict_proba(matrices[cid])[:, 1].astype(float).tolist()
    save_json(out / "prior-exclusions.json", exclusions)
    save_json(out / "numeric-probabilities.json", probabilities)
    for name in ("manifest.json", "reference-labels.csv", "long-selection.json", "xgboost_84.json", "stage-xgboost.json"):
        bindings[str((data / name).resolve())] = sha(data / name)
    save_json(out / "execution-bindings.json", bindings)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "train": len(train), "validation": len(validation), "new_request_ceiling": 168}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Grouped proposal frozen inputs changed")
    for name, expected in read_json(out / "execution-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Grouped proposal execution source or baseline changed")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a fresh prepared grouped experiment may execute, no retries")
    config = read_json(out / "protocol.json")
    train, validation = config["split"]["train"], config["split"]["validation"]
    records = {cid: load_case_features(out, cid) for cid in train + validation}
    initial_saved = read_json(out / "cached-initial-predictions.json")
    local_saved = read_json(out / "cached-local-predictions.json")
    probabilities = read_json(out / "numeric-probabilities.json")
    policy = ReviewPolicy()
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    failures, selections = {}, {}

    def new_requests():
        return len(list((out / "requests").glob("*.started.json")))

    def ask(job):
        cid, rep, index = job
        case, features = records[cid]
        start, stop = 0, len(case.dates)
        prompt = ACTIVITY_TASK.format(last=stop - 1) + (
            '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
            "Adjacent intervals never share endpoints; use [100,149],[150,200].")
        images = [out / "images" / f"{cid}-raw.png"]
        key = f"{cid}-direct-r{rep}"
        if index is not None:
            start, stop = regular_windows(stop, 180, 90)[index]
            key = f"{cid}-window-{index:02d}-image-r{rep}"
            prompt = review_prompt(features, raw_motion_evidence(case, features), start, stop, False)
            images.append(out / "images" / f"{cid}-window-{index:02d}.png")
        try:
            payload = transport.ask(key, prompt, images)
            activity = parse_activity(payload, features.observed) if index is None else parse_window_activity(
                payload, features.observed, start, stop)
            status, error = "success", None
        except (ValueError, RuntimeError) as exception:
            activity, status, error = np.full(len(case.dates), -1), "failed", str(exception)
        return key, {"case_id": cid, "repeat": rep, "index": index, "start": start, "stop": stop,
                     "activity": activity.tolist(), "status": status}, error

    def collect(jobs, phase):
        save_json(out / "state.json", {"status": "running", "phase": phase})
        with ThreadPoolExecutor(max_workers=config["parallel_requests"]) as pool:
            pending = [pool.submit(ask, job) for job in jobs]
            for task in as_completed(pending):
                key, result, error = task.result()
                if result["index"] is None:
                    initial_saved[key] = result
                else:
                    local_saved[key] = result
                if error is not None:
                    failures[key] = error
                save_json(out / "initial-predictions.json", initial_saved)
                save_json(out / "local-predictions.json", local_saved)
                save_json(out / "failures.json", failures)
                print(f"{phase}: {key} {result['status']}; new_started={new_requests()}", flush=True)

    def choose(group, phase):
        collect([(cid, rep, None) for cid in group for rep in (0, 1)
                 if f"{cid}-direct-r{rep}" not in initial_saved], phase + "_global")
        failed = sum(initial_saved[f"{cid}-direct-r{rep}"]["status"] != "success" for cid in group for rep in (0, 1))
        if failed >= 3:
            raise RuntimeError("Grouped direct quality gate failed; no local batch")
        jobs = []
        for cid in group:
            case, features = records[cid]
            windows = regular_windows(len(case.dates), 180, 90)
            for rep in (0, 1):
                initial = np.array(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
                _, selected = review_priorities(initial, features, raw_motion_evidence(case, features), windows, policy)
                centers = np.linspace(0, len(case.dates), 4)[1:-1]
                fixed_indices = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index])/2-center))
                                        for center in centers})
                fixed = [{"index": index, "start": windows[index][0], "stop": windows[index][1]} for index in fixed_indices]
                selections[f"{cid}-r{rep}"] = {"raw": selected, "fixed": fixed}
                needed = {row["index"] for row in selected}
                if group == validation:
                    needed.update(fixed_indices)
                for index in sorted(needed):
                    if f"{cid}-window-{index:02d}-image-r{rep}" not in local_saved:
                        jobs.append((cid, rep, index))
        save_json(out / "selections.json", selections)
        phase_snapshot = f"{phase}-selections.json"
        save_json(out / phase_snapshot, selections)
        save_json(out / f"{phase}-selection-freeze.json", {phase_snapshot: sha(out / phase_snapshot)})
        collect(jobs, phase + "_local")

    def proposal(cid, rep, target):
        case, features = records[cid]
        initial = np.array(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
        reviewed = np.array(local_saved[f"{cid}-window-{target['index']:02d}-image-r{rep}"]["activity"])
        long = raw_motion_evidence(case, features)
        neighbourhood = conflict_neighbourhood(initial, features, long, policy)[1]
        permission = target_scopes(neighbourhood, features.observed, [target])[0]
        row, changed = proposal_features(initial, reviewed, permission, features, long, probabilities[cid], policy)
        return initial, reviewed, permission, row, changed

    try:
        choose(train, "training")
        # Persist all cached answers as well if there were no uncached local jobs.
        save_json(out / "initial-predictions.json", initial_saved)
        save_json(out / "local-predictions.json", local_saved)
        actual = training_labels(Path(config["data"]), train)
        training, rows, targets = [], [], []
        for cid in train:
            for rep in (0, 1):
                for selected in selections[f"{cid}-r{rep}"]["raw"]:
                    initial, reviewed, permission, row, changed = proposal(cid, rep, selected)
                    utility = correction_utility(actual[cid], initial, reviewed, changed)
                    rows.append(row)
                    targets.append(utility["utility_fraction"])
                    training.append({"case_id": cid, "repeat": rep, "index": selected["index"],
                                     "features": row.tolist(), **utility})
        models = {"grouped_proposal_guard": fit_proposal_model(np.array(rows), targets, list(range(5))),
                  "visual_guard": fit_proposal_model(np.array(rows), targets, [0, 1, 2])}
        save_json(out / "training-utilities.json", training)
        save_json(out / "models.json", models)
        save_json(out / "model-freeze.json", {name: sha(out / name) for name in ("models.json", "training-utilities.json")})
        choose(validation, "validation")
        head = XGBClassifier()
        head.load_model(out / "stage-xgboost.json")
        predictions = {method: {cid: value for cid, value in values.items() if cid in validation}
                       for method, values in read_json(out / "numeric-predictions.json").items() if method != "ungated_rule"}
        log, permissions, used_requests = [], {}, {}
        for cid in validation:
            case, features = records[cid]
            codes = head.predict(features.matrix).astype(int) + 1
            for rep in (0, 1):
                initial = np.array(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
                method = f"direct_visual_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, initial, features.supported).to_dict()
                used_requests.setdefault(method, []).append(f"{cid}-direct-r{rep}")
                for stem in ("grouped_proposal_guard", "visual_guard", "numeric_vote", "raw_conflict",
                             "fixed_review", "fixed_grouped_guard"):
                    activity = initial.copy()
                    authority = np.zeros(len(initial), dtype=bool)
                    selected = selections[f"{cid}-r{rep}"]["fixed" if stem.startswith("fixed") else "raw"]
                    method = f"{stem}_r{rep}"
                    keys = [f"{cid}-direct-r{rep}"]
                    for choice in selected:
                        _, reviewed, permission, row, changed = proposal(cid, rep, choice)
                        if stem in ("raw_conflict", "fixed_review"):
                            expected = 1.
                        elif stem == "numeric_vote":
                            expected = float(row[4])
                        else:
                            expected = expected_proposal_utility(models["grouped_proposal_guard" if stem.startswith("fixed") else stem], row)
                        accepted = expected > 0 and changed.any()
                        if accepted:
                            activity = apply_proposal(activity, reviewed, permission, expected)
                            authority |= permission
                        keys.append(f"{cid}-window-{choice['index']:02d}-image-r{rep}")
                        log.append({"method": method, "case_id": cid, "index": choice["index"],
                                    "features": row.tolist(), "expected_utility": expected,
                                    "changed_days": int(changed.sum()), "accepted": bool(accepted)})
                    if np.any(activity[~authority] != initial[~authority]) or np.any(activity[~features.observed] != -1):
                        raise ValueError("Grouped guard changed unauthorized dates")
                    predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, features.supported).to_dict()
                    permissions[f"{method}-{cid}"] = np.flatnonzero(authority).tolist()
                    used_requests.setdefault(method, []).extend(keys)
        for name, value in (("predictions.json", predictions), ("proposals.json", log), ("permissions.json", permissions),
                            ("used-requests.json", used_requests), ("failures.json", failures)):
            save_json(out / name, value)
        verify(out)
        save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
            ("initial-predictions.json", "local-predictions.json", "models.json", "selections.json", "predictions.json",
             "proposals.json", "permissions.json", "used-requests.json", "failures.json")})
        save_json(out / "state.json", {"status": "predictions_complete", "new_requests": new_requests(),
                                      "training_proposals": len(training), "validation_proposals": len(log)})
    except Exception as exception:
        save_json(out / "state.json", {"status": "stopped", "reason": str(exception), "new_requests": new_requests()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.data is None or args.pilot is None:
            parser.error("prepare requires --data and --pilot")
        prepare(args.data, args.pilot, args.out)
    else:
        execute(args.out)
