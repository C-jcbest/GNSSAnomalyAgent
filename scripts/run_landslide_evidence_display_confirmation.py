"""Freeze, generate and execute the registered unseen evidence-display confirmation.

Generation and inference never read evaluation references. All local answers are
shared across the eight workflows; only their registered selection and authority
differ. A started experiment cannot be rerun or silently repaired.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_motion_probe import controlled_case
from probe_landslide_longer_motion_evidence import longer_conflict_scope, scale_positive
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, observation_features, predict_baselines, save_json, sha
from run_landslide_observation_probe import display_prompt
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
)
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evidence_observation import evidence_display_choices
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observation_view import render_observation_view
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
REGISTRATION = ROOT / "configs/landslide-evidence-display-confirmation-v1.json"
SCRIPT_SOURCES = [
    "run_landslide_evidence_display_confirmation.py", "prepare_landslide_motion_probe.py",
    "analyze_landslide_evidence_display_confirmation.py", "analyze_landslide_observation_probe.py",
    "finalize_landslide_adaptive_evidence.py",
    "run_landslide_comparison.py", "run_landslide_window_optimization.py",
    "run_landslide_visual_comparison.py", "run_landslide_observation_probe.py",
    "fit_landslide_raw_persistence_baselines.py", "probe_landslide_revision_scope.py",
    "probe_landslide_longer_motion_evidence.py", "analyze_landslide_adaptive_review.py",
]


def fixed_targets(windows, days, budget=2):
    """Reuse the original equal-calendar-spacing control, without signal access."""
    centers = np.linspace(0, days, budget + 2)[1:-1]
    indices = sorted({min(range(len(windows)), key=lambda index:
                         abs(sum(windows[index]) / 2 - center)) for center in centers})
    return [{"index": index, "start": windows[index][0], "stop": windows[index][1]}
            for index in indices]


def select_workflows(initial, case, features):
    """Freeze all target and display choices before any local response."""
    policy = ReviewPolicy()
    windows = regular_windows(len(initial), 180, 90)
    h42 = raw_motion_evidence(case, features)
    h84 = raw_motion_evidence(case, features, 84)
    priorities42, targets42 = review_priorities(initial, features, h42, windows, policy)
    merged, _, neighbourhood84, _ = longer_conflict_scope(initial, features, h42, h84, policy)
    priorities84, targets84 = review_priorities(initial, features, merged, windows, policy)
    _, neighbourhood42 = conflict_neighbourhood(initial, features, h42, policy)
    choices = evidence_display_choices(initial, features, h42, targets84, policy)
    fixed = fixed_targets(windows, len(initial))
    selected = {"direct_visual": []}
    for stem, targets in (("fixed", fixed), ("adaptive42", targets42), ("adaptive84", targets84)):
        for arm in ("narrow_raw", "narrow_blocks"):
            name = f"{stem}_{arm}" + ("_budget2" if stem == "fixed" else "")
            selected[name] = [{**target, "arm": arm} for target in targets]
    selected["adaptive84_evidence_display"] = choices
    scope42, _, _ = target_scopes(neighbourhood42, features.observed, targets42)
    scope84, _, _ = target_scopes(neighbourhood84, features.observed, targets84)
    scopes = {name: scope42 if name.startswith("adaptive42") else scope84
              for name in selected if name.startswith("adaptive")}
    return selected, scopes, {"priorities42": priorities42, "priorities84": priorities84,
                              "display_choices": choices}


def unique_local_queries(selected):
    """Deduplicate identical target/display requests without losing method usage."""
    unique = {}
    for targets in selected.values():
        for target in targets:
            key = (target["index"], target["arm"])
            if key in unique:
                if (unique[key]["start"], unique[key]["stop"]) != (target["start"], target["stop"]):
                    raise ValueError("One target index refers to different calendar ranges")
            else:
                unique[key] = {key: target[key] for key in ("index", "start", "stop", "arm")}
    return [unique[key] for key in sorted(unique)]


def freeze(out):
    config = read_json(REGISTRATION)
    for name, expected in config["source_bindings"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Registered candidate dependency changed: {name}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(REGISTRATION, out / "registration.json")
    sources = [ROOT / "scripts" / name for name in SCRIPT_SOURCES]
    sources += list((ROOT / "src/gnss_sim").glob("landslide*.py"))
    sources += [ROOT / "src/gnss_sim/visual.py", ROOT / "src/gnss_sim/__init__.py"]
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    model_sources = {
        "stage-xgboost.json": BASE / "motion-probe-v2/stage-xgboost.json",
        "activity-xgboost.json": BASE / "motion-probe-v2/activity-xgboost.json",
        "original-selection.json": BASE / "motion-probe-v2/frozen-selection.json",
        "raw-selection.json": BASE / "raw-baselines-v1/frozen-selection.json",
        "activity-xgboost-raw-persistence.json": BASE / "raw-baselines-v1/activity-xgboost-raw-persistence.json",
        "long-selection.json": BASE / "longer-evidence-review-v1/frozen-selection.json",
        "activity-xgboost-84.json": BASE / "longer-evidence-review-v1/activity-xgboost-84.json",
        "matched-selection.json": BASE / "longer-evidence-review-v1/matched-support-control-v1/frozen-selection.json",
        "activity-xgboost-42-matched-support.json": BASE / "longer-evidence-review-v1/matched-support-control-v1/activity-xgboost-42-matched-support.json",
    }
    for name, source in model_sources.items():
        shutil.copyfile(source, out / name)
    bindings = {str(path): sha(path) for path in model_sources.values()}
    for name in ("motion-probe-v2/manifest.json", "raw-baselines-v1/manifest.json",
                 "longer-evidence-review-v1/freeze.json",
                 "longer-evidence-review-v1/matched-support-control-v1/manifest.json"):
        path = BASE / name
        bindings[str(path)] = sha(path)
    save_json(out / "training-bindings.json", bindings)
    config.update({
        "execution_details": {
            "fixed_targets": "Nearest regular180/90 windows to calendar thirds; same historical fixed control",
            "fixed_revision": "Whole target, without numeric conflict authority; raw/blocks are identical except display",
            "adaptive_revision": "42: observed conflicts +/-49 within selected targets; 84: original conflict49/new84 conflict91",
            "stage": "All eight visual activities use the SAME frozen train24 learned ASD head; not pure VLM stages",
            "schedule": "repeat then ascending case_id; unique target/display order permuted with seed20261011",
            "failures": "No retry, every failed target unknown inside its authority; retain missing masks",
            "scoring_access": "References saved in generation but not opened until all predictions frozen",
            "new_features_or_tuning": False,
        }, "maximum_unique_requests": 312,
    })
    save_json(out / "protocol.json", config)
    save_json(out / "pre-generation-freeze.json", {
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "frozen_not_generated"})
    print("Execution, generation and fitted models frozen before case0085-0096 exist", flush=True)


def verify(out, generated=False):
    for name, expected in read_json(out / "pre-generation-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Pre-generation freeze changed: {name}")
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError(f"Executing source changed: {name}")
    for name, expected in read_json(out / "training-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Bound training/model artifact changed")
    if generated:
        for name, expected in read_json(out / "data-freeze.json").items():
            if sha(out / name) != expected:
                raise ValueError(f"Generated input changed: {name}")


def load_model(out, name):
    model = XGBClassifier()
    model.load_model(out / name)
    return model


def numerical_predictions(out, records):
    """Apply the nine existing fitted competitors without opening new references."""
    features = {cid: pair[1] for cid, pair in records.items()}
    head = load_model(out, "stage-xgboost.json")
    methods = predict_baselines(features, list(records), read_json(out / "original-selection.json"),
                                load_model(out, "activity-xgboost.json"), head)
    methods.pop("ungated_rule")
    raw_choice = read_json(out / "raw-selection.json")
    long_choice = read_json(out / "long-selection.json")
    matched_choice = read_json(out / "matched-selection.json")
    raw_model = load_model(out, "activity-xgboost-raw-persistence.json")
    long_model = load_model(out, "activity-xgboost-84.json")
    matched_model = load_model(out, "activity-xgboost-42-matched-support.json")
    for cid, (case, feature) in records.items():
        h42, h84 = raw_motion_evidence(case, feature), raw_motion_evidence(case, feature, 84)
        matrix42 = np.column_stack((feature.matrix, h42["rate"], h42["signal_to_noise"], h42["direction"]))
        matrix84 = np.column_stack((matrix42, h84["rate"], h84["signal_to_noise"], h84["direction"]))
        support84 = feature.observed & (np.isfinite(feature.progressive_rate)
                                        | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))
        rule42, learned42 = raw_choice["raw_multiscale_rule"], raw_choice["xgboost_raw_persistence"]
        rule84, learned84 = long_choice["raw_rule_84"], long_choice["xgboost_84"]
        policy84 = ReviewPolicy(minimum_rate=rule84["rate"], minimum_signal_to_noise=rule84["snr"])
        positive84 = ((feature.progressive_rate >= rule84["rate"])
                      & (feature.signal_to_noise >= rule84["snr"]))
        positive84 |= scale_positive(h42, policy84) | scale_positive(h84, policy84)
        gates = {
            "raw_multiscale_rule": raw_gate(feature, h42, rule42["rate"], rule42["snr"]),
            "xgboost_raw_persistence": activity_from_mask(raw_model.predict_proba(matrix42)[:, 1]
                                                          >= learned42["threshold"], feature.supported),
            "raw_rule_84": activity_from_mask(positive84, support84),
            "xgboost_42_matched_support": activity_from_mask(matched_model.predict_proba(matrix42)[:, 1]
                                                              >= matched_choice["threshold"], support84),
            "xgboost_84": activity_from_mask(long_model.predict_proba(matrix84)[:, 1]
                                              >= learned84["threshold"], support84),
        }
        codes = head.predict(feature.matrix).astype(int) + 1
        for name, activity in gates.items():
            methods.setdefault(name, {})[cid] = learned_stages(codes, activity, feature.supported)
    return {name: {cid: value.to_dict() for cid, value in cases.items()} for name, cases in methods.items()}


def generate(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "frozen_not_generated":
        raise ValueError("Generate only a new frozen batch")
    save_json(out / "state.json", {"status": "generating"})
    config = read_json(out / "protocol.json")
    specification = config["generation"]
    request = LandslideRequest(seed=specification["seed"], count=96, days=specification["days"])
    rows, records = [], {}
    (out / "images").mkdir()
    for index in specification["indices_zero_based"]:
        case, labels, truth = controlled_case(request, index)
        folder = out / "inputs" / case.case_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        reference = out / "mechanism-reference" / case.case_id
        reference.mkdir(parents=True)
        save_json(reference / "truth.json", truth)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
        features = observation_features(case, motions)
        records[case.case_id] = (case, features)
        rows.extend(labels)
        render_raw(case, out / "images" / f"{case.case_id}-raw.png")
        # Pre-render every eligible view without selecting targets from labels.
        for number, (start, stop) in enumerate(regular_windows(len(case.dates), 180, 90)):
            for arm in ("narrow_raw", "narrow_blocks"):
                render_observation_view(case, features.noise, start, stop, 0, arm == "narrow_blocks",
                    out / "images" / f"{case.case_id}-window-{number:02d}-{arm}.png")
        print(f"Generated and rendered {case.case_id}", flush=True)
    if list(records) != specification["case_ids"]:
        raise ValueError("Generated population differs from registration")
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    started = time.monotonic()
    save_json(out / "numeric-predictions.json", numerical_predictions(out, records))
    save_json(out / "numeric-cost.json", {"all_nine_prediction_seconds": time.monotonic() - started,
        "scope": "Shared feature extraction excluded; common model load/PELT/rule/ML predictions included; no API"})
    save_json(out / "data-freeze.json", {str(path.relative_to(out)): sha(path)
        for path in out.rglob("*") if path.is_file()
        and (str(path.relative_to(out)).startswith(("inputs", "images", "mechanism-reference"))
             or path.name in ("reference-subset.csv", "numeric-predictions.json", "numeric-cost.json"))})
    verify(out, generated=True)
    save_json(out / "state.json", {"status": "prepared"})


def execute(out):
    verify(out, generated=True)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only once; no retries or overwrite")
    config = read_json(out / "protocol.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"]["name"], 312)
    head = load_model(out, "stage-xgboost.json")
    predictions = read_json(out / "numeric-predictions.json")
    selections, reviews, failures, usage = {}, {}, {}, {}
    order = np.random.default_rng(20261011)
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running"})
    for rep in range(2):
        for cid in config["generation"]["case_ids"]:
            case, features = load_case_features(out, cid)
            initial_id = f"{cid}-direct-r{rep}"
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200].")
            try:
                payload = transport.ask(initial_id, prompt, [out / "images" / f"{cid}-raw.png"])
                initial = parse_activity(payload, features.observed)
            except (ValueError, RuntimeError) as error:
                failures[initial_id] = str(error)
                initial = np.full(len(case.dates), -1, dtype=int)
            selected, scopes, diagnostics = select_workflows(initial, case, features)
            if set(selected) != set(config["visual_controls"]):
                raise ValueError("Visual workflows differ from registration")
            local_plan = unique_local_queries(selected)
            key = f"{cid}-r{rep}"
            selections[key] = {"methods": selected, **diagnostics,
                "initial": initial.tolist(), "unique_local_plan": local_plan,
                "scope_days": {name: np.flatnonzero(scope).tolist() for name, scope in scopes.items()}}
            # Decision persistence precedes all local responses, including failed ones.
            save_json(out / "selections.json", selections)
            local_outputs = {}
            for ordinal in order.permutation(len(local_plan)):
                target = local_plan[int(ordinal)]
                number, arm = target["index"], target["arm"]
                request_id = f"{cid}-window-{number:02d}-{arm}-r{rep}"
                try:
                    payload = transport.ask(request_id, display_prompt(target["start"], target["stop"]),
                        [out / "images" / f"{cid}-raw.png",
                         out / "images" / f"{cid}-window-{number:02d}-{arm}.png"])
                    activity = parse_window_activity(payload, features.observed, target["start"], target["stop"])
                    status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[request_id] = str(error)
                    activity = np.full(len(initial), -1, dtype=int)
                    status = "failed"
                row = {**target, "activity": activity.tolist(), "status": status, "request_id": request_id}
                local_outputs[number, arm] = row
                reviews[request_id] = row
                save_json(out / "reviews.json", reviews)
                save_json(out / "failures.json", failures)
            codes = head.predict(features.matrix).astype(int) + 1
            for name, targets in selected.items():
                values = [{**local_outputs[target["index"], target["arm"]],
                           "activity": np.array(local_outputs[target["index"], target["arm"]]["activity"])}
                          for target in targets]
                activity = combine_reviews(initial, values, features.observed).activity
                if name in scopes:
                    activity = scoped_activity(initial, activity, scopes[name], features.observed)
                method = f"{name}_r{rep}"
                prediction = learned_stages(codes, activity, features.supported)
                if name == "direct_visual" and initial_id in failures:
                    prediction.status = "failed"
                predictions.setdefault(method, {})[cid] = prediction.to_dict()
                usage.setdefault(method, []).extend([initial_id] + [local_outputs[target["index"], target["arm"]]["request_id"]
                                                                   for target in targets])
            save_json(out / "predictions.json", predictions)
            save_json(out / "method-request-usage.json", usage)
            print(f"{cid} r{rep}: {len(local_plan)} unique locals; "
                  f"{len(reviews) + (rep * 12 + int(cid[-4:]) - 84)} total requests", flush=True)
    verify(out, generated=True)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in (
        "predictions.json", "selections.json", "reviews.json", "failures.json", "method-request-usage.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "generate", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "generate": generate, "run": execute}[args.action](args.out)


if __name__ == "__main__":
    main()
