"""Develop one longer evidence scale using unchanged cached visual queries."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from probe_landslide_longer_motion_evidence import PROTOCOL, longer_conflict_scope, scale_positive
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, calibration_objective, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_prompt,
)
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity


def prepare(out):
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    source = base / "revision-scope-confirmation-v1"
    probe = base / "longer-evidence-probe-v2"
    training = base / "motion-probe-v2"
    config = read_json(source / "protocol.json")
    for file, expected in read_json(source / "freeze.json").items():
        if sha(source / file) != expected:
            raise ValueError("Frozen original input changed")
    for file, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / file) != expected:
            raise ValueError("Frozen original prediction changed")
    for file, expected in read_json(training / "manifest.json")["files"].items():
        if sha(training / file) != expected:
            raise ValueError("Common training evidence changed")
    for file, expected in read_json(probe / "manifest.json")["files"].items():
        if sha(probe / file) != expected:
            raise ValueError("Evidence probe changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    for path in (Path(__file__), ROOT / "scripts/probe_landslide_longer_motion_evidence.py",
                 ROOT / "scripts/probe_landslide_revision_scope.py",
                 ROOT / "src/gnss_sim/landslide_adaptive_review.py",
                 ROOT / "src/gnss_sim/landslide_visual.py"):
        shutil.copyfile(path, out / "source" / path.name)
    shutil.copyfile(source / "stage-xgboost.json", out / "stage-xgboost.json")
    shutil.copyfile(source / "reference-subset.csv", out / "reference-subset.csv")
    shutil.copyfile(source / "predictions.json", out / "original-predictions.json")
    selections = read_json(probe / "selections.json")[source.name]
    save_json(out / "selections.json", selections)
    images, cached = out / "images", out / "cached-responses"
    images.mkdir()
    cached.mkdir()
    new_queries, records, training_features = [], {}, {}
    for cid in config["case_ids"]:
        destination = out / "inputs" / cid
        destination.mkdir(parents=True)
        for file in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(source / "inputs" / cid / file, destination / file)
        records[cid] = load_case_features(out, cid)
        case, features = records[cid]
        long = raw_motion_evidence(case, features)
        shutil.copyfile(source / "images" / f"{cid}-raw.png", images / f"{cid}-raw.png")
        for rep in range(config["repetitions"]):
            for target in selections[f"{cid}-r{rep}"]["selected"]:
                index = target["index"]
                request_id = f"{cid}-window-{index:02d}-image-r{rep}"
                image_name = f"{cid}-window-{index:02d}.png"
                if not (images / image_name).exists():
                    shutil.copyfile(source / "images" / image_name, images / image_name)
                prior = source / "requests" / f"{request_id}.response.json"
                prompt = review_prompt(features, long, target["start"], target["stop"], False)
                if prior.exists():
                    receipt = read_json(prior)
                    if receipt["prompt"] != prompt:
                        raise ValueError("Cached local prompt differs")
                    for file, expected in receipt["image_sha256"].items():
                        if sha(images / file) != expected:
                            raise ValueError("Cached local image differs")
                    shutil.copyfile(prior, cached / prior.name)
                else:
                    new_queries.append({"request_id": request_id, "prompt": prompt,
                                        "images": [f"{cid}-raw.png", image_name]})
    save_json(out / "query-plan.json", new_queries)

    # Give both numerical controls the exact added 84-day rate/noise/direction.
    # This is development comparison; original models are preserved separately.
    split = read_json(training / "protocol.json")["split"]
    train_validation = split["train"] + split["validation"]
    for cid in train_validation:
        training_features[cid] = load_case_features(training, cid)
    with (training / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = [row for row in reader if row["case_id"] in training_features]
    with (out / "training-reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    train_labels = load_labels(out / "training-reference.csv",
                              {cid: record[0] for cid, record in training_features.items()})
    all_records = {**training_features, **records}
    matrices, evidence, activity_supported = {}, {}, {}
    for cid, (case, features) in all_records.items():
        h42 = raw_motion_evidence(case, features)
        h84 = raw_motion_evidence(case, features, 84)
        evidence[cid] = (h42, h84)
        activity_supported[cid] = features.observed & (np.isfinite(features.progressive_rate)
                                    | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))
        matrices[cid] = np.column_stack((features.matrix, h42["rate"], h42["signal_to_noise"],
                                        h42["direction"], h84["rate"], h84["signal_to_noise"],
                                        h84["direction"]))
    stage_head = XGBClassifier()
    stage_head.load_model(out / "stage-xgboost.json")
    codes = {cid: stage_head.predict(features.matrix).astype(int) + 1
             for cid, (_, features) in all_records.items()}

    def rule_gate(cid, rate, snr):
        _, features = all_records[cid]
        choice = ReviewPolicy(minimum_rate=rate, minimum_signal_to_noise=snr)
        h42, h84 = evidence[cid]
        positive = ((features.progressive_rate >= rate) & (features.signal_to_noise >= snr))
        positive |= scale_positive(h42, choice) | scale_positive(h84, choice)
        return activity_from_mask(positive, activity_supported[cid])

    def validation_score(gates):
        scores = []
        for cid, gate in gates.items():
            feature = training_features[cid][1]
            prediction = learned_stages(codes[cid], gate, feature.supported)
            scores.append(evaluate_case(train_labels[cid], prediction))
        return aggregate_scores(scores)

    trials = []
    for rate in (.03, .06, .12):
        for snr in (1., 2., 3.):
            score = validation_score({cid: rule_gate(cid, rate, snr) for cid in split["validation"]})
            trials.append({"method": "raw_rule_84", "rate": rate, "snr": snr,
                           "objective": calibration_objective(score), "validation": score})
    train_x = np.concatenate([matrices[cid] for cid in split["train"]])
    train_y = np.array([int(row["activity_label"]) for cid in split["train"] for row in train_labels[cid]])
    eligible = np.concatenate([activity_supported[cid] for cid in split["train"]])
    eligible &= train_y >= 0
    models = {}
    for depth in (2, 4):
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                              n_jobs=2, random_state=20261007)
        model.fit(train_x[eligible], train_y[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]}
        for threshold in (.35, .5, .65):
            score = validation_score({cid: activity_from_mask(probabilities[cid] >= threshold,
                                      activity_supported[cid]) for cid in split["validation"]})
            trials.append({"method": "xgboost_84", "depth": depth, "threshold": threshold,
                           "objective": calibration_objective(score), "validation": score})
    choices = {method: max((row for row in trials if row["method"] == method), key=lambda row: row["objective"])
               for method in ("raw_rule_84", "xgboost_84")}
    save_json(out / "validation-trials.json", trials)
    save_json(out / "frozen-selection.json", choices)
    model = models[choices["xgboost_84"]["depth"]]
    model.save_model(out / "activity-xgboost-84.json")
    numeric = {method: {} for method in choices}
    for cid, (_, features) in records.items():
        rule, learned = choices["raw_rule_84"], choices["xgboost_84"]
        gates = {"raw_rule_84": rule_gate(cid, rule["rate"], rule["snr"]),
                 "xgboost_84": activity_from_mask(model.predict_proba(matrices[cid])[:, 1]
                                                 >= learned["threshold"], activity_supported[cid])}
        for method, gate in gates.items():
            numeric[method][cid] = learned_stages(codes[cid], gate, features.supported).to_dict()
    save_json(out / "numeric-predictions.json", numeric)
    config.update({"source_bundle": str(source), "scope": "Viewed twelve-record development "
                   "mechanism comparison, not new confirmation", "candidate_policy": PROTOCOL,
                   "primary_candidate": "extended84_evidence_scope", "maximum_requests": len(new_queries),
                   "comparison": "Current base42 scoped candidate, extended selection with original "
                                 "49-day scope, physical 49/91 scope and whole-target control",
                   "stop": "Do not retain scale extension without better final weak recall and "
                           "daily F1 in both repetitions relative to base42 scoped method; disclose "
                           "all TP loss and false reviews, no threshold rescue",
                   "numeric_grids": "Same original 9-rule and 6-XGBoost validation opportunities",
                   "numeric_activity_support": "Observed days with short/42/84 raw evidence; "
                                               "same union for retrained rule and activity XGBoost, "
                                               "common unchanged stage support retained"})
    save_json(out / "protocol.json", config)
    save_json(out / "source-bindings.json", {str(path): sha(path) for path in (
        source / "freeze.json", source / "prediction-freeze.json", probe / "manifest.json",
        training / "manifest.json", training / "stage-xgboost.json",
    )})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print(f"Longer-scale development prepared: {len(new_queries)} new image calls, same train/validation", flush=True)


def verify(out):
    for file, expected in read_json(out / "freeze.json").items():
        if sha(out / file) != expected:
            raise ValueError("Longer-scale frozen input changed")
    for file, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(file)) != expected:
            raise ValueError("Longer-scale bound comparison changed")
    for name in ("run_landslide_longer_motion_review.py", "probe_landslide_longer_motion_evidence.py",
                 "probe_landslide_revision_scope.py", "landslide_adaptive_review.py", "landslide_visual.py"):
        parent = "scripts" if name.startswith(("run_", "probe_")) else "src/gnss_sim"
        if sha(ROOT / parent / name) != sha(out / "source" / name):
            raise ValueError("Executing longer-scale method differs from snapshot")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only new prepared inference; no retry or overwrite")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    save_json(out / "state.json", {"status": "running"})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], len(plan))
    failures = {}
    for query in plan:
        try:
            transport.ask(query["request_id"], query["prompt"], [out / "images" / name for name in query["images"]])
        except (ValueError, RuntimeError) as error:
            failures[query["request_id"]] = str(error)
        save_json(out / "failures.json", failures)
    policy = ReviewPolicy(**PROTOCOL["policy"])
    initial_predictions = read_json(out / "original-predictions.json")
    selections = read_json(out / "selections.json")
    stage_head = XGBClassifier()
    stage_head.load_model(out / "stage-xgboost.json")
    predictions = read_json(out / "numeric-predictions.json")
    diagnostics = {}
    for cid in config["case_ids"]:
        case, features = load_case_features(out, cid)
        h42, h84 = raw_motion_evidence(case, features), raw_motion_evidence(case, features, 84)
        codes = stage_head.predict(features.matrix).astype(int) + 1
        for rep in range(config["repetitions"]):
            key = f"{cid}-r{rep}"
            initial = np.array(initial_predictions[f"direct_visual_r{rep}"][cid]["activity"])
            merged, conflict, physical_scope, additional = longer_conflict_scope(initial, features, h42, h84, policy)
            _, same49 = conflict_neighbourhood(initial, features, merged, policy)
            selected = selections[key]["selected"]
            physical, _, _ = target_scopes(physical_scope, features.observed, selected)
            scope49, _, _ = target_scopes(same49, features.observed, selected)
            reviews = []
            for target in selected:
                request_id = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                path = out / "cached-responses" / f"{request_id}.response.json"
                if not path.exists():
                    path = out / "requests" / path.name
                receipt = read_json(path)
                try:
                    if receipt["status"] != "returned_json":
                        raise ValueError("Request failed")
                    activity = parse_window_activity(json.loads(receipt["output"]), features.observed,
                                                     target["start"], target["stop"])
                    status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[request_id] = str(error)
                    activity = np.full(len(initial), -1)
                    status = "failed"
                reviews.append({**target, "activity": activity, "status": status})
            whole = combine_reviews(initial, reviews, features.observed).activity
            for method, activity in (("extended84_whole", whole),
                                     ("extended84_same49", scoped_activity(initial, whole, scope49, features.observed)),
                                     ("extended84_evidence_scope", scoped_activity(initial, whole, physical, features.observed))):
                predictions.setdefault(f"{method}_r{rep}", {})[cid] = learned_stages(
                    codes, activity, features.supported).to_dict()
            diagnostics[key] = {"additional_conflict_days": int(additional.sum()),
                                "conflict_days": int(conflict.sum()), "review_targets": selected}
    save_json(out / "predictions.json", predictions)
    save_json(out / "diagnostics.json", diagnostics)
    save_json(out / "failures.json", failures)
    verify(out)
    save_json(out / "prediction-freeze.json", {file: sha(out / file) for file in (
        "predictions.json", "diagnostics.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete"})
    print("Longer-scale development inference frozen", flush=True)


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
