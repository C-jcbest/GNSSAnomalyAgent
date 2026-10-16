"""Fit causal-prefix direction-informed controls on the common background train/validation split."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from fit_landslide_raw_persistence_baselines import raw_gate
from run_landslide_comparison import ROOT, calibration_objective, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_causal_direction_review import causal_direction_anchors
from gnss_sim.landslide_direction_review import (
    direction_information,
    raw_block_changes,
)
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages


def direction_features(case, feature, calibration):
    values, before, after = raw_block_changes(case)
    anchors = causal_direction_anchors(before, after, feature.observed, feature.noise,
                                       calibration["rate"], calibration["snr"])
    directional = direction_information(case, feature, anchors, before, after)
    evidence = {scale: raw_motion_evidence(case, feature, scale) for scale in (42, 84)}
    matrix = np.column_stack((feature.matrix,
                             *[evidence[scale][key] for scale in (42, 84)
                               for key in ("rate", "signal_to_noise", "direction")],
                             before, after, directional))
    support = feature.observed & (np.isfinite(feature.progressive_rate)
                                  | np.isfinite(evidence[42]["rate"]) | np.isfinite(evidence[84]["rate"]))
    return matrix, support, directional, anchors, values, evidence[42]


def direction_rule(feature, long_evidence, directional, calibration, rate, snr):
    strong = raw_gate(feature, long_evidence, calibration["rate"], calibration["snr"]) == 1
    weak = ((directional[:, :2] >= .5).all(axis=1)
            & (directional[:, 2:4] >= rate).all(axis=1)
            & (directional[:, 4:6] >= snr).all(axis=1))
    supported = feature.observed & (np.isfinite(feature.progressive_rate) | np.isfinite(long_evidence["rate"]))
    return activity_from_mask(strong | weak, supported)


def fit(source, out):
    verify_data(source)
    split = read_json(source / "manifest.json")["split"]
    calibration = read_json(source / "long-selection.json")["raw_multiscale_rule"]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "source": str(source.resolve()), "split": split, "scope": "Exposed development, not independent",
        "features": "Original raw21/42/84 matrix plus raw42 NEU changes, preceding strong anchor "
                    "direction, cosines, projected rates/noise, availability and age. No VLM outputs.",
        "anchor": "Existing calibrated raw42 .12/3 support, >=14days; most recent14 supported centers; all observations precede candidate, no episode-end wait",
        "rule_grid": {"rate": [.03, .06, .12], "snr": [1., 2., 3.]},
        "xgboost_grid": {"depth": [2, 4], "threshold": [.35, .5, .65]},
        "objective": "Original common validation dailyF1 + eventF1; no exposed test labels for selection",
        "stage": "Same frozen background-train24 stage head", "new_api_requests": 0,
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    bindings = [Path(__file__), ROOT / "src/gnss_sim/landslide_direction_review.py",
                source / "manifest.json", source / "reference-labels.csv", source / "long-selection.json",
                source / "stage-xgboost.json", ROOT / "src/gnss_sim/landslide_causal_direction_review.py"]
    save_json(out / "pre-fit-freeze.json", {str(path.resolve()): sha(path) for path in bindings})
    ids = split["train"] + split["validation"] + split["mechanism_test"]
    records = {cid: load_case_features(source, cid) for cid in ids}
    matrices = {cid: direction_features(case, feature, calibration) for cid, (case, feature) in records.items()}
    labels = {cid: [] for cid in split["train"] + split["validation"]}
    with (source / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    head = XGBClassifier()
    head.load_model(source / "stage-xgboost.json")
    codes = {cid: head.predict(feature.matrix).astype(int) + 1 for cid, (_, feature) in records.items()}
    rule_trials = []
    for rate in (.03, .06, .12):
        for snr in (1., 2., 3.):
            scores = []
            for cid in split["validation"]:
                feature = records[cid][1]
                gate = direction_rule(feature, matrices[cid][5], matrices[cid][2], calibration, rate, snr)
                scores.append(evaluate_case(labels[cid], learned_stages(codes[cid], gate, feature.supported)))
            measured = aggregate_scores(scores)
            rule_trials.append({"rate": rate, "snr": snr, "objective": calibration_objective(measured),
                                "validation": measured})
    selected_rule = max(rule_trials, key=lambda row: row["objective"])
    train_x = np.concatenate([matrices[cid][0] for cid in split["train"]])
    train_y = np.asarray([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    eligible = np.concatenate([matrices[cid][1] for cid in split["train"]]) & (train_y >= 0)
    models, trials = {}, []
    for depth in (2, 4):
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                              n_jobs=2, random_state=20261007)
        model.fit(train_x[eligible], train_y[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(matrices[cid][0])[:, 1] for cid in split["validation"]}
        for threshold in (.35, .5, .65):
            scores = []
            for cid in split["validation"]:
                gate = activity_from_mask(probabilities[cid] >= threshold, matrices[cid][1])
                scores.append(evaluate_case(labels[cid], learned_stages(codes[cid], gate, records[cid][1].supported)))
            measured = aggregate_scores(scores)
            trials.append({"depth": depth, "threshold": threshold,
                           "objective": calibration_objective(measured), "validation": measured})
    selected = max(trials, key=lambda row: row["objective"])
    model = models[selected["depth"]]
    model.save_model(out / "direction-xgboost.json")
    save_json(out / "validation-trials.json", {"rule": rule_trials, "xgboost": trials})
    save_json(out / "frozen-selection.json", {"rule": selected_rule, "xgboost": selected})
    save_json(out / "fitting-freeze.json", {name: sha(out / name) for name in
                                          ("direction-xgboost.json", "frozen-selection.json")})
    predictions = {"direction_rule": {}, "xgboost_direction_information": {}}
    for cid in split["mechanism_test"]:
        feature = records[cid][1]
        gate = direction_rule(feature, matrices[cid][5], matrices[cid][2], calibration,
                              selected_rule["rate"], selected_rule["snr"])
        predictions["direction_rule"][cid] = learned_stages(codes[cid], gate, feature.supported).to_dict()
        gate = activity_from_mask(model.predict_proba(matrices[cid][0])[:, 1] >= selected["threshold"], matrices[cid][1])
        predictions["xgboost_direction_information"][cid] = learned_stages(codes[cid], gate, feature.supported).to_dict()
    save_json(out / "predictions.json", predictions)
    save_json(out / "manifest.json", {"status": "predictions_frozen_not_test_scored",
                                    "train_eligible_days": int(eligible.sum()), "validation_trials": 15,
                                    "prediction_sha256": sha(out / "predictions.json"), "new_api_requests": 0})
    print({"status": "frozen", "rule": [selected_rule["rate"], selected_rule["snr"]],
           "xgboost": [selected["depth"], selected["threshold"]]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.source, args.out)


if __name__ == "__main__":
    main()
