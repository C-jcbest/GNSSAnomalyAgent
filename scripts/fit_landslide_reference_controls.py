"""Fit same-information annual-reference numerical controls on the shared split."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from run_landslide_comparison import calibration_objective, save_json, sha
from run_landslide_reference_observation_probe_v2 import BASE, REFERENCE
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_seasonal_reference import apply_annual_reference, observed_values

SOURCE = BASE / "background-benchmark-data-v1"
DEVELOPMENT_IDS = [f"case_{number}" for number in range(2025, 2037)]


def reference_information(case, feature, reference):
    """Expose raw84 features plus the same reference/residual used in the images."""
    adjusted, curve = apply_annual_reference(case, reference)
    residual, _ = observed_values(adjusted)
    original = {scale: raw_motion_evidence(case, feature, scale) for scale in (42, 84)}
    corrected = {scale: raw_motion_evidence(adjusted, feature, scale) for scale in (21, 42, 84)}
    support = feature.observed & (np.isfinite(feature.progressive_rate)
                                 | np.isfinite(original[42]["rate"]) | np.isfinite(original[84]["rate"]))
    raw84 = np.column_stack((feature.matrix, *[original[scale][key] for scale in (42, 84)
                                              for key in ("rate", "signal_to_noise", "direction")]))
    augmented = np.column_stack((raw84, curve, residual, *[corrected[scale][key] for scale in (21, 42, 84)
                                                        for key in ("rate", "signal_to_noise", "direction")]))
    return augmented, corrected, support


def residual_rule(evidence, support, rate, snr):
    positive = np.zeros(len(support), dtype=bool)
    for row in evidence.values():
        positive |= (row["rate"] >= rate) & (row["signal_to_noise"] >= snr)
    return activity_from_mask(positive, support)


def fit(out):
    verify_data(SOURCE)
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(SOURCE / "protocol.json")["split"]
    protocol = {"scope": "Same-information controls for previously viewed full12-record development; not new confirmation",
        "train": split["train"], "validation": split["validation"], "development": DEVELOPMENT_IDS,
        "annual_reference": "Same frozen train-only curve as VLM, offline grouped training not past-only forecasting",
        "raw_stage_head": "Original background-train stage model and support, unchanged for every activity gate",
        "rule_grid": {"rate": [.03, .06, .12], "snr": [1., 2., 3.]},
        "xgboost_grid": {"depth": [2, 4], "threshold": [.35, .5, .65]},
        "objective": "Validation daily F1 + event F1; same opportunities as original strong controls",
        "xgboost": "Raw84 matrix plus frozen reference NEU, residual NEU, residual21/42/84 persistence;120 estimators,lr.05,seed20261007",
        "support": "Identical observed original21/42/84 support; no narrower corrected eligibility or changed noise proxy",
        "selection": "Only train/validation labels accessed; no development labels or prediction-led tuning"}
    save_json(out / "protocol.json", protocol)
    shutil.copyfile(REFERENCE / "annual-reference.json", out / "annual-reference.json")
    shutil.copyfile(SOURCE / "stage-xgboost.json", out / "stage-xgboost.json")
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "pre-fit-freeze.json", {str(path): sha(path) for path in
        (Path(__file__), SOURCE / "manifest.json", REFERENCE / "annual-reference.json", SOURCE / "stage-xgboost.json")})
    reference = read_json(out / "annual-reference.json")
    ids = split["train"] + split["validation"] + DEVELOPMENT_IDS
    records = {cid: load_case_features(SOURCE, cid) for cid in ids}
    information = {cid: reference_information(case, feature, reference) for cid, (case, feature) in records.items()}
    labels = {cid: [] for cid in split["train"] + split["validation"]}
    with (SOURCE / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    codes = {cid: head.predict(feature.matrix).astype(int) + 1 for cid, (_, feature) in records.items()}

    def score(gates):
        return aggregate_scores([evaluate_case(labels[cid], learned_stages(codes[cid], gate, records[cid][1].supported))
                                 for cid, gate in gates.items()])

    trials = []
    for rate in protocol["rule_grid"]["rate"]:
        for snr in protocol["rule_grid"]["snr"]:
            gates = {cid: residual_rule(information[cid][1], information[cid][2], rate, snr) for cid in split["validation"]}
            measured = score(gates)
            trials.append({"method": "annual_residual_rule", "rate": rate, "snr": snr,
                           "objective": calibration_objective(measured), "validation": measured})
    train_x = np.concatenate([information[cid][0] for cid in split["train"]])
    train_y = np.array([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    eligible = np.concatenate([information[cid][2] for cid in split["train"]]) & (train_y >= 0)
    models = {}
    for depth in protocol["xgboost_grid"]["depth"]:
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05, n_jobs=2, random_state=20261007)
        model.fit(train_x[eligible], train_y[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(information[cid][0])[:, 1] for cid in split["validation"]}
        for threshold in protocol["xgboost_grid"]["threshold"]:
            gates = {cid: activity_from_mask(probabilities[cid] >= threshold, information[cid][2])
                     for cid in split["validation"]}
            measured = score(gates)
            trials.append({"method": "annual_augmented_xgboost", "depth": depth, "threshold": threshold,
                           "objective": calibration_objective(measured), "validation": measured})
    choices = {method: max((row for row in trials if row["method"] == method), key=lambda row: row["objective"])
               for method in ("annual_residual_rule", "annual_augmented_xgboost")}
    selected = models[choices["annual_augmented_xgboost"]["depth"]]
    selected.save_model(out / "annual-augmented-xgboost.json")
    save_json(out / "validation-trials.json", trials)
    save_json(out / "frozen-selection.json", choices)
    save_json(out / "fitting-freeze.json", {name: sha(out / name) for name in
              ("annual-augmented-xgboost.json", "annual-reference.json", "stage-xgboost.json", "frozen-selection.json")})
    predictions = {name: {} for name in choices}
    for cid in DEVELOPMENT_IDS:
        matrix, evidence, support = information[cid]
        rule, learned = choices["annual_residual_rule"], choices["annual_augmented_xgboost"]
        gates = {"annual_residual_rule": residual_rule(evidence, support, rule["rate"], rule["snr"]),
                 "annual_augmented_xgboost": activity_from_mask(selected.predict_proba(matrix)[:, 1] >= learned["threshold"], support)}
        for method, gate in gates.items():
            predictions[method][cid] = learned_stages(codes[cid], gate, records[cid][1].supported).to_dict()
    save_json(out / "predictions.json", predictions)
    save_json(out / "manifest.json", {"new_api_requests": 0, "train_eligible_days": int(eligible.sum()),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Same-information annual controls frozen: rule9 / XGBoost6 validation trials, no development scoring", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    fit(parser.parse_args().out)


if __name__ == "__main__":
    main()
