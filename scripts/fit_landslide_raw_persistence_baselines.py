"""Give numerical competitors the same long raw persistence used by the Agent."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from run_landslide_comparison import calibration_objective, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages


def raw_gate(features, long_evidence, rate, snr):
    short_supported = np.isfinite(features.progressive_rate)
    long_supported = np.isfinite(long_evidence["rate"])
    short = (features.progressive_rate >= rate) & (features.signal_to_noise >= snr)
    long = (long_evidence["rate"] >= rate) & (long_evidence["signal_to_noise"] >= snr)
    supported = features.observed & (short_supported | long_supported)
    return activity_from_mask((short_supported & short) | (long_supported & long), supported)


def fit(source, out):
    out.mkdir(parents=True, exist_ok=False)
    source_manifest = read_json(source / "manifest.json")
    split = source_manifest["split"]
    protocol = {"date": "2026-10-09", "scope": "Additional strong numerical controls, "
                "registered while Agent inference is in progress and before test scoring",
                "source_manifest_sha256": sha(source / "manifest.json"),
                "training_and_validation": "Same grouped records as every primary baseline",
                "rule_grid": {"rate": [.03, .06, .12], "snr": [1., 2., 3.]},
                "xgboost_grid": {"depth": [2, 4], "threshold": [.35, .5, .65]},
                "selection_objective": "Validation activity daily F1 + event F1",
                "test_tuning": "Forbidden; same stage head retained",
                "information": "Rule sees short OR long raw persistence; XGBoost sees "
                               "all original features plus long rate, noise ratio, direction"}
    save_json(out / "protocol.json", protocol)
    started = time.monotonic()
    features, evidence, matrices = {}, {}, {}
    for cid in split["train"] + split["validation"] + split["mechanism_test"]:
        case, feature = load_case_features(source, cid)
        features[cid] = feature
        long_evidence = raw_motion_evidence(case, feature)
        evidence[cid] = long_evidence
        matrices[cid] = np.column_stack((feature.matrix, long_evidence["rate"],
                                        long_evidence["signal_to_noise"],
                                        long_evidence["direction"]))
    labels = {cid: [] for cid in split["train"] + split["validation"]}
    with (source / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    trials = []
    stage_model = XGBClassifier()
    stage_model.load_model(source / "stage-xgboost.json")
    validation = split["validation"]

    def validation_score(gates):
        scores = []
        for cid, gate in gates.items():
            codes = stage_model.predict(features[cid].matrix).astype(int) + 1
            prediction = learned_stages(codes, gate, features[cid].supported)
            scores.append(evaluate_case(labels[cid], prediction))
        return aggregate_scores(scores)

    for rate in protocol["rule_grid"]["rate"]:
        for snr in protocol["rule_grid"]["snr"]:
            gates = {cid: raw_gate(features[cid], evidence[cid], rate, snr) for cid in validation}
            score = validation_score(gates)
            trials.append({"method": "raw_multiscale_rule", "rate": rate, "snr": snr,
                           "objective": calibration_objective(score), "validation": score})
    training_matrix = np.concatenate([matrices[cid] for cid in split["train"]])
    training_labels = np.array([int(row["activity_label"]) for cid in split["train"]
                                for row in labels[cid]])
    eligible = np.concatenate([features[cid].supported for cid in split["train"]])
    eligible &= training_labels >= 0
    models = {}
    for depth in protocol["xgboost_grid"]["depth"]:
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                              n_jobs=2, random_state=20261007)
        model.fit(training_matrix[eligible], training_labels[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in validation}
        for threshold in protocol["xgboost_grid"]["threshold"]:
            gates = {cid: activity_from_mask(probabilities[cid] >= threshold,
                                             features[cid].supported) for cid in validation}
            score = validation_score(gates)
            trials.append({"method": "xgboost_raw_persistence", "depth": depth,
                           "threshold": threshold, "objective": calibration_objective(score),
                           "validation": score})
    choices = {method: max((row for row in trials if row["method"] == method),
                           key=lambda row: row["objective"])
               for method in ("raw_multiscale_rule", "xgboost_raw_persistence")}
    save_json(out / "validation-trials.json", trials)
    save_json(out / "frozen-selection.json", choices)
    model = models[choices["xgboost_raw_persistence"]["depth"]]
    model.save_model(out / "activity-xgboost-raw-persistence.json")
    predictions = {name: {} for name in choices}
    for cid in split["mechanism_test"]:
        codes = stage_model.predict(features[cid].matrix).astype(int) + 1
        rule = choices["raw_multiscale_rule"]
        learned = choices["xgboost_raw_persistence"]
        gates = {"raw_multiscale_rule": raw_gate(features[cid], evidence[cid], rule["rate"], rule["snr"]),
                 "xgboost_raw_persistence": activity_from_mask(
                     model.predict_proba(matrices[cid])[:, 1] >= learned["threshold"],
                     features[cid].supported)}
        for method, gate in gates.items():
            predictions[method][cid] = learned_stages(codes, gate, features[cid].supported).to_dict()
    save_json(out / "predictions.json", predictions)
    save_json(out / "manifest.json", {"status": "predictions_frozen_not_test_scored",
                                      "seconds": time.monotonic() - started,
                                      "prediction_sha256": sha(out / "predictions.json"),
                                      "source_sha256": sha(Path(__file__))})
    print("Additional numerical baselines fitted and predictions frozen", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.source, args.out)


if __name__ == "__main__":
    main()
