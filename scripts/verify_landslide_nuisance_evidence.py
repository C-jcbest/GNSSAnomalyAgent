"""Replay frozen nuisance-pilot predictions and scoring without retuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_nuisance_evidence import (
    ARMS,
    DEVELOPMENT,
    OLD,
    ROOT,
    TRAINING,
    evidence_matrix,
    predict,
)
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_nuisance_evidence import record_nuisance_evidence


def verify(run, out):
    bindings = read_json(run / "source-bindings.json")
    manifest = read_json(run / "delivery-manifest.json")
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source/input changed: " + relative)
    for relative, expected in manifest.items():
        if sha(run / relative) != expected:
            raise ValueError("Frozen artifact changed: " + relative)
    protocol = read_json(run / "protocol.json")
    selection = read_json(run / "selection.json")
    saved = read_json(run / "new-predictions.json")
    probabilities = read_json(run / "probabilities.json")
    historical = read_json(OLD / "predictions.json")
    head = XGBClassifier()
    head.load_model(TRAINING / "stage-xgboost.json")
    record_features = {
        cid: load_case_features(DEVELOPMENT, cid)
        for cid in protocol["split"]["development"]
    }
    checked_days = 0
    for arm in ARMS:
        model = XGBClassifier()
        model.load_model(run / (arm + "-model.json"))
        for cid, (_, feature) in record_features.items():
            with np.load(run / (cid + "-features.npz")) as bank:
                matrix = evidence_matrix(bank["raw"], bank["evidence"], arm)
                probability = model.predict_proba(matrix)[:, 1]
                codes = head.predict(feature.matrix).astype(int) + 1
                replayed = predict(
                    probability, bank["support"], codes,
                    feature.supported, selection[arm]["threshold"],
                ).to_dict()
            if not np.array_equal(probability, probabilities[arm][cid]):
                raise ValueError("Probability replay differs")
            if replayed != saved[arm][cid]:
                raise ValueError("Activity/stage replay differs")
            if arm == "numeric_raw84" and replayed != historical["xgboost_84"][cid]:
                raise ValueError("Retrained raw84 no longer matches the original strong baseline")
            checked_days += len(probability)
    evidence_checks = []
    for group, ids in protocol["split"].items():
        source = DEVELOPMENT if group == "development" else TRAINING
        for cid in ids[:2]:
            case, feature = load_case_features(source, cid)
            values = np.array([
                row if row is not None else [np.nan] * 3
                for row in case.displacement_mm
            ])
            evidence, _ = record_nuisance_evidence(values, feature.noise)
            with np.load(run / (cid + "-features.npz")) as bank:
                if not np.array_equal(evidence, bank["evidence"], equal_nan=True):
                    raise ValueError("Raw evidence replay differs")
            evidence_checks.append(cid)
    labels = read_json(OLD / "reference-labels.json")
    observed = {cid: feature.observed for cid, (_, feature) in record_features.items()}
    weak = {
        cid: np.array([bool(int(row.get("weak_signal", 0))) for row in rows])
        for cid, rows in labels.items()
    }
    all_predictions = {**saved, **{"historical__" + name: value for name, value in historical.items()}}
    results, _, _ = score_target(labels, observed, weak, all_predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Full common scoring replay differs")
    write_json(out, {
        "status": "VERIFIED", "bound_source_files": len(bindings),
        "frozen_artifact_files": len(manifest), "model_day_predictions_replayed": checked_days,
        "evidence_records_replayed": evidence_checks,
        "original_raw84_prediction_identity": True,
        "all_method_score_identity": True,
        "new_training": False, "new_api_calls": 0,
    })
    print("Verified: predictions, original raw84 identity, six evidence records and all scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.out)


if __name__ == "__main__":
    main()
