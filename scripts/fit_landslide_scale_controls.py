"""Give a common train/validation XGBoost control the history-compression information."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from run_landslide_comparison import calibration_objective, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_scale_review import historical_scale_gains


def scale_information(case, features):
    """Provide original raw84 features and every candidate window's scale gains."""
    days = len(case.dates)
    windows = regular_windows(days, 180, 90)
    gains = historical_scale_gains(case, features.noise, windows)
    daily_gain = np.zeros((days, 3))
    for (start, stop), row in zip(windows, gains):
        daily_gain[start:stop] = np.maximum(daily_gain[start:stop], row["axis_gains"])
    raw = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    global_range = np.ptp(raw[features.observed], axis=0)
    evidence = {scale: raw_motion_evidence(case, features, scale) for scale in (42, 84)}
    matrix = np.column_stack((features.matrix,
                              *[evidence[scale][key] for scale in (42, 84)
                                for key in ("rate", "signal_to_noise", "direction")],
                              daily_gain, np.tile(global_range, (days, 1))))
    support = features.observed & (np.isfinite(features.progressive_rate)
                                  | np.isfinite(evidence[42]["rate"]) | np.isfinite(evidence[84]["rate"]))
    return matrix, support


def fit(source, out):
    verify_data(source)
    split = read_json(source / "manifest.json")["split"]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "scope": "Same-information numerical control on exposed background development; not independent",
        "source": str(source.resolve()), "split": split,
        "features": "Original raw21/42/84 matrix plus each axis maximum gain among covering180/90 windows "
                    "and global raw range; all candidate windows available, no VLM outputs or references",
        "support": "Same original observed21/42/84 support; same background-trained stage head",
        "grid": {"depth": [2, 4], "threshold": [.35, .5, .65]},
        "training": {"estimators": 120, "learning_rate": .05, "seed": 20261007, "n_jobs": 2},
        "objective": "Same validation activity daily F1 + event F1 as original strong controls",
        "no_test_labels": True, "no_extra_grid": True, "new_api_requests": 0,
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "pre-fit-freeze.json", {str(path.resolve()): sha(path) for path in (
        Path(__file__), Path(__file__).parents[1] / "src/gnss_sim/landslide_scale_review.py",
        source / "manifest.json", source / "stage-xgboost.json", source / "reference-labels.csv")})
    ids = split["train"] + split["validation"] + split["mechanism_test"]
    records = {cid: load_case_features(source, cid) for cid in ids}
    matrices = {cid: scale_information(case, feature) for cid, (case, feature) in records.items()}
    labels = {cid: [] for cid in split["train"] + split["validation"]}
    with (source / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    train_x = np.concatenate([matrices[cid][0] for cid in split["train"]])
    train_y = np.asarray([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    eligible = np.concatenate([matrices[cid][1] for cid in split["train"]]) & (train_y >= 0)
    head = XGBClassifier()
    head.load_model(source / "stage-xgboost.json")
    codes = {cid: head.predict(feature.matrix).astype(int) + 1 for cid, (_, feature) in records.items()}
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
                prediction = learned_stages(codes[cid], gate, records[cid][1].supported)
                scores.append(evaluate_case(labels[cid], prediction))
            measured = aggregate_scores(scores)
            trials.append({"depth": depth, "threshold": threshold,
                           "objective": calibration_objective(measured), "validation": measured})
    selected = max(trials, key=lambda row: row["objective"])
    model = models[selected["depth"]]
    model.save_model(out / "scale-xgboost.json")
    save_json(out / "validation-trials.json", trials)
    save_json(out / "frozen-selection.json", selected)
    save_json(out / "fitting-freeze.json", {name: sha(out / name) for name in
                                           ("scale-xgboost.json", "frozen-selection.json")})
    predictions = {}
    for cid in split["mechanism_test"]:
        gate = activity_from_mask(model.predict_proba(matrices[cid][0])[:, 1] >= selected["threshold"],
                                  matrices[cid][1])
        predictions[cid] = learned_stages(codes[cid], gate, records[cid][1].supported).to_dict()
    save_json(out / "predictions.json", {"xgboost_scale_information": predictions})
    save_json(out / "manifest.json", {
        "status": "predictions_frozen_not_test_scored", "train_eligible_days": int(eligible.sum()),
        "validation_trials": len(trials), "source_stage_head": sha(source / "stage-xgboost.json"),
        "prediction_sha256": sha(out / "predictions.json"), "new_api_requests": 0,
    })
    print({"status": "frozen", "trials": len(trials), "depth": selected["depth"],
           "threshold": selected["threshold"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.source, args.out)


if __name__ == "__main__":
    main()
