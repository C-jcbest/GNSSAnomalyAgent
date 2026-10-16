"""Separate longer-feature value from expanded raw support in the ML control."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, calibration_objective, load_labels, save_json, sha
from run_landslide_longer_motion_review import verify
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages


def fit(run, out):
    verify(run)
    training = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/motion-probe-v2"
    verify_manifest(training)
    config = read_json(run / "protocol.json")
    split = read_json(training / "protocol.json")["split"]
    ids = split["train"] + split["validation"] + config["case_ids"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "protocol.json", {
        "scope": "Viewed development ML support ablation; not new independent confirmation",
        "features": "Same original matrix +42-day rate/noise/direction, excludes 84-day values",
        "eligibility": "Exactly the new 84-day model's observed short/42/84 raw support union",
        "calibration": "Same train24/validation12, depths2/4, thresholds.35/.5/.65, "
                       "daily F1+event F1, 120 estimators, learning rate.05, seed20261007",
        "stage_head": "Unchanged common trained stage head and original stage support",
        "split": {name: split[name] for name in ("train", "validation")}, "test": config["case_ids"],
    })
    stage_head = XGBClassifier()
    stage_head.load_model(run / "stage-xgboost.json")
    features, matrices, supported, codes = {}, {}, {}, {}
    for cid in ids:
        source = training if cid in split["train"] + split["validation"] else run
        case, feature = load_case_features(source, cid)
        h42, h84 = raw_motion_evidence(case, feature), raw_motion_evidence(case, feature, 84)
        features[cid] = feature
        matrices[cid] = np.column_stack((feature.matrix, h42["rate"], h42["signal_to_noise"], h42["direction"]))
        supported[cid] = feature.observed & (np.isfinite(feature.progressive_rate)
                                            | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))
        codes[cid] = stage_head.predict(feature.matrix).astype(int) + 1
    cases = {cid: load_case_features(training, cid)[0] for cid in split["train"] + split["validation"]}
    labels = load_labels(run / "training-reference.csv", cases)
    train_x = np.concatenate([matrices[cid] for cid in split["train"]])
    train_y = np.array([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    eligible = np.concatenate([supported[cid] for cid in split["train"]]) & (train_y >= 0)
    trials, models = [], {}
    for depth in (2, 4):
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                              n_jobs=2, random_state=20261007)
        model.fit(train_x[eligible], train_y[eligible])
        models[depth] = model
        probabilities = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]}
        for threshold in (.35, .5, .65):
            scores = []
            for cid in split["validation"]:
                gate = activity_from_mask(probabilities[cid] >= threshold, supported[cid])
                prediction = learned_stages(codes[cid], gate, features[cid].supported)
                scores.append(evaluate_case(labels[cid], prediction))
            score = aggregate_scores(scores)
            trials.append({"depth": depth, "threshold": threshold,
                           "objective": calibration_objective(score), "validation": score})
    choice = max(trials, key=lambda row: row["objective"])
    save_json(out / "validation-trials.json", trials)
    save_json(out / "frozen-selection.json", choice)
    model = models[choice["depth"]]
    model.save_model(out / "activity-xgboost-42-matched-support.json")
    predictions = {}
    for cid in config["case_ids"]:
        gate = activity_from_mask(model.predict_proba(matrices[cid])[:, 1] >= choice["threshold"], supported[cid])
        predictions[cid] = learned_stages(codes[cid], gate, features[cid].supported).to_dict()
    save_json(out / "predictions.json", {"xgboost_42_matched_support": predictions})
    save_json(out / "support-population.json", {
        "train_eligible_days": int(eligible.sum()),
        "observed_raw_supported_days": {cid: int(mask.sum()) for cid, mask in supported.items()},
    })
    save_json(out / "manifest.json", {"new_vlm_requests": 0,
        "source_bindings": {str(path): sha(path) for path in (
            run / "freeze.json", training / "manifest.json", run / "stage-xgboost.json")},
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Matched raw-support 42-day XGBoost control fitted and frozen without API calls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.run, args.out)


if __name__ == "__main__":
    main()
