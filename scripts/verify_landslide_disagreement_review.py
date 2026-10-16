"""Replay held-record sources, revision permissions and all unified metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_disagreement_review import (
    MAIN,
    load_banks,
    neural_inputs,
    source_bindings,
    tree_inputs,
)
from run_landslide_window_optimization import load_case_features, read_json
from scipy.special import expit
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_disagreement_review import (
    review_benefit,
    review_features,
    revise_disagreements,
)
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input


def neural_model(path, dimensions, width):
    model = pretrained_activity_head(dimensions, 0, width)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def tree_model(path):
    model = XGBClassifier()
    model.load_model(path)
    return model


def verify(data, encoded, run, out):
    verify_manifest(run, "freeze.json")
    verify_manifest(encoded, "encoding-freeze.json")
    protocol = read_json(run / "protocol.json")
    if protocol["source_bindings"] != source_bindings():
        raise ValueError("Revision source changed")
    if sha(encoded / "encoding-freeze.json") != protocol["encoding_manifest_sha256"]:
        raise ValueError("Revision encoding changed")
    if sha(data / "data-freeze.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Revision data changed")
    banks = load_banks(encoded, protocol["split"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    oof = read_json(run / "out-of-fold.json")
    audit = read_json(run / "crossfit-audit.json")
    training_ids = protocol["split"]["train"]
    if sorted(cid for fold in protocol["folds"] for cid in fold) != sorted(training_ids):
        raise ValueError("Every TRAIN record must be held exactly once")
    oof_values = 0
    for index, held in enumerate(protocol["folds"]):
        expected_train = [cid for cid in training_ids if cid not in held]
        if audit[index] != dict(fold=index, excluded=held, training_ids=expected_train, overlap=[]):
            raise ValueError("Cross-fit record exclusion differs")
        scaling = fit_numeric_scaling(
            np.concatenate(
                [banks[cid]["numbers"][banks[cid]["selected"]] for cid in expected_train]
            )
        )
        if scaling != read_json(run / f"fold{index}-scaling.json"):
            raise ValueError("Held record influenced source numerical scaling")
        visual = neural_model(run / f"fold{index}-visual.pt", 548, 16)
        numeric = tree_model(run / f"fold{index}-numeric.json")
        for cid in held:
            bank = banks[cid]
            selected = bank["selected"]
            left = probabilities(visual, neural_inputs(bank, scaling, "visual_numeric")[selected])
            right = numeric.predict_proba(bank["numbers"][selected])[:, 1]
            if not np.array_equal(left, oof[cid]["visual"]) or not np.array_equal(
                right, oof[cid]["numeric"]
            ):
                raise ValueError("Held-record probability replay differs: " + cid)
            if oof[cid]["actual"] != bank["activity"][selected].tolist() or (
                oof[cid]["original_days"] != np.flatnonzero(selected).tolist()
            ):
                raise ValueError("Held-record label calendar differs")
            oof_values += len(left) * 2
    scaling = read_json(run / "numeric-scaling.json")
    choices = read_json(run / "selection.json")
    trials = read_json(run / "validation-trials.json")
    for name, choice in choices.items():
        if choice != max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        ):
            raise ValueError("Predefined first-maximum selection differs: " + name)
    base_saved = read_json(run / "fixed-source-probabilities.json")
    visual = neural_model(run / "visual_numeric-width16.pt", 548, 16)
    numeric = tree_model(run / "xgb_raw84-depth2.json")
    base = dict(visual={}, numeric={})
    validation_ids = protocol["split"]["validation"]
    numbers = {cid: numeric_input(banks[cid]["numbers"], scaling) for cid in validation_ids}
    for cid in validation_ids:
        base["visual"][cid] = probabilities(
            visual, neural_inputs(banks[cid], scaling, "visual_numeric")
        )
        base["numeric"][cid] = numeric.predict_proba(banks[cid]["numbers"])[:, 1]
        for source in base:
            if not np.array_equal(base[source][cid], base_saved[source][cid]):
                raise ValueError("Fixed source probability differs")
    stage = tree_model(run / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in validation_ids
    }
    review_models = read_json(run / "review-models.json")
    stacking = read_json(run / "stacking-models.json")
    saved_probability = read_json(run / "probabilities.json")
    saved_predictions = read_json(run / "predictions.json")
    predictions, check_rows, permission_rows = {}, [], []
    for name, choice in choices.items():
        parameter, threshold = choice["parameter"], choice["threshold"]
        probability, supports = {}, {}
        if name in {"visual_numeric", "pure_visual", "numeric_mlp"}:
            dimensions = {"visual_numeric": 548, "pure_visual": 512, "numeric_mlp": 36}[name]
            model = neural_model(run / f"{name}-width{parameter}.pt", dimensions, parameter)
        elif name.startswith("xgb_"):
            model = tree_model(run / f"{name}-depth{parameter}.json")
        for cid in validation_ids:
            bank = banks[cid]
            support = bank["supported"]
            if name in {"visual_numeric", "pure_visual", "numeric_mlp"}:
                value = probabilities(model, neural_inputs(bank, scaling, name))
            elif name.startswith("xgb_"):
                matrix, support = tree_inputs(bank, name)
                value = model.predict_proba(matrix)[:, 1]
            elif name == "fixed_blend":
                value = (1 - parameter) * base["visual"][cid] + parameter * base["numeric"][cid]
            elif name in {MAIN, "scores_only_review"}:
                features = review_features(
                    base["visual"][cid],
                    base["numeric"][cid],
                    numbers[cid] if name == MAIN else None,
                    threshold,
                )
                benefit = review_benefit(review_models[name][f"{parameter}:{threshold}"], features)
                value, switched = revise_disagreements(
                    base["visual"][cid], base["numeric"][cid], support, benefit, threshold
                )
                disagreements = (base["visual"][cid] >= threshold) != (
                    base["numeric"][cid] >= threshold
                )
                if np.any(switched & (~support | ~disagreements | (benefit <= 0))):
                    raise ValueError("Revision escaped its evidence permission")
                if not np.array_equal(value[~switched], base["visual"][cid][~switched]):
                    raise ValueError("Review changed an unpermitted query")
                permission_rows.append(
                    dict(
                        method=name,
                        case_id=cid,
                        switched=int(switched.sum()),
                        supported_disagreements=int((support & disagreements).sum()),
                    )
                )
            elif name in {"generic_raw_stacker", "generic_score_stacker"}:
                features = review_features(
                    base["visual"][cid],
                    base["numeric"][cid],
                    numbers[cid] if name == "generic_raw_stacker" else None,
                    0.5,
                )
                saved = stacking[name][str(parameter)]
                # StandardScaler preserves the input float32 dtype and rounds
                # subtraction/division separately; float64 recomputation would
                # change the stored classifier's probabilities slightly.
                scaled = features.copy()
                scaled -= np.array(saved["mean"])
                scaled /= np.array(saved["scale"])
                logits = scaled @ np.array(saved["coefficient"])[:, None] + saved["intercept"]
                value = expit(logits[:, 0])
            else:
                feature = load_case_features(data, cid)[1]
                if name == "robust_rule":
                    value = np.where(
                        feature.signal_to_noise >= parameter, feature.progressive_rate, -1.0
                    )
                    support = feature.supported
                else:
                    rates = pelt_segment_rates(feature, parameter)
                    value = np.minimum(rates, 2 * feature.progressive_rate)
                    value[feature.signal_to_noise < 1] = -1.0
                    support = feature.supported & np.isfinite(rates)
                value = np.where(np.isfinite(value), value, -1.0)
            exact = np.array_equal(value, saved_probability[name][cid])
            if not exact and not (
                name.startswith("generic_")
                and np.allclose(value, saved_probability[name][cid], rtol=0, atol=1e-12)
            ):
                raise ValueError("Selected revision probability differs: " + name + ":" + cid)
            probability[cid], supports[cid] = value, support
            check_rows.append(
                dict(
                    method=name,
                    case_id=cid,
                    scores=len(value),
                    exact=exact,
                    max_abs_error=float(np.max(np.abs(value - saved_probability[name][cid]))),
                )
            )
        predictions[name] = {
            cid: decode(
                probability[cid], threshold, banks[cid], codes[cid], supports[cid]
            ).to_dict()
            for cid in validation_ids
        }
        if predictions[name] != saved_predictions[name]:
            raise ValueError("Activity/stage replay differs: " + name)
    labels = {cid: [] for cid in validation_ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Unified review per-method scoring differs")
    expected_summary = read_json(run / "summary.json")
    if {row["method"]: row for row in summary} != {row["method"]: row for row in expected_summary}:
        raise ValueError("Unified review summary scoring differs")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "model-replay.csv", check_rows)
    write_csv(out / "revision-permission.csv", permission_rows)
    write_csv(out / "per-case-replay.csv", records)
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED",
            oof_records=len(oof),
            oof_probability_values=oof_values,
            methods=len(predictions),
            model_cases=len(check_rows),
            validation_scores_replayed=sum(row["scores"] for row in check_rows),
            exact_probability_cases=sum(row["exact"] for row in check_rows),
            generic_stacker_probability_tolerance=1e-12,
            max_probability_error=max(row["max_abs_error"] for row in check_rows),
            all_predictions_activity_stage_scores_exact=True,
            no_record_leak_in_source_fits=True,
            no_training=True,
            test_truth_opened=False,
            new_api_calls=0,
            calibration_manifest_sha256=sha(run / "freeze.json"),
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.data.resolve(), args.encoded.resolve(), args.run.resolve(), args.out.resolve())
