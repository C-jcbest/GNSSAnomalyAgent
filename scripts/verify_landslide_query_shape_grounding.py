"""Rebuild teachers, all14 neural outputs,266 calibration trials and unified scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_comparison import calibration_objective
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_query_shape_grounding import grounded_chart_model
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, raw_chart_patches


def independent_teacher(descriptors, groups):
    positions = -descriptors[:, :183].reshape(len(descriptors), 3, 61)
    centers = np.full((len(descriptors), 3, 3), np.nan)
    for group_index, (start, stop) in enumerate(groups):
        for row in range(len(descriptors)):
            for axis in range(3):
                values = positions[row, axis, start:stop]
                finite = values[np.isfinite(values)]
                if len(finite) >= 4:
                    centers[row, group_index, axis] = np.median(finite)
    return np.column_stack((centers[:, 1] - centers[:, 0], centers[:, 2] - centers[:, 1]))


def independent_scaling(matrix):
    mean, scale = [], []
    for index in range(matrix.shape[1]):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        center = float(np.mean(finite)) if len(finite) else 0.0
        standard = float(np.std(finite)) if len(finite) else 1.0
        mean.append(center)
        scale.append(standard if standard >= 1e-12 else 1.0)
    return dict(mean=mean, scale=scale)


def forward(model, arm, images, numbers, scaling):
    probability, auxiliary = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), 256):
            modalities = []
            if arm["image"]:
                features = model.image_encoder(torch.from_numpy(images[start:start + 256]))
                modalities.append(features)
                auxiliary.append(
                    model.shape_head(features).numpy() * np.array(scaling["scale"]) + np.array(scaling["mean"])
                )
            if arm["numeric"]:
                modalities.append(torch.from_numpy(numbers[start:start + 256]))
            logits = model.classifier(torch.cat(modalities, dim=1)).squeeze(1)
            probability.extend(torch.sigmoid(logits).numpy().tolist())
    return np.array(probability), np.concatenate(auxiliary) if auxiliary else None


def verify(run, out):
    verify_manifest(run, "freeze.json")
    protocol = read_json(run / "protocol.json")
    for name in ("source_bindings", "input_bindings"):
        for path, expected in protocol[name].items():
            if sha(Path(path)) != expected:
                raise ValueError("Frozen grounding input/source changed: " + path)
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Completed training required")
    split, data, encoded = protocol["split"], Path(protocol["data"]), Path(protocol["encoded"])
    banks = load_banks(encoded, split)
    images, query, global_shape = {}, {}, {}
    for cid, bank in banks.items():
        case, features = load_case_features(data, cid)
        images[cid] = raw_chart_patches(case, features.noise)
        descriptors = chart_numeric_descriptors(images[cid])
        np.testing.assert_allclose(descriptors, bank["chart_descriptors"], atol=2e-7, rtol=2e-7, equal_nan=True)
        query[cid] = independent_teacher(descriptors, ((23, 28), (28, 33), (33, 38)))
        global_shape[cid] = independent_teacher(descriptors, ((0, 5), (28, 33), (56, 61)))
        with np.load(run / "targets" / (cid + ".npz")) as stored:
            np.testing.assert_array_equal(query[cid], stored["query"])
            np.testing.assert_array_equal(global_shape[cid], stored["global_shape"])
    training = np.concatenate([bank["numbers"][bank["selected"]] for cid, bank in banks.items() if cid in split["train"]])
    scaling = read_json(run / "scaling.json")
    independent = dict(
        numbers=independent_scaling(training),
        query=independent_scaling(np.concatenate([query[cid][banks[cid]["selected"]] for cid in split["train"]])),
        global_shape=independent_scaling(np.concatenate([global_shape[cid][banks[cid]["selected"]] for cid in split["train"]])),
    )
    for name, value in independent.items():
        for key in ("mean", "scale"):
            np.testing.assert_allclose(value[key], scaling[name][key], rtol=1e-12, atol=1e-12)
    numeric_scaling = scaling["numbers"]
    numbers = {}
    for cid, bank in banks.items():
        available = np.isfinite(bank["numbers"])
        replaced = np.where(available, bank["numbers"], numeric_scaling["mean"])
        standardized = (replaced - numeric_scaling["mean"]) / numeric_scaling["scale"]
        numbers[cid] = np.column_stack((standardized, ~available)).astype(np.float32)
    stage = XGBClassifier()
    stage.load_model(Path(protocol["stage_source"]) / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in split["validation"]}
    if {cid: values.tolist() for cid, values in codes.items()} != read_json(run / "stage-codes.json"):
        raise ValueError("Shared48TRAIN stage outputs differ")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    torch.set_num_threads(2)
    choices = read_json(run / "selection.json")
    saved_probability = read_json(run / "probabilities.json")
    saved_auxiliary = read_json(run / "auxiliary.json")
    trials, predictions, fidelity = [], {}, {}
    maximum_error = 0.0
    for name, arm in protocol["arms"].items():
        scale = scaling["global_shape"] if arm["supervision"] == "global" else scaling["query"]
        outputs = {}
        for width in (16, 32):
            model = grounded_chart_model(width, numbers[split["train"][0]].shape[1], arm["image"], arm["numeric"])
            model.load_state_dict(torch.load(run / "models" / f"{name}-width{width}.pt", map_location="cpu", weights_only=True))
            probability, auxiliary = {}, {}
            for cid in labels:
                probability[cid], auxiliary[cid] = forward(model, arm, images[cid], numbers[cid], scale)
            for threshold in THRESHOLDS:
                decoded = {cid: decode(probability[cid], threshold, banks[cid], codes[cid]) for cid in labels}
                aggregate = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in labels])
                trials.append(dict(method=name, parameter=width, threshold=threshold,
                                   objective=calibration_objective(aggregate), validation=aggregate))
                if width == choices[name]["parameter"] and threshold == choices[name]["threshold"]:
                    outputs = {cid: value.to_dict() for cid, value in decoded.items()}
            if width != choices[name]["parameter"]:
                continue
            errors = []
            for cid in labels:
                expected = np.array(saved_probability[name][cid])
                maximum_error = max(maximum_error, float(np.max(np.abs(probability[cid] - expected))))
                np.testing.assert_array_equal(probability[cid], expected)
                value = auxiliary[cid]
                if value is None:
                    continue
                np.testing.assert_array_equal(value, np.array(saved_auxiliary[name][cid]))
                valid = np.isfinite(query[cid]) & banks[cid]["supported"][:, None]
                errors.extend(np.abs(value - np.nan_to_num(query[cid]))[valid].tolist())
            fidelity[name] = float(np.mean(errors)) if errors else None
        predictions[name] = outputs
        selected = max((row for row in trials if row["method"] == name), key=lambda row: row["objective"])
        if selected != choices[name]:
            raise ValueError("Stable-first equal-budget selection differs")
    def trial_key(row):
        return row["method"], row["parameter"], row["threshold"]
    if sorted(trials, key=trial_key) != sorted(read_json(run / "trials.json"), key=trial_key) or fidelity != read_json(run / "shape-fidelity.json"):
        raise ValueError("All266 trials or evidence fidelity differ")
    predictions.update(read_json(Path(protocol["parent"]) / "predictions.json"))
    if predictions != read_json(run / "predictions.json"):
        raise ValueError("All activity/stage outputs or preserved strong controls differ")
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in values]) for cid, values in labels.items()}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or sorted(summary, key=lambda row: row["method"]) != sorted(
        read_json(run / "summary.json"), key=lambda row: row["method"]
    ):
        raise ValueError("Unified day/event/boundary/stage score differs")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "verification.json", dict(
        status="VERIFIED", records_reconstructed=len(banks), neural_models=14, calibration_trials=len(trials),
        methods=len(predictions), outputs=len(predictions) * len(labels), maximum_probability_error=maximum_error,
        teachers="Independent median groups from regenerated original chart ordinates", scaling="Independent TRAIN-only moments",
        coverage="All14 retained weights/all266trials/stable selection/all probabilities/auxiliary/activity/stage/scores",
        limitations="No optimizer retraining;source/training log bindings preserved;not independent effectiveness",
        new_api_calls=0, reserved_content_loaded=False, run_manifest_sha256=sha(run / "freeze.json"),
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.run.resolve(), arguments.out.resolve())
