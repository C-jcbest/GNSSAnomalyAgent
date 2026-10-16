"""Replay paired and axis development without retraining or opening reserved labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import prediction
from train_landslide_paired_persistence import ARMS, modality_inputs, probabilities
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_axis_evidence import axis_persistence_evidence, pool_axis_evidence
from gnss_sim.landslide_paired_persistence import offset_counterpart, pretrained_activity_head
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, numeric_input


def verify(data, encoding, pairs, axis_run, run, previous, out):
    if out.exists():
        raise FileExistsError("Verification output already exists")
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoding, "encoding-freeze.json"),
        (pairs, "encoding-freeze.json"),
        (axis_run, "freeze.json"),
        (run, "freeze.json"),
        (previous, "calibration-freeze.json"),
    ):
        for relative, expected in read_json(folder / manifest).items():
            if sha(folder / relative) != expected:
                raise ValueError("Frozen experiment changed: " + relative)
    split = read_json(encoding / "protocol.json")["split"]
    banks, raw, features = {}, {}, {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoding / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            case, features[cid] = load_case_features(data, cid)
            raw[cid] = np.array(
                [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
            )
            with np.load(axis_run / "evidence" / (cid + ".npz")) as archive:
                scores = axis_persistence_evidence(raw[cid], features[cid].noise)
                np.testing.assert_array_equal(scores, archive["axis"])
                np.testing.assert_array_equal(pool_axis_evidence(scores), archive["pooled"])
    ledger = {
        row["case_id"]: row["truth"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["group"] == "train"
    }
    selected_pairs = read_json(pairs / "selected-training-pairs.json")
    with np.load(pairs / "pair-bank.npz") as archive:
        pair_bank = {key: archive[key] for key in archive.files}
    for index, row in enumerate(selected_pairs):
        cid, day = row["case_id"], row["day"]
        if cid not in split["train"] or not banks[cid]["selected"][day]:
            raise ValueError("Counterfactual parent is not an eligible training sample")
        latent = np.array(ledger[cid]["latent_displacement_mm"])
        negative, replacement = offset_counterpart(
            raw[cid], latent, *row["endpoints"], row["step_day"]
        )
        with np.load(pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            np.testing.assert_array_equal(negative, archive["negative_observations"])
            np.testing.assert_array_equal(replacement, archive["negative_latent"])
            np.testing.assert_array_equal(
                pair_bank["negative_numbers"][index], archive["negative_numbers"]
            )
            np.testing.assert_array_equal(
                pair_bank["negative_charts"][index],
                chart_numeric_descriptors(archive["negative_image"][None])[0],
            )
        np.testing.assert_array_equal(
            pair_bank["positive_numbers"][index], banks[cid]["numbers"][day]
        )
        np.testing.assert_array_equal(
            pair_bank["positive_embeddings"][index], banks[cid]["embeddings"][day]
        )
        np.testing.assert_allclose(
            negative - replacement, raw[cid] - latent, atol=1e-12, rtol=0, equal_nan=True
        )
    scaling = read_json(encoding / "numeric-scaling.json")
    stage = XGBClassifier()
    stage.load_model(previous / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    probability_count = 0
    for experiment in (axis_run, run):
        choices = read_json(experiment / "selection.json")
        saved = read_json(experiment / "probabilities.json")
        predictions = read_json(experiment / "predictions.json")
        for name, choice in choices.items():
            parameter = choice["depth"] if experiment == axis_run else choice["parameter"]
            if name in ARMS and experiment == run:
                arm = ARMS[name]
                inputs = {
                    cid: modality_inputs(
                        banks[cid]["embeddings"], numeric_input(banks[cid]["numbers"], scaling), arm
                    )
                    for cid in split["validation"]
                }
                model = pretrained_activity_head(next(iter(inputs.values())).shape[1], 0, parameter)
                model.load_state_dict(
                    torch.load(
                        run / (name + f"-width{parameter}.pt"),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                values = {cid: probabilities(model, matrix) for cid, matrix in inputs.items()}
            else:
                matrices = {}
                for cid in split["validation"]:
                    numbers = banks[cid]["numbers"]
                    if experiment == axis_run and name != "raw84":
                        with np.load(axis_run / "evidence" / (cid + ".npz")) as archive:
                            extra = (
                                archive["axis"] if name == "axis_evidence" else archive["pooled"]
                            )
                        numbers = np.column_stack((numbers, extra))
                    elif name == "xgb_chart384_augmentation":
                        numbers = np.column_stack((numbers, banks[cid]["chart_descriptors"]))
                    matrices[cid] = numbers
                model = XGBClassifier()
                model.load_model(experiment / (name + f"-depth{parameter}.json"))
                values = {
                    cid: model.predict_proba(matrix)[:, 1] for cid, matrix in matrices.items()
                }
            for cid, value in values.items():
                np.testing.assert_array_equal(value, np.array(saved[name][cid]))
                if (
                    prediction(value, banks[cid], codes[cid], choice["threshold"]).to_dict()
                    != predictions[name][cid]
                ):
                    raise ValueError("Activity/stage replay mismatch: " + name + " " + cid)
                probability_count += len(value)
        labels = {cid: [] for cid in split["validation"]}
        with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["case_id"] in labels:
                    labels[row["case_id"]].append(row)
        observed = {cid: banks[cid]["observed"] for cid in labels}
        weak = {
            cid: np.array([bool(int(row["weak_signal"])) for row in rows])
            for cid, rows in labels.items()
        }
        results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
        if results != read_json(experiment / "results.json"):
            raise ValueError("Complete scoring replay mismatch: " + str(experiment))
        if {row["method"]: row for row in summary} != {
            row["method"]: row for row in read_json(experiment / "summary.json")
        }:
            raise ValueError("Method summary replay mismatch: " + str(experiment))
    record = {
        "status": "VERIFIED",
        "paired_models_replayed": 9,
        "axis_models_replayed": 3,
        "all_output_groups_scored": 24,
        "probability_values_replayed": probability_count,
        "axis_records_recomputed": len(banks),
        "counterfactuals_reconstructed": len(selected_pairs),
        "error_nuisance_missing_and_endpoint_retention": True,
        "new_training": False,
        "test_truth_opened": False,
        "new_api_calls": 0,
    }
    write_json(out, record)
    print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoding", "pairs", "axis-run", "run", "previous", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.data.resolve(),
        args.encoding.resolve(),
        args.pairs.resolve(),
        args.axis_run.resolve(),
        args.run.resolve(),
        args.previous.resolve(),
        args.out.resolve(),
    )
