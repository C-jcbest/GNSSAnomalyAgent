"""Re-fit TRAIN-only expectations and replay every selected conditional-visual output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT
from run_landslide_conditional_visual import ARMS, arm_inputs, motion_numbers, tree_inputs
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from threadpoolctl import threadpool_limits
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_conditional_visual import fit_visual_expectation
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input


def verify(data, encoded, parent, run, out):
    stage_source = data.parent / "disagreement-calibration-v1"
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (parent, "freeze.json"),
        (stage_source, "freeze.json"),
        (run, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(run / "protocol.json")
    for folder, key, manifest in (
        (data, "data_manifest_sha256", "data-freeze.json"),
        (encoded, "encoding_manifest_sha256", "encoding-freeze.json"),
        (parent, "parent_manifest_sha256", "freeze.json"),
        (stage_source, "stage_manifest_sha256", "freeze.json"),
    ):
        if sha(folder / manifest) != protocol[key]:
            raise ValueError("Bound experiment input changed: " + key)
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    split = protocol["split"]
    banks = load_banks(encoded, split)
    evidence = {}
    for cid in banks:
        with np.load(parent / "evidence" / (cid + ".npz")) as archive:
            evidence[cid] = {key: archive[key] for key in archive.files}
    numeric_training = np.concatenate(
        [
            motion_numbers(banks[cid], evidence[cid])[banks[cid]["selected"]]
            for cid in split["train"]
        ]
    )
    scaling = fit_numeric_scaling(numeric_training)
    if scaling != read_json(run / "numeric-scaling.json"):
        raise ValueError("TRAIN-only motion scaling changed")
    scaled = numeric_input(numeric_training, scaling)
    visual = np.concatenate(
        [banks[cid]["embeddings"][banks[cid]["selected"]] for cid in split["train"]]
    )
    expectation = fit_visual_expectation(scaled, visual)
    shuffled = fit_visual_expectation(scaled, visual, shuffled=True)
    if expectation != read_json(run / "visual-expectation.json") or shuffled != read_json(
        run / "shuffled-expectation.json"
    ):
        raise ValueError("TRAIN-only visual expectation refit changed")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    original_predictions = read_json(run / "predictions.json")
    original_probabilities = read_json(run / "probabilities.json")
    predictions = read_json(parent / "predictions.json")
    choices = read_json(run / "selection.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for name, choice in choices.items():
        if name in ARMS:
            inputs = {
                cid: arm_inputs(
                    banks[cid], evidence[cid], scaling, expectation, shuffled, ARMS[name]
                )
                for cid in labels
            }
            model = pretrained_activity_head(
                next(iter(inputs.values())).shape[1], 0, choice["parameter"]
            )
            model.load_state_dict(
                torch.load(
                    run / f"{name}-width{choice['parameter']}.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            probability = {cid: probabilities(model, inputs[cid]) for cid in labels}
        else:
            model = XGBClassifier()
            model.load_model(run / f"{name}-depth{choice['parameter']}.json")
            probability = {
                cid: model.predict_proba(tree_inputs(banks[cid], evidence[cid], name))[:, 1]
                for cid in labels
            }
        for cid in labels:
            np.testing.assert_array_equal(
                probability[cid], np.array(original_probabilities[name][cid])
            )
        predictions[name] = {
            cid: decode(probability[cid], choice["threshold"], banks[cid], codes[cid]).to_dict()
            for cid in labels
        }
    if predictions != original_predictions:
        raise ValueError("Selected outputs or retained original controls changed")
    if predictions["refit_global_motion_mass"] != predictions["global_motion_mass"]:
        raise ValueError("Numerical control refit differs from its original")
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or {r["method"]: r for r in summary} != {
        r["method"]: r for r in read_json(run / "summary.json")
    }:
        raise ValueError("Unified activity/stage/boundary scores changed")
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED",
            original_records=len(banks),
            expectation_fits=2,
            exact_train_scaling_projection_refit=True,
            activity_labels_in_projection=False,
            new_encoder_calls=0,
            selected_models=len(choices),
            probabilities=len(choices) * len(labels) * 730,
            methods=len(predictions),
            record_outputs=len(predictions) * len(labels),
            numerical_refit_identical=True,
            exact_probabilities_activity_stage_scores=True,
            classifier_retraining=False,
            test_truth_opened=False,
            run_manifest_sha256=sha(run / "freeze.json"),
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(
            args.data.resolve(),
            args.encoded.resolve(),
            args.parent.resolve(),
            args.run.resolve(),
            args.out.resolve(),
        )
