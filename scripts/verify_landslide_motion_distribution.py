"""Replay all selected pilot outputs and sample auxiliary visual encodings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_motion_distribution import ARMS
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_motion_distribution import information_matrix, inputs_for_arm
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_motion_distribution_view import (
    distribution_view_inputs,
    render_distribution_view,
)
from gnss_sim.landslide_motion_mass import motion_mass_descriptors
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling


def verify(data, existing, encoded, controls, run, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (existing, "encoding-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (controls, "calibration-freeze.json"),
        (run, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    plan = read_json(run / "training-plan.json")
    if plan["encoding_manifest_sha256"] != sha(encoded / "encoding-freeze.json") or plan[
        "controls_manifest_sha256"
    ] != sha(controls / "calibration-freeze.json"):
        raise ValueError("Models are bound to different evidence")
    out.mkdir(parents=True, exist_ok=False)
    banks, evidence, examples, training_numbers, training_information = {}, {}, [], [], []
    for group, ids in protocol["split"].items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                bank = {key: archive[key] for key in archive.files}
            with np.load(existing / group / (cid + ".npz")) as archive:
                for key in archive.files:
                    np.testing.assert_array_equal(bank[key], archive[key])
            case, features = load_case_features(data, cid)
            raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
            view = distribution_view_inputs(raw, features.noise)
            blocks, available, _, _ = discovery_block_matrix(raw, features.noise)
            np.testing.assert_array_equal(bank["distribution_descriptors"], view["descriptors"])
            np.testing.assert_array_equal(
                bank["raw_profile"], np.column_stack((blocks.reshape(len(raw), -1), available))
            )
            evidence[cid] = motion_mass_descriptors(raw, features.noise)
            with np.load(run / "mass-evidence" / (cid + ".npz")) as archive:
                for key, value in evidence[cid].items():
                    np.testing.assert_array_equal(value, archive[key])
            indices = np.flatnonzero(bank["selected"])
            if group == "train":
                training_numbers.append(bank["numbers"][indices])
                training_information.append(information_matrix(bank)[indices])
            for index in (indices[0], indices[len(indices) // 2]):
                for permuted, key in ((False, "distribution"), (True, "permuted")):
                    examples.append(
                        (
                            render_distribution_view(
                                view["magnitude"][index], view["agreement"][index], permuted
                            ),
                            bank[key][index],
                        )
                    )
            banks[cid] = bank
    scaling = read_json(encoded / "numeric-scaling.json")
    information_scaling = read_json(encoded / "information-scaling.json")
    if (
        fit_numeric_scaling(np.concatenate(training_numbers)) != scaling
        or fit_numeric_scaling(np.concatenate(training_information)) != information_scaling
    ):
        raise ValueError("TRAIN-only feature scaling changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0",
        pretrained=None,
        image_mean=config["mean"],
        image_std=config["std"],
        image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"],
    )
    weights = BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"
    if sha(weights) != protocol["weights_sha256"]:
        raise ValueError("Frozen visual weights changed")
    encoder.visual.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    encoder.eval()
    encoder.requires_grad_(False)
    maximum_error = 0.0
    with torch.inference_mode():
        for first in range(0, len(examples), 32):
            batch = examples[first : first + 32]
            values = torch.nn.functional.normalize(
                encoder.visual(torch.stack([transform(item[0]) for item in batch])), dim=-1
            ).numpy()
            expected = np.stack([item[1] for item in batch])
            np.testing.assert_allclose(values, expected, atol=1e-6, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(values - expected))))
    labels = {cid: [] for cid in protocol["split"]["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(controls / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    choices = read_json(run / "selection.json")
    original_probability = read_json(run / "probabilities.json")
    original_predictions = read_json(run / "predictions.json")
    predictions = read_json(controls / "predictions.json")
    for name, choice in choices.items():
        if name in ARMS:
            inputs = {
                cid: inputs_for_arm(banks[cid], ARMS[name], scaling, information_scaling)
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
            matrices = {}
            for cid in labels:
                bank = banks[cid]
                if name == "distribution_information_xgb":
                    matrices[cid] = information_matrix(bank)
                elif name == "all_distribution_information_xgb":
                    matrices[cid] = information_matrix(bank, include_chart=True)
                else:
                    key = (
                        "global_distribution"
                        if name == "pilot_global_mass_xgb"
                        else "query_distribution"
                    )
                    matrices[cid] = np.column_stack((bank["numbers"], evidence[cid][key]))
            model = XGBClassifier()
            model.load_model(run / f"{name}-depth{choice['parameter']}.json")
            probability = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in labels}
        for cid in labels:
            np.testing.assert_array_equal(
                probability[cid], np.array(original_probability[name][cid])
            )
        predictions[name] = {
            cid: decode(probability[cid], choice["threshold"], banks[cid], codes[cid]).to_dict()
            for cid in labels
        }
    if predictions != original_predictions:
        raise ValueError("Selected activity/stage outputs or original controls changed")
    if predictions["duplicate_motion_global"] != predictions["duplicate_global_numeric"]:
        raise ValueError("Re-fitted duplicate does not reproduce the identical original16/8control")
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or {r["method"]: r for r in summary} != {
        r["method"]: r for r in read_json(run / "summary.json")
    }:
        raise ValueError("Unified scores changed")
    receipt = dict(
        status="VERIFIED",
        original_records=len(banks),
        exact_raw_descriptors_profiles_scalings_and_mass=True,
        original_embedding_support_labels_exact=True,
        sampled_visual_encodings=len(examples),
        encoding_atol=1e-6,
        maximum_embedding_error=maximum_error,
        selected_models=len(choices),
        probabilities=len(choices) * len(labels) * 730,
        methods=len(predictions),
        record_outputs=len(predictions) * len(labels),
        exact_probabilities_activity_stage_scores=True,
        duplicate_refit_identical=True,
        no_training=True,
        test_truth_opened=False,
        run_manifest_sha256=sha(run / "freeze.json"),
        encoding_manifest_sha256=sha(encoded / "encoding-freeze.json"),
    )
    write_json(out / "verification.json", receipt)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "encoded", "controls", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.data.resolve(),
        args.existing.resolve(),
        args.encoded.resolve(),
        args.controls.resolve(),
        args.run.resolve(),
        args.out.resolve(),
    )
