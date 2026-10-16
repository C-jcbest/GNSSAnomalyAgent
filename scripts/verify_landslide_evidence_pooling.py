"""Reconstruct physical guides, sample encoder pools and replay all selected scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_evidence_pooling import ARMS
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from threadpoolctl import threadpool_limits
from train_landslide_evidence_pooling import numeric_information, pooling_inputs
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evidence_pooling import (
    continuity_guides,
    evidence_weighted_embedding,
    feature_weights,
)
from gnss_sim.landslide_motion_mass import motion_mass_descriptors
from gnss_sim.landslide_paired_persistence import pretrained_activity_head, render_encoder_chart
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, raw_chart_patches


def verify(data, encoded, evidence_source, parent, run, out):
    stage_source = data.parent / "spatial-pilot-calibration-v1"
    for folder, manifest in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
        (evidence_source, "freeze.json"), (parent, "freeze.json"),
        (run, "freeze.json"), (stage_source, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    if read_json(run / "protocol.json") != protocol:
        raise ValueError("Encoding and classifier protocols differ")
    for folder, manifest in (
        (data, "data-freeze.json"), (evidence_source, "freeze.json"),
        (parent, "freeze.json"), (stage_source, "calibration-freeze.json"),
    ):
        if protocol["input_bindings"][str(folder.resolve())] != sha(folder / manifest):
            raise ValueError("Bound input changed")
    if sha(encoded / "encoding-freeze.json") != read_json(run / "training-plan.json")["encoding_manifest_sha256"]:
        raise ValueError("Classifier uses another encoding population")
    out.mkdir(parents=True, exist_ok=False)
    split = protocol["split"]
    banks = load_banks(encoded, split)
    training_numbers = []
    sampled_charts = {}
    for cid, bank in banks.items():
        with np.load(evidence_source / "evidence" / (cid + ".npz")) as archive:
            evidence = {key: archive[key] for key in archive.files}
        case, features = load_case_features(data, cid)
        values = np.array([
            row if row is not None else [np.nan] * 3 for row in case.displacement_mm
        ])
        rebuilt_evidence = motion_mass_descriptors(values, features.noise)
        for name in ("blocks", "edge_available", "global_distribution", "raw_profile"):
            np.testing.assert_array_equal(rebuilt_evidence[name], evidence[name])
        indices = np.flatnonzero(bank["selected"])
        selected = indices[[0, len(indices) // 2, -1]]
        images = raw_chart_patches(case, features.noise)
        sampled_charts[cid] = [render_encoder_chart(images[index]) for index in selected]
        continuous, jumps = continuity_guides(evidence)
        for name, values in (
            ("continuity_guide", continuous), ("jump_guide", jumps),
            ("continuity_weights", feature_weights(continuous)),
            ("permuted_weights", feature_weights(continuous, permuted=True)),
            ("jump_weights", feature_weights(jumps)),
            ("shared_numbers", np.column_stack((
                bank["numbers"], evidence["global_distribution"],
                continuous.reshape(len(continuous), 24), jumps.reshape(len(jumps), 24),
            ))),
        ):
            np.testing.assert_array_equal(values, bank[name])
        if cid in split["train"]:
            training_numbers.append(bank["shared_numbers"][bank["selected"]])
    scaling = fit_numeric_scaling(np.concatenate(training_numbers))
    if scaling != read_json(encoded / "numeric-scaling.json"):
        raise ValueError("TRAIN-only shared numeric scaling differs")
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    if sha(weights) != protocol["weights_sha256"]:
        raise ValueError("Frozen pretrained visual weights differ")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained=None, image_mean=config["mean"], image_std=config["std"],
        image_interpolation=config["interpolation"], image_resize_mode=config["resize_mode"],
    )
    encoder.visual.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    maximum_error, samples = 0., 0
    for cid, bank in banks.items():
        indices = np.flatnonzero(bank["selected"])
        selected = indices[[0, len(indices) // 2, -1]]
        tensor = torch.stack([transform(chart) for chart in sampled_charts[cid]])
        with torch.inference_mode():
            maps = encoder.visual.trunk.forward_features(tensor)
            for destination, name in (
                ("continuity_embedding", "continuity_weights"),
                ("permuted_embedding", "permuted_weights"),
                ("jump_embedding", "jump_weights"),
            ):
                rebuilt = evidence_weighted_embedding(
                    maps, encoder.visual.trunk.head.fc, bank[name][selected]
                ).numpy()
                expected = bank[destination][selected]
                maximum_error = max(maximum_error, float(np.max(np.abs(rebuilt - expected))))
                if not np.allclose(rebuilt, expected, atol=1e-6, rtol=1e-5):
                    raise ValueError("Sampled weighted visual encoding differs")
                samples += len(selected)
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    predictions = read_json(parent / "predictions.json")
    saved = read_json(run / "probabilities.json")
    choices = read_json(run / "selection.json")
    for name, choice in choices.items():
        if name in ARMS:
            inputs = {cid: pooling_inputs(banks[cid], ARMS[name], scaling) for cid in labels}
            model = pretrained_activity_head(next(iter(inputs.values())).shape[1], 0, choice["parameter"])
            model.load_state_dict(torch.load(run / f"{name}-width{choice['parameter']}.pt",
                                             map_location="cpu", weights_only=True), strict=True)
            probability = {cid: probabilities(model, inputs[cid]) for cid in labels}
        else:
            model = XGBClassifier()
            model.load_model(run / f"{name}-depth{choice['parameter']}.json")
            probability = {cid: model.predict_proba(numeric_information(banks[cid], name))[:, 1] for cid in labels}
        for cid in labels:
            np.testing.assert_array_equal(probability[cid], np.array(saved[name][cid]))
        predictions[name] = {
            cid: decode(probability[cid], choice["threshold"], banks[cid], codes[cid]).to_dict()
            for cid in labels
        }
    if predictions != read_json(run / "predictions.json"):
        raise ValueError("Selected new or retained control outputs differ")
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or {
        row["method"]: row for row in summary
    } != {row["method"]: row for row in read_json(run / "summary.json")}:
        raise ValueError("Unified activity/event/boundary/stage scores differ")
    write_json(out / "verification.json", dict(
        status="VERIFIED", original_records=len(banks), exact_guides_scaling=True,
        sampled_auxiliary_encodings=samples, maximum_auxiliary_encoding_error=maximum_error,
        selected_models=len(choices),
        probabilities=sum(len(values) for by_case in saved.values() for values in by_case.values()),
        methods=len(predictions), record_outputs=len(predictions) * len(labels),
        exact_probabilities_outputs_scores=True, classifier_retraining=False,
        reserved_content_loaded=False, run_manifest_sha256=sha(run / "freeze.json"),
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "evidence-source", "parent", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.data.resolve(), args.encoded.resolve(), args.evidence_source.resolve(),
               args.parent.resolve(), args.run.resolve(), args.out.resolve())
