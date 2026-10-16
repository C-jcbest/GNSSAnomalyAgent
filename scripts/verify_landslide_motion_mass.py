"""Recompute raw movement evidence and exactly replay every selected probe output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_motion_mass import matrices_for_arm
from probe_landslide_nuisance_evidence import ROOT
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_motion_mass import motion_mass_descriptors


def verify(data, encoded, parent, run, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (parent, "freeze.json"),
        (run, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(run / "protocol.json")
    if protocol["encoded_manifest_sha256"] != sha(encoded / "encoding-freeze.json"):
        raise ValueError("Original encoding changed")
    if protocol["parent_manifest_sha256"] != sha(parent / "freeze.json"):
        raise ValueError("Original controls changed")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen evidence source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    banks = load_banks(encoded, protocol["split"])
    evidence = {}
    for cid in banks:
        case, features = load_case_features(data, cid)
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        evidence[cid] = motion_mass_descriptors(raw, features.noise)
        with np.load(run / "evidence" / (cid + ".npz")) as archive:
            for key, values in evidence[cid].items():
                np.testing.assert_array_equal(values, archive[key])
    labels = {cid: [] for cid in protocol["split"]["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(parent / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    choices = read_json(run / "selection.json")
    saved_probabilities = read_json(run / "probabilities.json")
    saved_predictions = read_json(run / "predictions.json")
    predictions = {}
    for name, choice in choices.items():
        matrices = matrices_for_arm(banks, evidence, name)
        model = XGBClassifier()
        model.load_model(run / f"{name}-depth{choice['parameter']}.json")
        probability = {cid: model.predict_proba(matrices[cid])[:, 1] for cid in labels}
        for cid in labels:
            np.testing.assert_array_equal(
                probability[cid], np.array(saved_probabilities[name][cid])
            )
        predictions[name] = {
            cid: decode(probability[cid], choice["threshold"], banks[cid], codes[cid]).to_dict()
            for cid in labels
        }
        if predictions[name] != saved_predictions[name]:
            raise ValueError("Selected output changed: " + name)
    parent_predictions = read_json(parent / "predictions.json")
    if any(saved_predictions[name] != cases for name, cases in parent_predictions.items()):
        raise ValueError("Reused controls changed")
    predictions.update(parent_predictions)
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Unified per-case and aggregate results changed")
    # JSON stores selection keys in sorted order. Compare the same complete
    # method collection without treating its presentation order as a score.
    replayed_summary = {row["method"]: row for row in summary}
    original_summary = {row["method"]: row for row in read_json(run / "summary.json")}
    if replayed_summary != original_summary:
        raise ValueError("Unified method summaries changed")
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED",
            evidence_records=len(evidence),
            selected_models=len(choices),
            probabilities=len(choices) * len(labels) * 730,
            methods=len(predictions),
            record_outputs=len(predictions) * len(labels),
            exact_evidence_probability_activity_stage_score_replay=True,
            raw84_refit_identical=True,
            no_training=True,
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
    verify(
        args.data.resolve(),
        args.encoded.resolve(),
        args.parent.resolve(),
        args.run.resolve(),
        args.out.resolve(),
    )
