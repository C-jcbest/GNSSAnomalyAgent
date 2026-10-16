"""Replay the dated-motion probe using its frozen selected models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_dated_evidence import ARMS
from probe_landslide_nuisance_evidence import ROOT
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from threadpoolctl import threadpool_limits
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_dated_evidence import DatedEvidenceHead, dated_inputs, original_edges
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input


def verify(data, encoded, evidence_source, parent, run, out):
    for folder, filename in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
        (evidence_source, "freeze.json"), (parent, "freeze.json"), (run, "freeze.json"),
    ):
        verify_manifest(folder, filename)
    protocol = read_json(run / "protocol.json")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    for folder, filename in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
        (evidence_source, "freeze.json"), (parent, "freeze.json"),
    ):
        if sha(folder / filename) != protocol["input_bindings"][str(folder.resolve())]:
            raise ValueError("Bound input changed: " + str(folder))
    out.mkdir(parents=True, exist_ok=False)
    split = protocol["split"]
    banks = load_banks(encoded, split)
    evidence = {}
    for cid in banks:
        with np.load(evidence_source / "evidence" / (cid + ".npz")) as archive:
            evidence[cid] = {key: archive[key] for key in archive.files}
    training = np.concatenate([
        original_edges(evidence[cid])[banks[cid]["selected"]] for cid in split["train"]
    ])
    edge_scaling = fit_numeric_scaling(training)
    if edge_scaling != read_json(run / "edge-scaling.json"):
        raise ValueError("TRAIN-only edge scaling differs")
    scaling = read_json(parent / "numeric-scaling.json")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(data.parent / "disagreement-calibration-v1/stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    original_probability = read_json(run / "probabilities.json")
    choices = read_json(run / "selection.json")
    predictions = read_json(parent / "predictions.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for name, kind in ARMS.items():
        choice = choices[name]
        model = DatedEvidenceHead(choice["parameter"], kind)
        model.load_state_dict(
            torch.load(run / f"{name}-width{choice['parameter']}.pt",
                       map_location="cpu", weights_only=True),
            strict=True,
        )
        predictions[name] = {}
        for cid in labels:
            bank = banks[cid]
            numbers = numeric_input(
                np.column_stack((bank["numbers"], evidence[cid]["global_distribution"])), scaling
            )
            inputs = dated_inputs(evidence[cid], edge_scaling, numbers, bank["embeddings"], kind)
            probability = probabilities(model, inputs)
            np.testing.assert_array_equal(probability, np.array(original_probability[name][cid]))
            predictions[name][cid] = decode(probability, choice["threshold"], bank, codes[cid]).to_dict()
    if predictions != read_json(run / "predictions.json"):
        raise ValueError("New or original selected outputs differ")
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or {
        row["method"]: row for row in summary
    } != {row["method"]: row for row in read_json(run / "summary.json")}:
        raise ValueError("Unified scores differ")
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED", original_records=len(banks), selected_models=len(ARMS),
            probabilities=sum(len(values) for by_case in original_probability.values() for values in by_case.values()),
            methods=len(predictions), record_outputs=len(predictions) * len(labels),
            exact_train_edge_scaling=True, exact_probabilities_outputs_scores=True,
            classifier_retraining=False, new_visual_encodings=0, reserved_content_loaded=False,
            run_manifest_sha256=sha(run / "freeze.json"),
        ),
    )
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
