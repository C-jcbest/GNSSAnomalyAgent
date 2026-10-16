"""Replay the complete DQ-VN calibration and quality denominators without refitting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from run_landslide_comparison import predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import (
    ARMS,
    ROOT,
    THRESHOLDS,
    modality_inputs,
    prediction,
    probabilities,
    quality_query_scores,
)
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_deformation_quality import deformation_quality_head


def check_files(directory, manifest_name):
    for relative, expected in read_json(directory / manifest_name).items():
        if sha(directory / relative) != expected:
            raise ValueError("Frozen file changed: " + relative)


def verify(data, encoding, run, out):
    check_files(data, "data-freeze.json")
    check_files(encoding, "encoding-freeze.json")
    check_files(run, "calibration-freeze.json")
    for relative, expected in read_json(run / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Training source changed: " + relative)
    protocol = read_json(encoding / "protocol.json")
    split = protocol["split"]
    selections = read_json(run / "selection.json")
    trials = read_json(run / "validation-trials.json")
    expected_methods = set(ARMS) | {"xgb_raw84", "xgb_chart384"}
    if set(selections) != expected_methods or len(trials) != len(expected_methods) * 38:
        raise ValueError("Calibration arms or equal opportunity counts changed")
    for method in expected_methods:
        rows = [row for row in trials if row["method"] == method]
        if len(rows) != 38 or sorted(set(row["threshold"] for row in rows)) != THRESHOLDS:
            raise ValueError("Calibration grid changed")
        chosen = max(rows, key=lambda row: row["objective"])
        if chosen != selections[method]:
            raise ValueError("Selected model no longer matches the frozen validation objective")
    saved = read_json(run / "predictions.json")
    saved_probability = read_json(run / "probabilities.json")
    banks = {}
    features = {}
    for cid in split["validation"]:
        with np.load(encoding / "validation" / (cid + ".npz")) as archive:
            banks[cid] = {key: archive[key] for key in archive.files}
        _, features[cid] = load_case_features(data, cid)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    scaling = read_json(encoding / "numeric-scaling.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    stage_head = XGBClassifier()
    stage_head.load_model(run / "stage-xgboost.json")
    codes = {cid: stage_head.predict(features[cid].matrix).astype(int) + 1 for cid in banks}
    reconstructed = {}
    checked_probabilities = 0
    for name in selections:
        choice = selections[name]
        if name in ARMS:
            inputs = {
                cid: modality_inputs(bank, ARMS[name], scaling) for cid, bank in banks.items()
            }
            model = deformation_quality_head(
                next(iter(inputs.values())).shape[1], choice["parameter"]
            )
            model.load_state_dict(
                torch.load(run / (name + f"-width{choice['parameter']}.pt"), weights_only=True),
                strict=True,
            )
            pairs = {cid: probabilities(model, values) for cid, values in inputs.items()}
            if ARMS[name]["quality"]:
                for cid, pair in pairs.items():
                    if not np.array_equal(pair[1], saved_probability[name][cid]["quality"]):
                        raise ValueError("Quality probability replay differs")
                    checked_probabilities += len(pair[1])
        else:
            model = XGBClassifier()
            model.load_model(run / (name + f"-depth{choice['parameter']}.json"))
            inputs = {
                cid: np.column_stack((bank["numbers"], bank["chart_descriptors"]))
                if name == "xgb_chart384"
                else bank["numbers"]
                for cid, bank in banks.items()
            }
            pairs = {
                cid: (model.predict_proba(values)[:, 1], None) for cid, values in inputs.items()
            }
        reconstructed[name] = {}
        for cid, (activity, _) in pairs.items():
            if not np.array_equal(activity, saved_probability[name][cid]["activity"]):
                raise ValueError("Activity probability replay differs")
            checked_probabilities += len(activity)
            value = prediction(activity, banks[cid], codes[cid], choice["threshold"]).to_dict()
            if value != saved[name][cid]:
                raise ValueError("Activity/stage replay differs")
            reconstructed[name][cid] = value
    original_head = XGBClassifier()
    original_head.load_model(run / "legacy-baselines/activity-xgboost.json")
    original = predict_baselines(
        features,
        split["validation"],
        read_json(run / "legacy-baselines/frozen-selection.json"),
        original_head,
        stage_head,
    )
    for name, cases in original.items():
        values = {cid: value.to_dict() for cid, value in cases.items()}
        if values != saved["original_" + name]:
            raise ValueError("Original numerical baseline replay differs")
        reconstructed["original_" + name] = values
    observed = {cid: banks[cid]["observed"] for cid in banks}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in banks}
    results, summary, _ = score_target(labels, observed, weak, reconstructed, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Common activity/event/boundary/stage scoring replay differs")
    if {row["method"]: row for row in summary} != {
        row["method"]: row for row in read_json(run / "summary.json")
    }:
        raise ValueError("Summary replay differs")
    quality = {
        name: quality_query_scores(
            banks, {cid: values["quality"] for cid, values in saved_probability[name].items()}
        )
        for name, arm in ARMS.items()
        if arm["quality"]
    }
    if quality != read_json(run / "quality-query-summary.json"):
        raise ValueError("Auxiliary quality scoring replay differs")
    write_json(
        out,
        {
            "status": "VERIFIED",
            "models_replayed": len(reconstructed),
            "records_per_model": len(banks),
            "probability_values_replayed": checked_probabilities,
            "calibration_trials_verified": len(trials),
            "activity_event_boundary_stage_identity": True,
            "quality_probability_and_score_identity": True,
            "reserved_test_labels_opened": False,
            "new_training": False,
            "new_api_calls": 0,
        },
    )
    print(
        "Verified complete calibration, probability/prediction identity and separate quality scores"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.data.resolve(), args.encoding.resolve(), args.run.resolve(), args.out.resolve())
