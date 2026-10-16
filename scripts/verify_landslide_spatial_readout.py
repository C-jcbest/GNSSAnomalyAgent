"""Replay spatial pilot probabilities, common decoders and unified scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import ARMS, verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode, readout_inputs
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_paired_persistence import pretrained_activity_head


def replay_missing_diagnostic(base):
    diagnostic = base / "missing-duration-probe-v1"
    verify_manifest(diagnostic, "freeze.json")
    existing = base / "quality-encoding-v1"
    saved = read_json(diagnostic / "predictions.json")
    ids = read_json(existing / "protocol.json")["split"]["validation"]
    stage = XGBClassifier()
    stage.load_model(base / "quality-calibration-v1" / "stage-xgboost.json")
    count = 0
    for prefix, folder in (
        ("dq", base / "quality-calibration-v1"),
        ("paired", base / "diverse-paired-calibration-v1"),
    ):
        probability_bank = read_json(folder / "probabilities.json")
        choices = read_json(folder / "selection.json")
        for name, cases in probability_bank.items():
            for cid in ids:
                with np.load(existing / "validation" / (cid + ".npz")) as archive:
                    bank = {key: archive[key] for key in archive.files}
                probability = cases[cid]["activity"] if prefix == "dq" else cases[cid]
                activity = activity_across_missing(
                    np.array(probability) >= choices[name]["threshold"],
                    bank["supported"],
                    bank["observed"],
                )
                codes = stage.predict(bank["stage_matrix"]).astype(int) + 1
                output = learned_stages(codes, activity, bank["stage_supported"]).to_dict()
                if output != saved[prefix + ":" + name + ":raw_missing_link"][cid]:
                    raise ValueError("Common duration decoder differs from original diagnostic")
                count += 1
    return count


def verify(data, encoded, run, out):
    verify_manifest(encoded, "encoding-freeze.json")
    verify_manifest(run, "calibration-freeze.json")
    protocol = read_json(encoded / "protocol.json")
    split = protocol["split"]
    scaling = read_json(encoded / "numeric-scaling.json")
    choices = read_json(run / "selection.json")
    saved_probability = read_json(run / "probabilities.json")
    saved_predictions = read_json(run / "predictions.json")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(run / "stage-xgboost.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    banks, features, codes = {}, {}, {}
    for cid in labels:
        with np.load(encoded / "validation" / (cid + ".npz")) as archive:
            banks[cid] = {key: archive[key] for key in archive.files}
        _, features[cid] = load_case_features(data, cid)
        codes[cid] = stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
    predictions = {}
    checks = []
    for name, choice in choices.items():
        predictions[name] = {}
        if name in ARMS:
            dimensions = readout_inputs(next(iter(banks.values())), ARMS[name], scaling).shape[1]
            model = pretrained_activity_head(dimensions, 0, choice["parameter"])
            model.load_state_dict(
                torch.load(run / (name + f"-width{choice['parameter']}.pt"), weights_only=True)
            )
        elif name.startswith("xgb_"):
            model = XGBClassifier()
            model.load_model(run / (name + f"-depth{choice['parameter']}.json"))
        for cid, bank in banks.items():
            support = bank["supported"]
            if name in ARMS:
                probability = probabilities(model, readout_inputs(bank, ARMS[name], scaling))
            elif name.startswith("xgb_"):
                if name == "xgb_raw12":
                    values = bank["stage_matrix"]
                    support = bank["stage_supported"]
                elif name == "xgb_raw84":
                    values = bank["numbers"]
                else:
                    values = np.column_stack((bank["numbers"], bank["chart_descriptors"]))
                probability = model.predict_proba(values)[:, 1]
            elif name == "robust_rule":
                feature = features[cid]
                support = feature.supported
                probability = np.where(
                    feature.signal_to_noise >= choice["parameter"], feature.progressive_rate, -1.0
                )
            else:
                feature = features[cid]
                rates = pelt_segment_rates(feature, choice["parameter"])
                support = feature.supported & np.isfinite(rates)
                probability = np.minimum(rates, 2 * feature.progressive_rate)
                probability[feature.signal_to_noise < 1] = -1.0
                probability[~np.isfinite(probability)] = -1.0
            if not np.array_equal(probability, np.array(saved_probability[name][cid])):
                raise ValueError("Probability replay differs: " + name + ":" + cid)
            output = decode(probability, choice["threshold"], bank, codes[cid], support).to_dict()
            if output != saved_predictions[name][cid]:
                raise ValueError("Output replay differs: " + name + ":" + cid)
            activity = np.array(output["activity"])
            phase = np.array(output["stage"])
            if np.any(activity[~bank["observed"]] != -1) or np.any(phase[~bank["observed"]] != -1):
                raise ValueError("Missing days were assigned predictions")
            if np.any((phase > 0) & (activity != 1)):
                raise ValueError("A definite stage escaped its activity gate")
            predictions[name][cid] = output
            checks.append(
                {
                    "method": name,
                    "case_id": cid,
                    "probabilities": len(probability),
                    "exact_probability_replay": True,
                    "exact_prediction_replay": True,
                }
            )
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, per_case = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Unified result replay differs")
    if {row["method"]: row for row in summary} != {
        row["method"]: row for row in read_json(run / "summary.json")
    }:
        raise ValueError("Summary replay differs")
    diagnostic_replays = replay_missing_diagnostic(data.parent)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "replayed-model-cases.csv", checks)
    write_json(
        out / "verification.json",
        {
            "verification_status": "VERIFIED",
            "selected_models": 9,
            "methods": len(predictions),
            "method_case_predictions": len(checks),
            "probabilities_replayed": sum(row["probabilities"] for row in checks),
            "original_diagnostic_decoder_model_cases_replayed": diagnostic_replays,
            "all_activity_stage_scores_exact": True,
            "encoding_manifest_sha256": sha(encoded / "encoding-freeze.json"),
            "training_manifest_sha256": sha(run / "calibration-freeze.json"),
            "test_truth_opened": False,
            "no_training_or_api_calls": True,
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.data.resolve(), args.encoded.resolve(), args.run.resolve(), args.out.resolve())
