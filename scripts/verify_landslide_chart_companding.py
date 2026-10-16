"""Replay chart transformations, model outputs and scores without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from encode_landslide_chart_companding import ARMS, EXISTING_CONTROLS
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_chart_companding import companded_inputs
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode, readout_inputs
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_companded_chart import companded_chart_patches, raw_window_medians
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, raw_chart_patches


def verify(data, encoded, previous, run, out):
    for folder, manifest in (
        (encoded, "encoding-freeze.json"),
        (previous, "calibration-freeze.json"),
        (run, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    split = protocol["split"]
    scaling = read_json(encoded / "numeric-scaling.json")
    choices = read_json(run / "selection.json")
    saved_probability = read_json(run / "probabilities.json")
    saved_predictions = read_json(run / "predictions.json")
    stage = XGBClassifier()
    stage.load_model(run / "stage-xgboost.json")
    banks, features, codes = {}, {}, {}
    transformation_checks = []
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                bank = {key: archive[key] for key in archive.files}
            case, feature = load_case_features(data, cid)
            medians, anchor = raw_window_medians(case)
            original = raw_chart_patches(case, feature.noise)
            for view, local in (("target", True), ("global", False)):
                patches = companded_chart_patches(
                    medians, anchor, feature.noise, target_anchored=local
                )
                descriptors = chart_numeric_descriptors(patches)
                if not np.array_equal(descriptors, bank[view + "_descriptors"], equal_nan=True):
                    raise ValueError("Original-day transformed chart descriptor replay differs")
                if local and not np.array_equal(
                    patches[feature.observed, 3:], original[feature.observed, 3:]
                ):
                    raise ValueError("Companding changed observed query column availability")
                if np.count_nonzero(bank[view + "_embeddings"][~bank["selected"]]):
                    raise ValueError("Unselected training/query centers contain encoded features")
                transformation_checks.append(
                    {
                        "case_id": cid,
                        "group": group,
                        "view": view,
                        "descriptor_values": int(descriptors.size),
                        "exact_replay": True,
                    }
                )
            if group == "validation":
                banks[cid], features[cid] = bank, feature
                codes[cid] = stage.predict(bank["stage_matrix"]).astype(int) + 1
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    predictions, model_checks = {}, []
    for name, choice in choices.items():
        predictions[name] = {}
        if name in ARMS:
            size = companded_inputs(next(iter(banks.values())), ARMS[name], scaling).shape[1]
            model = pretrained_activity_head(size, 0, choice["parameter"])
            model.load_state_dict(
                torch.load(run / (name + f"-width{choice['parameter']}.pt"), weights_only=True)
            )
        elif name in {"global_numeric", "numeric"}:
            arm = "global" if name == "global_numeric" else "numeric"
            size = readout_inputs(next(iter(banks.values())), arm, scaling).shape[1]
            model = pretrained_activity_head(size, 0, choice["parameter"])
            model.load_state_dict(
                torch.load(previous / (name + f"-width{choice['parameter']}.pt"), weights_only=True)
            )
        elif name.startswith("xgb_"):
            model = XGBClassifier()
            folder = previous if name in EXISTING_CONTROLS else run
            model.load_model(folder / (name + f"-depth{choice['parameter']}.json"))
        for cid, bank in banks.items():
            support = bank["supported"]
            if name in ARMS:
                probability = probabilities(model, companded_inputs(bank, ARMS[name], scaling))
            elif name in {"global_numeric", "numeric"}:
                probability = probabilities(
                    model,
                    readout_inputs(
                        bank, "global" if name == "global_numeric" else "numeric", scaling
                    ),
                )
            elif name.startswith("xgb_"):
                if name == "xgb_raw12":
                    values, support = bank["stage_matrix"], bank["stage_supported"]
                elif name == "xgb_raw84":
                    values = bank["numbers"]
                else:
                    descriptors = {
                        "xgb_chart384": "chart_descriptors",
                        "xgb_target_chart384": "target_descriptors",
                        "xgb_global_companded_chart384": "global_descriptors",
                    }[name]
                    values = np.column_stack((bank["numbers"], bank[descriptors]))
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
                raise ValueError("Saved score replay differs: " + name + ":" + cid)
            output = decode(probability, choice["threshold"], bank, codes[cid], support).to_dict()
            if output != saved_predictions[name][cid]:
                raise ValueError("Activity/stage replay differs: " + name + ":" + cid)
            if np.any(np.array(output["activity"])[~bank["observed"]] != -1):
                raise ValueError("Missing dates were assigned activity")
            predictions[name][cid] = output
            model_checks.append(
                {"method": name, "case_id": cid, "scores": len(probability), "exact_replay": True}
            )
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Full unified scoring replay differs")
    if {row["method"]: row for row in summary} != {
        row["method"]: row for row in read_json(run / "summary.json")
    }:
        raise ValueError("Summary replay differs")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "transformation-replay.csv", transformation_checks)
    write_csv(out / "model-case-replay.csv", model_checks)
    write_json(
        out / "verification.json",
        {
            "status": "VERIFIED",
            "selected_activity_models": 11,
            "methods": len(predictions),
            "model_cases": len(model_checks),
            "scores_replayed": sum(row["scores"] for row in model_checks),
            "original_chart_banks_replayed": len(transformation_checks),
            "descriptor_values_replayed": sum(
                row["descriptor_values"] for row in transformation_checks
            ),
            "all_activity_stage_and_unified_scores_exact": True,
            "encoding_manifest_sha256": sha(encoded / "encoding-freeze.json"),
            "calibration_manifest_sha256": sha(run / "calibration-freeze.json"),
            "new_training": False,
            "new_api_calls": 0,
            "test_truth_opened": False,
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
    for name in ("data", "encoded", "previous", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.data.resolve(),
        args.encoded.resolve(),
        args.previous.resolve(),
        args.run.resolve(),
        args.out.resolve(),
    )
