"""Compare target-anchored companding with applicable same-split strong controls."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_chart_companding import ARMS, EXISTING_CONTROLS
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities, train_head
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import numeric_input


def companded_inputs(bank, arm, scaling):
    key = "embeddings" if arm["view"] == "original" else arm["view"] + "_embeddings"
    if arm["numeric"]:
        return np.column_stack((bank[key], numeric_input(bank["numbers"], scaling))).astype(
            np.float32
        )
    return bank[key]


def calibrate(data, encoded, previous, out):
    for folder, manifest in (
        (encoded, "encoding-freeze.json"),
        (previous, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    if sha(previous / "calibration-freeze.json") != protocol["original_control_manifest_sha256"]:
        raise ValueError("Original same-split controls changed")
    if sha(data / "data-freeze.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Data changed")
    split = protocol["split"]
    # The exact original cache is resolved from the new encoder's immutable
    # parent manifest, rather than trusting a populated old-population model.
    original_folder = Path(protocol["original_encoding_path"])
    if (
        sha(original_folder / "encoding-freeze.json")
        != protocol["original_encoding_manifest_sha256"]
    ):
        raise ValueError("Exact same-split cache is unavailable")
    if read_json(original_folder / "protocol.json")["split"] != split:
        raise ValueError("Baseline and candidate splits differ")
    if read_json(original_folder / "numeric-scaling.json") != read_json(
        encoded / "numeric-scaling.json"
    ):
        raise ValueError("Baseline and candidate numerical normalization differs")
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "training-plan.json",
        {
            "protocol_sha256": sha(encoded / "protocol.json"),
            "encoding_manifest_sha256": sha(encoded / "encoding-freeze.json"),
            "previous_manifest_sha256": sha(previous / "calibration-freeze.json"),
            "thresholds": THRESHOLDS,
            "widths": [16, 32],
            "depths": [2, 4],
            "seed": 20261020,
            "epochs": 12,
            "arms": ARMS,
            "reuse": "Seven exact same16/8 controls under same common decoder; no incompatible fitted model reuse",
            "test_truth_opened": False,
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(previous / "stage-xgboost.json", out / "stage-xgboost.json")
    write_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    banks = {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            with np.load(original_folder / group / (cid + ".npz")) as original:
                for key in (
                    "embeddings",
                    "numbers",
                    "selected",
                    "supported",
                    "observed",
                    "activity",
                    "stage_supported",
                    "stage_matrix",
                    "chart_descriptors",
                ):
                    if not np.array_equal(banks[cid][key], original[key], equal_nan=True):
                        raise ValueError("Same-split control inputs changed: " + cid + ":" + key)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(out / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    scaling = read_json(encoded / "numeric-scaling.json")
    actual = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    trials, choices, predictions, saved_probability, costs = [], {}, {}, {}, []

    def evaluate(name, parameter, probability):
        candidates = {}
        for threshold in THRESHOLDS:
            outputs = {
                cid: decode(probability[cid], threshold, banks[cid], codes[cid])
                for cid in split["validation"]
            }
            score = aggregate_scores([evaluate_case(labels[cid], outputs[cid]) for cid in outputs])
            trials.append(
                {
                    "method": name,
                    "parameter": parameter,
                    "threshold": threshold,
                    "objective": calibration_objective(score),
                    "validation": score,
                }
            )
            candidates[threshold] = {cid: output.to_dict() for cid, output in outputs.items()}
        return candidates

    def select(name, candidates, probabilities_by_parameter):
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidates[choice["parameter"]][choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist()
            for cid, value in probabilities_by_parameter[choice["parameter"]].items()
        }

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plain = {"augment": False, "pair": False, "shuffle": False}
    for name, arm in ARMS.items():
        inputs = {cid: companded_inputs(bank, arm, scaling) for cid, bank in banks.items()}
        training = np.concatenate([inputs[cid][banks[cid]["selected"]] for cid in split["train"]])
        candidates, by_width = {}, {}
        for width in (16, 32):
            empty = np.empty((0, training.shape[1]), dtype=np.float32)
            model, history, seconds, samples = train_head(
                training, actual, empty, empty, plain, width
            )
            torch.save(model.state_dict(), out / (name + f"-width{width}.pt"))
            write_json(out / (name + f"-width{width}-loss.json"), history)
            probability = {cid: probabilities(model, inputs[cid]) for cid in split["validation"]}
            candidates[width] = evaluate(name, width, probability)
            by_width[width] = probability
            costs.append(
                {
                    "method": name,
                    "parameter": width,
                    "seconds": seconds,
                    "training_samples": samples,
                    "input_dimensions": training.shape[1],
                    "parameters": sum(value.numel() for value in model.parameters()),
                }
            )
            print("trained", name, width, flush=True)
        select(name, candidates, by_width)
    for name, view in (
        ("xgb_target_chart384", "target"),
        ("xgb_global_companded_chart384", "global"),
    ):
        matrices = {
            cid: np.column_stack((bank["numbers"], bank[view + "_descriptors"]))
            for cid, bank in banks.items()
        }
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        actual_all = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
        eligible = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (
            actual_all >= 0
        )
        candidates, by_depth = {}, {}
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training[eligible], actual_all[eligible])
            model.save_model(out / (name + f"-depth{depth}.json"))
            probability = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            candidates[depth] = evaluate(name, depth, probability)
            by_depth[depth] = probability
            costs.append(
                {
                    "method": name,
                    "parameter": depth,
                    "seconds": time.perf_counter() - started,
                    "training_samples": int(eligible.sum()),
                    "input_dimensions": training.shape[1],
                    "parameters": None,
                }
            )
        select(name, candidates, by_depth)
    old_predictions = read_json(previous / "predictions.json")
    old_probability = read_json(previous / "probabilities.json")
    old_choices = read_json(previous / "selection.json")
    for name in EXISTING_CONTROLS:
        predictions[name] = old_predictions[name]
        saved_probability[name] = old_probability[name]
        choices[name] = old_choices[name]
    write_json(out / "selection.json", choices)
    write_json(out / "validation-trials.json", trials)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", costs)
    valid_labels = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in valid_labels}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]])
        for cid in valid_labels
    }
    results, summary, per_case = score_target(valid_labels, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["case_id"] in valid_labels
    }
    negative = [
        cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}
    ]
    mixed = [
        cid
        for cid, family in families.items()
        if family in {"deformation_step", "deformation_spike"}
    ]

    def subset(name, ids):
        return aggregate_scores([results[name]["per_case"][cid] for cid in ids])["daily"]

    main_name = "target_companded_numeric"
    main = table[main_name]
    controls = ["global_numeric", "global_companded_numeric"]
    net = {
        name: table[name]["fp"] + table[name]["fn"] - main["fp"] - main["fn"] for name in controls
    }
    checks = {
        "daily_gain_both": all(main["f1"] > table[name]["f1"] for name in controls),
        "event_nondecrease_both": all(
            main["event_f1_iou03"] >= table[name]["event_f1_iou03"] for name in controls
        ),
        "net_correction_both": min(net.values()) > 0,
        "nuisance_fp_nondecrease_avoided": subset(main_name, negative)["fp"]
        <= subset("global_numeric", negative)["fp"],
        "mixed_recall_nondecrease": subset(main_name, mixed)["recall"]
        >= subset("global_numeric", mixed)["recall"],
        "visual_value": main["f1"] > table["numeric"]["f1"]
        or (
            main["event_f1_iou03"] > table["numeric"]["event_f1_iou03"]
            and main["f1"] >= table["numeric"]["f1"] - 0.01
        ),
        "numeric_feedback_value": main["f1"] > table["target_companded_visual"]["f1"],
    }
    decision = {
        "decision": "PROCEED_COMPANDING_FULL_CALIBRATION"
        if all(checks.values())
        else "STOP_COMPANDING_PILOT",
        "checks": checks,
        "net_correct": net,
        "negative_fp": {name: subset(name, negative)["fp"] for name in [main_name, *controls]},
        "mixed_recall": {name: subset(name, mixed)["recall"] for name in [main_name, *controls]},
        "test_truth_opened": False,
        "new_api_calls": 0,
        "scope": "Exposed validation; no independent contribution claim or choice of best ablation as replacement",
    }
    write_json(out / "decision.json", decision)
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    write_json(
        out / "calibration-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "previous", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    calibrate(
        args.data.resolve(), args.encoded.resolve(), args.previous.resolve(), args.out.resolve()
    )
