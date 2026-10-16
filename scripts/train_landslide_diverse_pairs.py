"""Compare established PP-VN and ablations on the wider deformation population."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS, prediction
from train_landslide_paired_persistence import ARMS, modality_inputs, probabilities, train_head
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import numeric_input


def calibrate(data, encoding, pairs, previous, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoding, "encoding-freeze.json"),
        (pairs, "encoding-freeze.json"),
        (previous, "calibration-freeze.json"),
    ):
        for relative, expected in read_json(folder / manifest).items():
            if sha(folder / relative) != expected:
                raise ValueError("Upstream data changed: " + relative)
    if read_json(pairs / "protocol.json")["encoding_manifest_sha256"] != sha(
        encoding / "encoding-freeze.json"
    ):
        raise ValueError("Paired supervision belongs to different original encodings")
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(encoding / "protocol.json")["split"]
    write_json(
        out / "training-plan.json",
        {
            "parent_protocol_sha256": sha(pairs / "protocol.json"),
            "arms": ARMS,
            "training_implementation": "Reuse established train_landslide_paired_persistence.train_head; fixed seed20261020,12epochs,margin1/coefficient1",
            "thresholds": THRESHOLDS,
            "widths": [16, 32],
            "tree_depths": [2, 4],
            "original_controls": "All12 previously completed same48/24 DQ controls retained; PP-VN plain is separately refit with identical PP initialization and architecture",
            "trees": "Both same-augmentation raw84/chart384 use full known supported training dates plus the exact same new negative centers; stronger original-date coverage than stride3 neural arms",
            "test_truth_opened": False,
        },
    )
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/train_landslide_deformation_quality.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
    ]
    for source in sources:
        destination = out / "source" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    banks = {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoding / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
    with np.load(pairs / "pair-bank.npz") as archive:
        pair_bank = {key: archive[key] for key in archive.files}
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(previous / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    scaling = read_json(encoding / "numeric-scaling.json")
    original_numbers = np.concatenate(
        [banks[cid]["numbers"][banks[cid]["selected"]] for cid in split["train"]]
    )
    original_embeddings = np.concatenate(
        [banks[cid]["embeddings"][banks[cid]["selected"]] for cid in split["train"]]
    )
    actual = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    original_scaled = numeric_input(original_numbers, scaling)
    positive_scaled = numeric_input(pair_bank["positive_numbers"], scaling)
    negative_scaled = numeric_input(pair_bank["negative_numbers"], scaling)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    trials, choices, predictions, saved_probability, costs = [], {}, {}, {}, []

    def evaluate(name, parameter, probability):
        outputs = {}
        for threshold in THRESHOLDS:
            output = {
                cid: prediction(probability[cid], banks[cid], codes[cid], threshold)
                for cid in split["validation"]
            }
            score = aggregate_scores([evaluate_case(labels[cid], output[cid]) for cid in output])
            trials.append(
                {
                    "method": name,
                    "parameter": parameter,
                    "threshold": threshold,
                    "objective": calibration_objective(score),
                    "validation": score,
                }
            )
            outputs[threshold] = {cid: value.to_dict() for cid, value in output.items()}
        return outputs

    for name, arm in ARMS.items():
        original = modality_inputs(original_embeddings, original_scaled, arm)
        positive = modality_inputs(pair_bank["positive_embeddings"], positive_scaled, arm)
        negative = modality_inputs(pair_bank["negative_embeddings"], negative_scaled, arm)
        candidate, probability_bank = {}, {}
        for width in (16, 32):
            model, history, seconds, count = train_head(
                original, actual, positive, negative, arm, width
            )
            torch.save(model.state_dict(), out / (name + f"-width{width}.pt"))
            write_json(out / (name + f"-width{width}-history.json"), history)
            probability_bank[width] = {
                cid: probabilities(
                    model,
                    modality_inputs(
                        banks[cid]["embeddings"], numeric_input(banks[cid]["numbers"], scaling), arm
                    ),
                )
                for cid in split["validation"]
            }
            candidate[width] = evaluate(name, width, probability_bank[width])
            costs.append(
                {
                    "method": name,
                    "parameter": width,
                    "training_seconds": seconds,
                    "training_centers": count,
                }
            )
            print("Trained", name, width, flush=True)
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidate[choice["parameter"]][choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in probability_bank[choice["parameter"]].items()
        }
    for name, chart in (("xgb_raw84_augmentation", False), ("xgb_chart384_augmentation", True)):
        matrices = {
            cid: np.column_stack((bank["numbers"], bank["chart_descriptors"]))
            if chart
            else bank["numbers"]
            for cid, bank in banks.items()
        }
        training = np.concatenate(
            [
                matrices[cid][banks[cid]["supported"] & (banks[cid]["activity"] >= 0)]
                for cid in split["train"]
            ]
        )
        target = np.concatenate(
            [
                banks[cid]["activity"][banks[cid]["supported"] & (banks[cid]["activity"] >= 0)]
                for cid in split["train"]
            ]
        )
        negative = (
            np.column_stack((pair_bank["negative_numbers"], pair_bank["negative_charts"]))
            if chart
            else pair_bank["negative_numbers"]
        )
        training = np.concatenate((training, negative))
        target = np.concatenate((target, np.zeros(len(negative))))
        candidate, probability_bank = {}, {}
        for depth in (2, 4):
            before = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training, target)
            model.save_model(out / (name + f"-depth{depth}.json"))
            probability_bank[depth] = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            candidate[depth] = evaluate(name, depth, probability_bank[depth])
            costs.append(
                {
                    "method": name,
                    "parameter": depth,
                    "training_seconds": time.perf_counter() - before,
                    "training_centers": len(target),
                }
            )
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidate[choice["parameter"]][choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in probability_bank[choice["parameter"]].items()
        }
        print("Trained", name, flush=True)
    predictions.update(
        {"dq_" + name: cases for name, cases in read_json(previous / "predictions.json").items()}
    )
    write_json(out / "predictions.json", predictions)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "trials.json", trials)
    write_json(out / "selection.json", choices)
    write_csv(out / "training-costs.csv", costs)
    write_json(
        out / "prediction-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    validation = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in validation
    }
    results, summary, per_case = score_target(validation, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["group"] == "validation"
    }
    main = "visual_numeric_pair"
    controls = (
        main,
        "visual_numeric_augmentation",
        "visual_numeric_plain",
        "numeric_pair",
        "visual_numeric_shuffled_pairs",
    )

    def subset(name, selected_families):
        return aggregate_scores(
            [
                results[name]["per_case"][cid]
                for cid in validation
                if families[cid] in selected_families
            ]
        )["daily"]

    negative = {name: subset(name, {"quiet", "step_only", "spike_only"})["fp"] for name in controls}
    mixed = {
        name: subset(name, {"deformation_step", "deformation_spike"})["recall"] for name in controls
    }
    table = {row["method"]: row for row in summary}
    primary, augmentation, plain, numeric, shuffled = [table[name] for name in controls]
    net = {
        name: table[name]["fp"] + table[name]["fn"] - primary["fp"] - primary["fn"]
        for name in controls[1:]
    }
    checks = {
        "gain_over_augmentation": primary["f1"] > augmentation["f1"],
        "nondecrease_vs_shuffled": primary["f1"] >= shuffled["f1"],
        "net_correct_vs_plain": net["visual_numeric_plain"] > 0,
        "net_correct_vs_augmentation": net["visual_numeric_augmentation"] > 0,
        "events_nondecrease": primary["event_f1_iou03"] >= augmentation["event_f1_iou03"],
        "nuisance_fp_nondecrease_avoided": negative[main] <= negative["visual_numeric_plain"],
        "mixed_recall_nondecrease": mixed[main] >= mixed["visual_numeric_augmentation"],
        "visual_value": primary["f1"] > numeric["f1"]
        or (
            primary["event_f1_iou03"] > numeric["event_f1_iou03"]
            and primary["f1"] >= numeric["f1"] - 0.01
        ),
    }
    decision = {
        "decision": "PROCEED_DIVERSE_PPVN_CONFIRMATION"
        if all(checks.values())
        else "STOP_DIVERSE_PPVN_CALIBRATION",
        "checks": checks,
        "net_correct": net,
        "negative_fp": negative,
        "mixed_recall": mixed,
        "test_truth_opened": False,
        "new_api_calls": 0,
        "interpretation": "New population development extension; old narrower confirmation remains historical, not broad verification",
    }
    write_json(out / "decision.json", decision)
    for source in sources:
        if sha(source) != sha(out / "source" / source.name):
            raise ValueError("Learning source changed during execution: " + str(source))
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    calibrate(
        args.data.resolve(),
        args.encoding.resolve(),
        args.pairs.resolve(),
        args.previous.resolve(),
        args.out.resolve(),
    )
