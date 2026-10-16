"""Calibrate a frozen spatial-readout pilot and strong same-split controls."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import ARMS, verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities, train_head
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_evaluation import STAGES, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_visual_invariance import numeric_input


def readout_inputs(bank, arm, scaling):
    numbers = numeric_input(bank["numbers"], scaling)
    if arm == "numeric":
        return numbers
    parts = [bank["embeddings"]]
    if arm == "duplicate":
        parts.append(bank["embeddings"])
    elif arm == "center":
        parts.append(bank["center"])
    elif arm in {"contrast", "visual"}:
        parts.append(bank["contrast"])
    if arm != "visual":
        parts.append(numbers)
    return np.column_stack(parts).astype(np.float32)


def decode(probability, threshold, bank, codes, support=None):
    support = bank["supported"] if support is None else support
    activity = activity_across_missing(probability >= threshold, support, bank["observed"])
    return learned_stages(codes, activity, bank["stage_supported"])


def calibrate(data, encoded, out):
    verify_manifest(encoded, "encoding-freeze.json")
    protocol = read_json(encoded / "protocol.json")
    if sha(data / "data-freeze.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    verify_manifest(data, "data-freeze.json")
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    split = protocol["split"]
    write_json(
        out / "training-plan.json",
        {
            "encoding_manifest_sha256": sha(encoded / "encoding-freeze.json"),
            "protocol_sha256": sha(encoded / "protocol.json"),
            "seed": 20261020,
            "epochs": 12,
            "widths": [16, 32],
            "depths": [2, 4],
            "thresholds": THRESHOLDS,
            "arms": ARMS,
            "postprocessing": "Common observation-duration decoder; no missing imputation",
            "test_truth_opened": False,
            "new_api_calls": 0,
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    banks, features = {}, {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            _, features[cid] = load_case_features(data, cid)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    matrix = np.concatenate([features[cid].matrix for cid in split["train"]])
    support = np.concatenate([features[cid].supported for cid in split["train"]])
    stage_labels = np.array(
        [STAGES.get(row["feature"], -1) for cid in split["train"] for row in labels[cid]]
    )
    eligible = support & (stage_labels > 0)
    if set(stage_labels[eligible]) != {1, 2, 3}:
        raise ValueError("Pilot training must support all three auxiliary stage classes")
    stage = XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.05,
        n_jobs=2,
        random_state=20261007,
        objective="multi:softprob",
        num_class=3,
    )
    stage.fit(matrix[eligible], stage_labels[eligible] - 1)
    stage.save_model(out / "stage-xgboost.json")
    codes = {
        cid: stage.predict(features[cid].matrix).astype(int) + 1 for cid in split["validation"]
    }
    trials, selections, predictions, probability_bank, costs = [], {}, {}, {}, []

    def evaluate(name, parameter, probability, thresholds=THRESHOLDS, supports=None):
        candidates = {}
        for threshold in thresholds:
            outputs = {
                cid: decode(
                    probability[cid],
                    threshold,
                    banks[cid],
                    codes[cid],
                    None if supports is None else supports[cid],
                )
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
            candidates[threshold] = {
                cid: prediction.to_dict() for cid, prediction in outputs.items()
            }
        return candidates

    def select(name, candidates, probabilities_by_parameter):
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        selections[name] = choice
        predictions[name] = candidates[choice["parameter"]][choice["threshold"]]
        probability_bank[name] = {
            cid: value.tolist()
            for cid, value in probabilities_by_parameter[choice["parameter"]].items()
        }

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scaling = read_json(encoded / "numeric-scaling.json")
    actual = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    plain = {"augment": False, "pair": False, "shuffle": False}
    for name, arm in ARMS.items():
        inputs = {cid: readout_inputs(bank, arm, scaling) for cid, bank in banks.items()}
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
                    "trainable_parameters": sum(value.numel() for value in model.parameters()),
                }
            )
            print("trained", name, width, f"{seconds:.2f}s", flush=True)
        select(name, candidates, by_width)
    for name, kind in (("xgb_raw12", "raw12"), ("xgb_raw84", "raw84"), ("xgb_chart384", "chart")):
        matrices = {}
        supports = {}
        for cid, bank in banks.items():
            if kind == "raw12":
                matrices[cid] = bank["stage_matrix"]
                supports[cid] = bank["stage_supported"]
            elif kind == "raw84":
                matrices[cid] = bank["numbers"]
                supports[cid] = bank["supported"]
            else:
                matrices[cid] = np.column_stack((bank["numbers"], bank["chart_descriptors"]))
                supports[cid] = bank["supported"]
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        labels_train = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
        eligible = np.concatenate([supports[cid] for cid in split["train"]]) & (labels_train >= 0)
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
            model.fit(training[eligible], labels_train[eligible])
            model.save_model(out / (name + f"-depth{depth}.json"))
            probability = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            candidates[depth] = evaluate(name, depth, probability, supports=supports)
            by_depth[depth] = probability
            costs.append(
                {
                    "method": name,
                    "parameter": depth,
                    "seconds": time.perf_counter() - started,
                    "training_samples": int(eligible.sum()),
                    "input_dimensions": training.shape[1],
                    "trainable_parameters": None,
                }
            )
        select(name, candidates, by_depth)
    # Reuse original robust and PELT evidence, with calibration under the same
    # decoder. Their evidence coverage may differ and is preserved explicitly.
    for name in ("robust_rule", "pelt_rule"):
        candidates, by_parameter = {}, {}
        settings = (1.0, 2.0, 3.0) if name == "robust_rule" else (10.0, 40.0, 160.0)
        for setting in settings:
            probability = {}
            supports = {}
            for cid in split["validation"]:
                feature = features[cid]
                if name == "robust_rule":
                    probability[cid] = np.where(
                        feature.signal_to_noise >= setting, feature.progressive_rate, -np.inf
                    )
                    supports[cid] = feature.supported
                else:
                    rates = pelt_segment_rates(feature, setting)
                    probability[cid] = np.minimum(rates, 2 * feature.progressive_rate)
                    probability[cid][feature.signal_to_noise < 1] = -np.inf
                    supports[cid] = feature.supported & np.isfinite(rates)
            candidates[setting] = evaluate(
                name, setting, probability, thresholds=(0.03, 0.06, 0.12), supports=supports
            )
            # Save finite scores plus an explicit unavailable marker for replay.
            by_parameter[setting] = {
                cid: np.where(np.isfinite(value), value, -1.0) for cid, value in probability.items()
            }
        select(name, candidates, by_parameter)
    write_json(out / "selection.json", selections)
    write_json(out / "validation-trials.json", trials)
    write_json(out / "probabilities.json", probability_bank)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", costs)
    validation_labels = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation_labels}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]])
        for cid in validation_labels
    }
    results, summary, per_case = score_target(
        validation_labels, observed, weak, predictions, {}, {}
    )
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in ledger
        if row["case_id"] in validation_labels
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

    main = table["global_contrast_numeric"]
    duplicate = table["duplicate_global_numeric"]
    center = table["global_center_numeric"]
    numeric = table["numeric"]
    net = {
        name: row["fp"] + row["fn"] - main["fp"] - main["fn"]
        for name, row in (("duplicate", duplicate), ("center", center))
    }
    checks = {
        "gain_over_duplicate": main["f1"] > duplicate["f1"],
        "gain_over_center": main["f1"] > center["f1"],
        "event_nondecrease": main["event_f1_iou03"]
        >= max(duplicate["event_f1_iou03"], center["event_f1_iou03"]),
        "positive_net_correction_both": min(net.values()) > 0,
        "negative_fp_nondecrease_avoided": subset("global_contrast_numeric", negative)["fp"]
        <= subset("duplicate_global_numeric", negative)["fp"],
        "mixed_recall_nondecrease": subset("global_contrast_numeric", mixed)["recall"]
        >= subset("duplicate_global_numeric", mixed)["recall"],
        "visual_value": main["f1"] > numeric["f1"]
        or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"]
            and main["f1"] >= numeric["f1"] - 0.01
        ),
    }
    decision = {
        "decision": "PROCEED_SPATIAL_FULL_CALIBRATION"
        if all(checks.values())
        else "STOP_SPATIAL_READOUT_PILOT",
        "checks": checks,
        "net_correct": net,
        "negative_fp": {name: subset(name, negative)["fp"] for name in ARMS},
        "mixed_recall": {name: subset(name, mixed)["recall"] for name in ARMS},
        "test_truth_opened": False,
        "new_api_calls": 0,
        "interpretation": "Selected exposed validation pilot; not an independent performance conclusion",
    }
    write_json(out / "decision.json", decision)
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed during training: " + relative)
    write_json(
        out / "calibration-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.data.resolve(), args.encoded.resolve(), args.out.resolve())
