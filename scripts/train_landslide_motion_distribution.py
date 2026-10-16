"""Fit the registered distribution-view pilot and equally informed numeric models."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_motion_distribution import ARMS, MAIN, TREES
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities, train_head
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_motion_mass import motion_mass_descriptors
from gnss_sim.landslide_visual_invariance import numeric_input


def information_matrix(bank, include_chart=False):
    parts = [bank["numbers"], bank["distribution_descriptors"], bank["raw_profile"]]
    if include_chart:
        parts.append(bank["chart_descriptors"])
    return np.column_stack(parts)


def inputs_for_arm(bank, arm, scaling, information_scaling):
    if arm == "same_numeric":
        return numeric_input(information_matrix(bank), information_scaling)
    parts = []
    if arm != "auxiliary":
        parts.append(bank["embeddings"])
    if arm == "duplicate":
        parts.append(bank["embeddings"])
    elif arm == "permuted":
        parts.append(bank["permuted"])
    else:
        parts.append(bank["distribution"])
    if arm != "visual":
        parts.append(numeric_input(bank["numbers"], scaling))
    return np.column_stack(parts).astype(np.float32)


def calibrate(data, encoded, controls, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (controls, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    if sha(controls / "calibration-freeze.json") != protocol["controls_manifest_sha256"]:
        raise ValueError("Frozen control models changed")
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "training-plan.json",
        dict(
            encoding_manifest_sha256=sha(encoded / "encoding-freeze.json"),
            controls_manifest_sha256=sha(controls / "calibration-freeze.json"),
            widths=[16, 32],
            depths=[2, 4],
            thresholds=THRESHOLDS,
            arms=ARMS,
            trees=TREES,
            test_truth_opened=False,
        ),
    )
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    split = protocol["split"]
    banks = {}
    evidence = {}
    (out / "mass-evidence").mkdir()
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            case, features = load_case_features(data, cid)
            raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
            evidence[cid] = motion_mass_descriptors(raw, features.noise)
            np.savez_compressed(out / "mass-evidence" / (cid + ".npz"), **evidence[cid])
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(controls / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    predictions = read_json(controls / "predictions.json")
    trials, selections, saved_probability, costs = [], {}, {}, []

    def evaluate_and_select(name, models):
        candidates, by_parameter = {}, {}
        for parameter, probability in models:
            by_parameter[parameter] = probability
            for threshold in THRESHOLDS:
                decoded = {
                    cid: decode(probability[cid], threshold, banks[cid], codes[cid])
                    for cid in split["validation"]
                }
                score = aggregate_scores(
                    [evaluate_case(labels[cid], decoded[cid]) for cid in decoded]
                )
                trials.append(
                    dict(
                        method=name,
                        parameter=parameter,
                        threshold=threshold,
                        objective=calibration_objective(score),
                        validation=score,
                    )
                )
                candidates[parameter, threshold] = {
                    cid: output.to_dict() for cid, output in decoded.items()
                }
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        selections[name] = choice
        predictions[name] = candidates[choice["parameter"], choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in by_parameter[choice["parameter"]].items()
        }

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scaling = read_json(encoded / "numeric-scaling.json")
    information_scaling = read_json(encoded / "information-scaling.json")
    target = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    plain = dict(augment=False, pair=False, shuffle=False)
    for name, arm in ARMS.items():
        inputs = {
            cid: inputs_for_arm(bank, arm, scaling, information_scaling)
            for cid, bank in banks.items()
        }
        training = np.concatenate([inputs[cid][banks[cid]["selected"]] for cid in split["train"]])
        models = []
        for width in (16, 32):
            empty = np.empty((0, training.shape[1]), dtype=np.float32)
            model, history, seconds, samples = train_head(
                training, target, empty, empty, plain, width
            )
            torch.save(model.state_dict(), out / f"{name}-width{width}.pt")
            write_json(out / f"{name}-width{width}-loss.json", history)
            models.append(
                (width, {cid: probabilities(model, inputs[cid]) for cid in split["validation"]})
            )
            costs.append(
                dict(
                    method=name,
                    parameter=width,
                    training_seconds=seconds,
                    samples=samples,
                    dimensions=training.shape[1],
                    trainable_parameters=sum(value.numel() for value in model.parameters()),
                )
            )
            print("trained", name, width, f"{seconds:.2f}s", flush=True)
        evaluate_and_select(name, models)
    actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
    eligible = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (actual >= 0)
    for name in TREES:
        matrices = {}
        for cid, bank in banks.items():
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
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        models = []
        for depth in (2, 4):
            before = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training[eligible], actual[eligible])
            model.save_model(out / f"{name}-depth{depth}.json")
            models.append(
                (
                    depth,
                    {cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]},
                )
            )
            costs.append(
                dict(
                    method=name,
                    parameter=depth,
                    training_seconds=time.perf_counter() - before,
                    samples=int(eligible.sum()),
                    dimensions=training.shape[1],
                    trainable_parameters=None,
                )
            )
        evaluate_and_select(name, models)
        print("calibrated", name, flush=True)
    write_json(out / "selection.json", selections)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "trials.json", trials)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", costs)
    validation = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in validation[cid]])
        for cid in validation
    }
    results, summary, per_case = score_target(validation, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in validation
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

    main, duplicate, permuted, original = (
        table[name]
        for name in (
            MAIN,
            "duplicate_motion_global",
            "permuted_distribution_numeric",
            "global_numeric",
        )
    )
    net = {
        name: row["fp"] + row["fn"] - main["fp"] - main["fn"]
        for name, row in (("duplicate", duplicate), ("permuted", permuted))
    }
    numerical = [
        table[name]
        for name in (*TREES, "same_information_numeric_mlp", "numeric", "xgb_raw84", "xgb_chart384")
    ]
    best_numeric = max(numerical, key=lambda row: row["f1"])
    checks = dict(
        daily_gain=main["f1"] > max(duplicate["f1"], original["f1"]),
        ordered_gain=main["f1"] > permuted["f1"],
        event_nonloss=main["event_f1_iou03"]
        >= max(duplicate["event_f1_iou03"], original["event_f1_iou03"], permuted["event_f1_iou03"]),
        net_correction=min(net.values()) > 0,
        nuisance_nonloss=subset(MAIN, negative)["fp"]
        <= subset("duplicate_motion_global", negative)["fp"],
        mixed_nonloss=subset(MAIN, mixed)["recall"]
        >= subset("duplicate_motion_global", mixed)["recall"],
        visual_value=main["f1"] > best_numeric["f1"]
        or (
            main["event_f1_iou03"] > best_numeric["event_f1_iou03"]
            and main["f1"] >= best_numeric["f1"] - 0.01
        ),
    )
    decision = dict(
        status="PROCEED_DISTRIBUTION_FULL_CALIBRATION"
        if all(checks.values())
        else "STOP_DISTRIBUTION_VIEW_PILOT",
        checks=checks,
        net_correct=net,
        best_numeric=best_numeric["method"],
        negative_fp={name: subset(name, negative)["fp"] for name in predictions},
        mixed_recall={name: subset(name, mixed)["recall"] for name in predictions},
        test_truth_opened=False,
        new_api_calls=0,
    )
    write_json(out / "decision.json", decision)
    for relative, expected in read_json(encoded / "source-bindings.json").items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "controls", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    calibrate(
        args.data.resolve(), args.encoded.resolve(), args.controls.resolve(), args.out.resolve()
    )
