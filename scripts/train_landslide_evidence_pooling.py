"""Calibrate continuity-guided visual pooling and equally informed controls."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_evidence_pooling import ARMS, MAIN, TREES
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from threadpoolctl import threadpool_limits
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities, train_head
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import numeric_input


def pooling_inputs(bank, kind, scaling):
    numbers = numeric_input(bank["shared_numbers"], scaling)
    if kind == "numeric":
        return numbers
    parts = [bank["embeddings"]]
    if kind in {"continuity", "no_branch"}:
        parts.append(bank["continuity_embedding"])
    elif kind == "permuted":
        parts.append(bank["permuted_embedding"])
    elif kind == "jump":
        parts.append(bank["jump_embedding"])
    elif kind in {"uniform", "visual"}:
        parts.append(bank["embeddings"])
    else:
        raise ValueError("Unregistered visual pooling arm")
    if kind not in {"no_branch", "visual"}:
        parts.append(numbers)
    return np.column_stack(parts).astype(np.float32)


def numeric_information(bank, name):
    parts = [bank["shared_numbers"]]
    if name == "all_guide_information_xgb":
        parts.extend((bank["chart_descriptors"], bank["raw_profile"]))
    return np.column_stack(parts)


def calibrate(data, encoded, parent, out):
    stage_source = data.parent / "spatial-pilot-calibration-v1"
    for folder, manifest in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
        (parent, "freeze.json"), (stage_source, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(encoded / "protocol.json")
    if protocol["input_bindings"][str(parent.resolve())] != sha(parent / "freeze.json"):
        raise ValueError("Bound original controls changed")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", protocol)
    write_json(out / "training-plan.json", dict(
        encoding_manifest_sha256=sha(encoded / "encoding-freeze.json"),
        parent_manifest_sha256=sha(parent / "freeze.json"),
        stage_manifest_sha256=sha(stage_source / "calibration-freeze.json"),
        widths=[16, 32], depths=[2, 4], thresholds=THRESHOLDS,
        arms=ARMS, trees=TREES,
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    split = protocol["split"]
    banks = load_banks(encoded, split)
    scaling = read_json(encoded / "numeric-scaling.json")
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    predictions = read_json(parent / "predictions.json")
    selections, trials, saved_probability, costs = {}, [], {}, []

    def select(name, models):
        candidates, by_parameter = {}, {}
        for parameter, probability in models:
            by_parameter[parameter] = probability
            for threshold in THRESHOLDS:
                decoded = {
                    cid: decode(probability[cid], threshold, banks[cid], codes[cid])
                    for cid in split["validation"]
                }
                score = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in decoded])
                trials.append(dict(
                    method=name, parameter=parameter, threshold=threshold,
                    objective=calibration_objective(score), validation=score,
                ))
                candidates[parameter, threshold] = {cid: result.to_dict() for cid, result in decoded.items()}
        choice = max(
            (row for row in trials if row["method"] == name),
            key=lambda row: row["objective"],
        )
        selections[name] = choice
        predictions[name] = candidates[choice["parameter"], choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in by_parameter[choice["parameter"]].items()
        }

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    target = np.concatenate([banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]])
    plain = dict(augment=False, pair=False, shuffle=False)
    for name, kind in ARMS.items():
        inputs = {cid: pooling_inputs(bank, kind, scaling) for cid, bank in banks.items()}
        training = np.concatenate([inputs[cid][banks[cid]["selected"]] for cid in split["train"]])
        models = []
        for width in (16, 32):
            empty = np.empty((0, training.shape[1]), dtype=np.float32)
            model, history, seconds, samples = train_head(training, target, empty, empty, plain, width)
            torch.save(model.state_dict(), out / f"{name}-width{width}.pt")
            write_json(out / f"{name}-width{width}-loss.json", history)
            models.append((width, {
                cid: probabilities(model, inputs[cid]) for cid in split["validation"]
            }))
            costs.append(dict(
                method=name, parameter=width, seconds=seconds, samples=samples,
                dimensions=training.shape[1],
                trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
            ))
        select(name, models)
        print("calibrated", name, flush=True)
    actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
    supported = np.concatenate([banks[cid]["supported"] for cid in split["train"]])
    eligible = supported & (actual >= 0)
    for name in TREES:
        inputs = {cid: numeric_information(bank, name) for cid, bank in banks.items()}
        training = np.concatenate([inputs[cid] for cid in split["train"]])
        models = []
        for depth in (2, 4):
            before = time.perf_counter()
            model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                                  random_state=20261007, n_jobs=2)
            model.fit(training[eligible], actual[eligible])
            model.save_model(out / f"{name}-depth{depth}.json")
            models.append((depth, {
                cid: model.predict_proba(inputs[cid])[:, 1] for cid in split["validation"]
            }))
            costs.append(dict(method=name, parameter=depth, seconds=time.perf_counter() - before,
                              samples=int(eligible.sum()), dimensions=training.shape[1],
                              trainable_parameters=None))
        select(name, models)
        print("calibrated", name, flush=True)
    write_json(out / "selection.json", selections)
    write_json(out / "trials.json", trials)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", costs)
    validation = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in validation[cid]]) for cid in validation}
    results, summary, per_case = score_target(validation, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json") if row["case_id"] in validation
    }
    negatives = [cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid, family in families.items() if family in {"deformation_step", "deformation_spike"}]

    def subset(name, ids):
        return aggregate_scores([results[name]["per_case"][cid] for cid in ids])["daily"]

    controls = ("uniform_pool_numeric", "permuted_pool_numeric", "jump_pool_numeric")
    main = table[MAIN]
    net = {name: table[name]["fp"] + table[name]["fn"] - main["fp"] - main["fn"] for name in controls}
    best = max(
        (table[name] for name in (
            "guide_numeric_mlp", *TREES, "pilot_global_mass_xgb", "pilot_query_mass_xgb",
            "distribution_information_xgb", "all_distribution_information_xgb",
            "same_information_numeric_mlp", "xgb_raw84", "xgb_chart384",
        )),
        key=lambda row: row["f1"],
    )
    checks = dict(
        gain_vs_pooling_controls=main["f1"] > max(table[name]["f1"] for name in controls),
        positive_net_correction=min(net.values()) > 0,
        event_nonloss=main["event_f1_iou03"] >= max(table[name]["event_f1_iou03"] for name in controls),
        nuisance_nonloss=subset(MAIN, negatives)["fp"] <= subset(controls[0], negatives)["fp"],
        mixed_nonloss=subset(MAIN, mixed)["recall"] >= subset(controls[0], mixed)["recall"],
        visual_value=main["f1"] > best["f1"]
        or (main["event_f1_iou03"] > best["event_f1_iou03"] and main["f1"] >= best["f1"] - .01),
    )
    decision = dict(
        status="ELIGIBLE_EVIDENCE_POOLING_EXPANSION_PENDING_REPLAY" if all(checks.values()) else "STOP_EVIDENCE_POOLING_PILOT",
        checks=checks, net_correct=net, best_nonvisual=best["method"],
        nuisance_fp={name: subset(name, negatives)["fp"] for name in ARMS},
        mixed_recall={name: subset(name, mixed)["recall"] for name in ARMS},
        reserved_content_loaded=False, new_api_calls=0, stable_contribution_established=False,
    )
    write_json(out / "decision.json", decision)
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        calibrate(args.data.resolve(), args.encoded.resolve(), args.parent.resolve(), args.out.resolve())
