"""Freeze and test numerical-conditioned visual residuals with equal-capacity controls."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
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
from gnss_sim.landslide_conditional_visual import (
    conditional_visual_features,
    fit_visual_expectation,
)
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input

MAIN = "conditional_visual_motion"
ARMS = {
    MAIN: "conditional",
    "raw_visual_motion": "raw",
    "centered_visual_motion": "mean",
    "shuffled_visual_motion": "shuffled",
    "unnormalized_visual_motion": "unnormalized",
    "conditional_visual_only": "visual",
    "motion_numeric_mlp": "numeric",
}
TREES = (
    "refit_global_motion_mass",
    "embedding_motion_xgb",
    "raw_motion_information_xgb",
    "all_visual_motion_xgb",
)


def motion_numbers(bank, evidence):
    return np.column_stack((bank["numbers"], evidence["global_distribution"]))


def arm_inputs(bank, evidence, scaling, expectation, shuffled_expectation, kind):
    numbers = numeric_input(motion_numbers(bank, evidence), scaling)
    if kind == "numeric":
        return numbers
    selected_expectation = shuffled_expectation if kind == "shuffled" else expectation
    representation = "conditional" if kind in {"shuffled", "visual"} else kind
    visual = conditional_visual_features(
        bank["embeddings"], numbers, selected_expectation, representation
    )
    if kind == "visual":
        return visual
    return np.column_stack((visual, numbers)).astype(np.float32)


def tree_inputs(bank, evidence, name):
    parts = [motion_numbers(bank, evidence)]
    if name in {"raw_motion_information_xgb", "all_visual_motion_xgb"}:
        parts.extend((bank["chart_descriptors"], evidence["raw_profile"]))
    if name in {"embedding_motion_xgb", "all_visual_motion_xgb"}:
        parts.append(bank["embeddings"])
    return np.column_stack(parts)


def run(data, encoded, parent, out):
    stage_source = data.parent / "disagreement-calibration-v1"
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (parent, "freeze.json"),
        (stage_source, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    split = read_json(encoded / "protocol.json")["split"]
    if split != read_json(parent / "protocol.json")["split"] or read_json(parent / "protocol.json")[
        "encoded_manifest_sha256"
    ] != sha(encoded / "encoding-freeze.json"):
        raise ValueError("Motion evidence and original image cache use different populations")
    if read_json(parent / "protocol.json")["parent_manifest_sha256"] != sha(
        stage_source / "freeze.json"
    ):
        raise ValueError("Numerical probe used a different original stage/source calibration")
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_conditional_visual.py",
        ROOT / "tests/test_landslide_conditional_visual.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
    ]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protocol = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        candidate="NCVR / Numerical-Conditioned Visual Residuals; exploratory GNSS candidate",
        hypothesis="Global image embeddings mix motion and other appearance variation. Retain the residual not linearly predicted by original numerical movement distribution, and give both residual and unchanged evidence to the activity head",
        mechanism="TRAIN-only Ridge(alpha1,intercept) maps78scaled39motion inputs to512frozen visual embedding; unit residual plus78numbers=>590head inputs; no activity labels in representation fitting; one frozen projection at inference",
        prior="Residualization, regression and multimodal redundancy reduction already exist. Proposed scope is conditioning GNSS frozen chart features on verified motion distribution; no general mathematical novelty or causal nuisance-removal claim",
        correspondence_control="Permutation2026101326 of whole TRAIN embedding rows only in expectation fitting; activity head examples/labels/order unchanged",
        normalization_controls="Raw embedding, TRAIN mean-only unit residual, conditional without renormalization; zero residual stays zero using1e-12 denominator protection",
        split=split,
        scope="48TRAIN/24exposedVALIDATION; reserved24 inputs/reference not loaded; no newly independent records or vision/API calls",
        data_manifest_sha256=sha(data / "data-freeze.json"),
        encoding_manifest_sha256=sha(encoded / "encoding-freeze.json"),
        parent_manifest_sha256=sha(parent / "freeze.json"),
        stage_manifest_sha256=sha(stage_source / "freeze.json"),
        source_bindings=bindings,
        arms=ARMS,
        trees=TREES,
        training="Same10187stride3supported known TRAIN centers; seed20261020,12epochs,Adam.001/wd.0001,widths16/32; same class balancing; XGB120,lr.05,depth2/4,seed20261007: numerical-only trees all30659supported known TRAIN days, embedding trees identical10187encoded TRAIN centers;19thresholds,38opportunities per11arms=>418new",
        fairness="Raw/mean/conditional/shuffled/unnormalized590 dimensions, identical head sizes; numeric78/visual512 disclosed. Motion trees39/551/457/969dimensions; original18controls retained. Numerical contestants see all same raw observation sources",
        projection_scaling="Fit numeric missing-value means/scales and expectation on identical eligible TRAIN centers, reuse for validation; missingness flags retained; no validation fit or per-record recentering",
        inference="One projection and one activity head, common14positive observed decoder with<=7raw missing only, unchanged48TRAIN stage head; no Agent runtime or extra language call",
        gate=[
            "main dailyF1 above raw fusion and mean-only normalization",
            "main dailyF1 above shuffled conditional fit",
            "positive net known-day correction vs raw,mean and shuffled",
            "eventF1>=raw,mean,shuffled",
            "quiet/step-only/spike-only FP<=raw and mean",
            "mixed deformation recall>=raw and mean",
            "visual value: greater dailyF1 than strongest same-information nonvisual model OR greater eventF1 with dailyF1 within.01",
        ],
        stop="If any gate fails stop this exact version; no alpha/seed/scale/threshold rescue, no renamed best control, no reserved scoring",
        next="Only all conditions supports new-record controlled confirmation and subsequent observation-reference assessment",
        test_truth_opened=False,
        new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    banks = load_banks(encoded, split)
    evidence = {}
    for cid in banks:
        with np.load(parent / "evidence" / (cid + ".npz")) as archive:
            evidence[cid] = {key: archive[key] for key in archive.files}
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    numeric_training = np.concatenate(
        [
            motion_numbers(banks[cid], evidence[cid])[banks[cid]["selected"]]
            for cid in split["train"]
        ]
    )
    scaling = fit_numeric_scaling(numeric_training)
    scaled_training = numeric_input(numeric_training, scaling)
    visual_training = np.concatenate(
        [banks[cid]["embeddings"][banks[cid]["selected"]] for cid in split["train"]]
    )
    before = time.perf_counter()
    expectation = fit_visual_expectation(scaled_training, visual_training)
    shuffled_expectation = fit_visual_expectation(scaled_training, visual_training, shuffled=True)
    write_json(out / "numeric-scaling.json", scaling)
    write_json(out / "visual-expectation.json", expectation)
    write_json(out / "shuffled-expectation.json", shuffled_expectation)
    write_json(
        out / "projection-cost.json",
        dict(
            seconds=time.perf_counter() - before,
            samples=len(scaled_training),
            numerical_dimensions=scaled_training.shape[1],
            visual_dimensions=visual_training.shape[1],
            activity_labels_used=False,
        ),
    )
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    predictions = read_json(parent / "predictions.json")
    choices, trials, saved_probability, costs = {}, [], {}, []

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
                    cid: value.to_dict() for cid, value in decoded.items()
                }
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidates[choice["parameter"], choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in by_parameter[choice["parameter"]].items()
        }

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    target = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    plain = dict(augment=False, pair=False, shuffle=False)
    for name, kind in ARMS.items():
        inputs = {
            cid: arm_inputs(bank, evidence[cid], scaling, expectation, shuffled_expectation, kind)
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
                    seconds=seconds,
                    samples=samples,
                    dimensions=training.shape[1],
                    trainable_parameters=sum(value.numel() for value in model.parameters()),
                )
            )
        evaluate_and_select(name, models)
        print("calibrated", name, flush=True)
    actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
    eligible = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (actual >= 0)
    for name in TREES:
        training_eligible = eligible
        if name in {"embedding_motion_xgb", "all_visual_motion_xgb"}:
            training_eligible = np.concatenate(
                [banks[cid]["selected"] for cid in split["train"]]
            ) & (actual >= 0)
        matrices = {cid: tree_inputs(bank, evidence[cid], name) for cid, bank in banks.items()}
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
            model.fit(training[training_eligible], actual[training_eligible])
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
                    seconds=time.perf_counter() - before,
                    samples=int(training_eligible.sum()),
                    dimensions=training.shape[1],
                    trainable_parameters=None,
                )
            )
        evaluate_and_select(name, models)
        print("calibrated", name, flush=True)
    if predictions["refit_global_motion_mass"] != predictions["global_motion_mass"]:
        raise ValueError("Same-split numerical refit does not reproduce its original control")
    write_json(out / "selection.json", choices)
    write_json(out / "trials.json", trials)
    write_json(out / "probabilities.json", saved_probability)
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
    negatives = [
        cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}
    ]
    mixed = [
        cid
        for cid, family in families.items()
        if family in {"deformation_step", "deformation_spike"}
    ]

    def subset(name, ids):
        return aggregate_scores([results[name]["per_case"][cid] for cid in ids])["daily"]

    controls = ("raw_visual_motion", "centered_visual_motion", "shuffled_visual_motion")
    main = table[MAIN]
    net = {
        name: table[name]["fp"] + table[name]["fn"] - main["fp"] - main["fn"] for name in controls
    }
    best = max(
        (
            table[name]
            for name in (
                "motion_numeric_mlp",
                "refit_global_motion_mass",
                "raw_motion_information_xgb",
                "query_motion_mass",
                "all_motion_information",
                "xgb_chart384",
            )
        ),
        key=lambda row: row["f1"],
    )
    checks = dict(
        gain_vs_raw_and_centered=main["f1"] > max(table[name]["f1"] for name in controls[:2]),
        gain_vs_shuffled=main["f1"] > table[controls[2]]["f1"],
        positive_net_correction=min(net.values()) > 0,
        event_nonloss=main["event_f1_iou03"]
        >= max(table[name]["event_f1_iou03"] for name in controls),
        nuisance_nonloss=subset(MAIN, negatives)["fp"]
        <= min(subset(name, negatives)["fp"] for name in controls[:2]),
        mixed_nonloss=subset(MAIN, mixed)["recall"]
        >= max(subset(name, mixed)["recall"] for name in controls[:2]),
        visual_value=main["f1"] > best["f1"]
        or (main["event_f1_iou03"] > best["event_f1_iou03"] and main["f1"] >= best["f1"] - 0.01),
    )
    decision = dict(
        status="ELIGIBLE_CONDITIONAL_VISUAL_PENDING_REPLAY"
        if all(checks.values())
        else "STOP_CONDITIONAL_VISUAL",
        checks=checks,
        net_correct=net,
        best_nonvisual=best["method"],
        nuisance_fp={name: subset(name, negatives)["fp"] for name in (*ARMS, *TREES)},
        mixed_recall={name: subset(name, mixed)["recall"] for name in (*ARMS, *TREES)},
        test_truth_opened=False,
        new_api_calls=0,
    )
    write_json(out / "decision.json", decision)
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data.resolve(), args.encoded.resolve(), args.parent.resolve(), args.out.resolve())
