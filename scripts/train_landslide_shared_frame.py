"""Compare shared versus independent chart coordinates with identical supervised exposure."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import modality_inputs, probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_shared_frame import train_shared_pool
from gnss_sim.landslide_visual_invariance import numeric_input

MAIN = "shared_visual_numeric_pair"
ARMS = {
    MAIN: dict(image=True, numeric=True, pair=True, shuffle=False, frame="shared"),
    "shared_visual_numeric_augmentation": dict(
        image=True, numeric=True, pair=False, shuffle=False, frame="shared"
    ),
    "independent_visual_numeric_pair": dict(
        image=True, numeric=True, pair=True, shuffle=False, frame="independent"
    ),
    "shared_visual_numeric_shuffled": dict(
        image=True, numeric=True, pair=True, shuffle=True, frame="shared"
    ),
    "shared_visual_pair": dict(image=True, numeric=False, pair=True, shuffle=False, frame="shared"),
    "shared_numeric_pair": dict(
        image=False, numeric=True, pair=True, shuffle=False, frame="shared"
    ),
    "shared_numeric_augmentation": dict(
        image=False, numeric=True, pair=False, shuffle=False, frame="shared"
    ),
}


def calibrate(data, encoded, pairs, parent, out):
    for folder, manifest in (
        (encoded, "encoding-freeze.json"),
        (pairs, "encoding-freeze.json"),
        (parent, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(pairs / "protocol.json")
    split = protocol["split"]
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Parent models used different train/validation records")
    if sha(encoded / "encoding-freeze.json") != protocol["encoded_manifest_sha256"]:
        raise ValueError("Original encoder input changed")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared implementation changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        dict(
            **protocol,
            arms=ARMS,
            parent_manifest_sha256=sha(parent / "freeze.json"),
            pairs_manifest_sha256=sha(pairs / "encoding-freeze.json"),
            new_trials=9 * 38,
            parent_trials=455,
            parent_reuse="All 13 parent methods on exact48/24 original images/numbers/calendar/new decoder; new same-supervision trees refitted",
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(parent / "stage-xgboost.json", out / "stage-xgboost.json")
    banks = load_banks(encoded, split)
    scaling = read_json(encoded / "numeric-scaling.json")
    if scaling != read_json(parent / "numeric-scaling.json"):
        raise ValueError("Parent and new models differ in TRAIN numerical scaling")
    write_json(out / "numeric-scaling.json", scaling)
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    with np.load(pairs / "pair-bank.npz") as archive:
        pair_bank = {key: archive[key] for key in archive.files}
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
    embeddings = np.concatenate(
        [banks[cid]["embeddings"][banks[cid]["selected"]] for cid in split["train"]]
    )
    numbers = numeric_input(
        np.concatenate([banks[cid]["numbers"][banks[cid]["selected"]] for cid in split["train"]]),
        scaling,
    )
    actual = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    trials, choices, predictions, saved_probability, costs = [], {}, {}, {}, []

    def evaluate(name, parameter, probability):
        candidates = {}
        for threshold in THRESHOLDS:
            decoded = {
                cid: decode(probability[cid], threshold, banks[cid], codes[cid])
                for cid in split["validation"]
            }
            score = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in decoded])
            trials.append(
                dict(
                    method=name,
                    parameter=parameter,
                    threshold=threshold,
                    objective=calibration_objective(score),
                    validation=score,
                )
            )
            candidates[threshold] = {cid: value.to_dict() for cid, value in decoded.items()}
        return candidates

    def choose(name, candidates, probability_bank):
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidates[choice["parameter"]][choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in probability_bank[choice["parameter"]].items()
        }

    for name, arm in ARMS.items():
        original = modality_inputs(embeddings, numbers, arm)
        positive_numbers = numeric_input(pair_bank["positive_numbers"], scaling)
        negative_numbers = numeric_input(pair_bank["negative_numbers"], scaling)
        shared_positive = modality_inputs(
            pair_bank["shared_positive_embeddings"], positive_numbers, arm
        )
        shared_negative = modality_inputs(
            pair_bank["shared_negative_embeddings"], negative_numbers, arm
        )
        independent_negative = modality_inputs(
            pair_bank["independent_negative_embeddings"], negative_numbers, arm
        )
        supervised = np.concatenate(
            (original, shared_positive, shared_negative, independent_negative)
        )
        supervised_target = np.concatenate(
            (
                actual,
                np.ones(len(shared_positive)),
                np.zeros(len(shared_negative) + len(independent_negative)),
            )
        )
        kind = arm["frame"]
        positive = modality_inputs(pair_bank[kind + "_positive_embeddings"], positive_numbers, arm)
        negative = modality_inputs(pair_bank[kind + "_negative_embeddings"], negative_numbers, arm)
        candidates, probability_bank = {}, {}
        for width in (16, 32):
            model, history, seconds, samples = train_shared_pool(
                supervised, supervised_target, positive, negative, arm, width
            )
            torch.save(model.state_dict(), out / f"{name}-width{width}.pt")
            write_json(out / f"{name}-width{width}-history.json", history)
            probability_bank[width] = {
                cid: probabilities(
                    model,
                    modality_inputs(
                        banks[cid]["embeddings"], numeric_input(banks[cid]["numbers"], scaling), arm
                    ),
                )
                for cid in split["validation"]
            }
            candidates[width] = evaluate(name, width, probability_bank[width])
            costs.append(
                dict(
                    method=name,
                    parameter=width,
                    seconds=seconds,
                    samples=samples,
                    dimensions=original.shape[1],
                )
            )
        choose(name, candidates, probability_bank)
        print("Calibrated", name, flush=True)
    for name, chart in (
        ("shared_xgb_raw84_augmentation", False),
        ("shared_xgb_chart384_augmentation", True),
    ):
        matrices = {
            cid: np.column_stack((bank["numbers"], bank["chart_descriptors"]))
            if chart
            else bank["numbers"]
            for cid, bank in banks.items()
        }
        masks = {
            cid: banks[cid]["supported"] & (banks[cid]["activity"] >= 0) for cid in split["train"]
        }
        augmented_numbers = np.concatenate(
            (
                pair_bank["positive_numbers"],
                pair_bank["negative_numbers"],
                pair_bank["negative_numbers"],
            )
        )
        if chart:
            augmented_charts = np.concatenate(
                (
                    pair_bank["shared_positive_charts"],
                    pair_bank["shared_negative_charts"],
                    pair_bank["independent_negative_charts"],
                )
            )
            augmented = np.column_stack((augmented_numbers, augmented_charts))
        else:
            augmented = augmented_numbers
        training = np.concatenate(
            [matrices[cid][masks[cid]] for cid in split["train"]] + [augmented]
        )
        pair_count = len(pair_bank["positive_numbers"])
        target = np.concatenate(
            [banks[cid]["activity"][masks[cid]] for cid in split["train"]]
            + [np.ones(pair_count), np.zeros(2 * pair_count)]
        )
        candidates, probability_bank = {}, {}
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training, target)
            model.save_model(out / f"{name}-depth{depth}.json")
            probability_bank[depth] = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            candidates[depth] = evaluate(name, depth, probability_bank[depth])
            costs.append(
                dict(
                    method=name,
                    parameter=depth,
                    seconds=time.perf_counter() - started,
                    samples=len(target),
                    dimensions=training.shape[1],
                )
            )
        choose(name, candidates, probability_bank)
        print("Calibrated", name, flush=True)
    parent_predictions = read_json(parent / "predictions.json")
    if set(parent_predictions) & set(predictions):
        raise ValueError("New methods must not overwrite parent names")
    predictions.update(parent_predictions)
    write_json(out / "selection.json", choices)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_json(out / "trials.json", trials)
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
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in validation
    }
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            group_score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in validation if families[cid] == family]
            )
            family_rows.append(
                dict(
                    method=name,
                    family=family,
                    **group_score["daily"],
                    event_f1=group_score["events"]["f1"],
                )
            )
    write_csv(out / "family-summary.csv", family_rows)
    scores = {row["method"]: row for row in summary}
    augmentation = "shared_visual_numeric_augmentation"
    frame_control = "independent_visual_numeric_pair"

    def nuisance_fp(name):
        return sum(
            results[name]["per_case"][cid]["daily"]["fp"]
            for cid in validation
            if families[cid] in {"quiet", "step_only", "spike_only"}
        )

    def mixed_recall(name):
        score = aggregate_scores(
            [
                results[name]["per_case"][cid]
                for cid in validation
                if families[cid] in {"deformation_step", "deformation_spike"}
            ]
        )
        return score["daily"]["recall"]

    def net(name):
        return scores[name]["fp"] + scores[name]["fn"] - scores[MAIN]["fp"] - scores[MAIN]["fn"]

    checks = dict(
        daily_increment=scores[MAIN]["f1"]
        > max(scores[augmentation]["f1"], scores[frame_control]["f1"])
        and scores[MAIN]["f1"] >= scores["visual_numeric"]["f1"],
        event_nonloss=scores[MAIN]["event_f1_iou03"]
        >= max(scores[augmentation]["event_f1_iou03"], scores[frame_control]["event_f1_iou03"]),
        net_correction=net(augmentation) > 0 and net(frame_control) > 0,
        nuisance_nonloss=nuisance_fp(MAIN)
        <= min(nuisance_fp(augmentation), nuisance_fp("visual_numeric")),
        mixed_nonloss=mixed_recall(MAIN) >= mixed_recall(augmentation),
        visual_value=scores[MAIN]["f1"] > scores["shared_numeric_pair"]["f1"]
        or (
            scores[MAIN]["event_f1_iou03"] > scores["shared_numeric_pair"]["event_f1_iou03"]
            and scores[MAIN]["f1"] >= scores["shared_numeric_pair"]["f1"] - 0.01
        ),
    )
    write_json(
        out / "decision.json",
        dict(
            status="ELIGIBLE_PENDING_REPLAY" if all(checks.values()) else "STOP_SHARED_FRAME",
            performance_checks=checks,
            all_six_performance_checks=all(checks.values()),
            strong_numeric_reported=True,
            replay_pending=True,
            net_vs_augmentation=net(augmentation),
            net_vs_independent=net(frame_control),
            nuisance_fp={
                name: nuisance_fp(name) for name in (MAIN, augmentation, "visual_numeric")
            },
            mixed_recall={name: mixed_recall(name) for name in (MAIN, augmentation)},
            test_truth_opened=False,
        ),
    )
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Executing implementation changed: " + relative)
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(read_json(out / "decision.json"), flush=True)
    print(
        [
            {key: row[key] for key in ("method", "f1", "event_f1_iou03", "fp", "fn")}
            for row in summary
            if row["method"]
            in {
                MAIN,
                augmentation,
                frame_control,
                "visual_numeric",
                "xgb_raw84",
                "shared_numeric_pair",
            }
        ],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "pairs", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    calibrate(
        args.data.resolve(),
        args.encoded.resolve(),
        args.pairs.resolve(),
        args.parent.resolve(),
        args.out.resolve(),
    )
