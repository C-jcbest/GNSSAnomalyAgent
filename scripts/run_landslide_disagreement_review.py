"""Cross-fit numerical revision utility and compare same-population strong controls."""

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
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities, train_head
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_disagreement_review import (
    fit_review,
    review_benefit,
    review_features,
    revise_disagreements,
)
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input

PLAIN = {"augment": False, "pair": False, "shuffle": False}
MAIN = "raw_evidence_review"


def neural_inputs(bank, scaling, name):
    numbers = numeric_input(bank["numbers"], scaling)
    if name == "numeric_mlp":
        return numbers
    if name == "pure_visual":
        return bank["embeddings"]
    return np.column_stack((bank["embeddings"], numbers)).astype(np.float32)


def fit_neural(banks, ids, scaling, name, width, prefix, costs):
    inputs = np.concatenate(
        [neural_inputs(banks[cid], scaling, name)[banks[cid]["selected"]] for cid in ids]
    )
    actual = np.concatenate([banks[cid]["activity"][banks[cid]["selected"]] for cid in ids])
    empty = np.empty((0, inputs.shape[1]), dtype=np.float32)
    model, history, seconds, samples = train_head(inputs, actual, empty, empty, PLAIN, width)
    torch.save(model.state_dict(), prefix.with_suffix(".pt"))
    write_json(prefix.with_suffix(".loss.json"), history)
    costs.append(
        dict(model=prefix.name, seconds=seconds, samples=samples, dimensions=inputs.shape[1])
    )
    return model


def tree_inputs(bank, name):
    if name == "xgb_raw12":
        return bank["stage_matrix"], bank["stage_supported"]
    if name == "xgb_chart384":
        return np.column_stack((bank["numbers"], bank["chart_descriptors"])), bank["supported"]
    return bank["numbers"], bank["supported"]


def fit_tree(banks, ids, name, depth, prefix, costs):
    parts = [tree_inputs(banks[cid], name) for cid in ids]
    matrix = np.concatenate([part[0] for part in parts])
    actual = np.concatenate([banks[cid]["activity"] for cid in ids])
    eligible = np.concatenate([part[1] for part in parts]) & (actual >= 0)
    started = time.perf_counter()
    model = XGBClassifier(
        n_estimators=120, max_depth=depth, learning_rate=0.05, random_state=20261007, n_jobs=2
    )
    model.fit(matrix[eligible], actual[eligible])
    model.save_model(prefix.with_suffix(".json"))
    costs.append(
        dict(
            model=prefix.name,
            seconds=time.perf_counter() - started,
            samples=int(eligible.sum()),
            dimensions=matrix.shape[1],
        )
    )
    return model


def load_banks(encoded, split):
    banks = {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoded / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
    return banks


def source_bindings():
    files = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_disagreement_review.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "scripts/score_landslide_claim_review.py",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def run(data, encoded, stage_source, out):
    verify_manifest(encoded, "encoding-freeze.json")
    verify_manifest(stage_source, "calibration-freeze.json")
    if (
        sha(data / "data-freeze.json")
        != read_json(encoded / "protocol.json")["data_manifest_sha256"]
    ):
        raise ValueError("Cross-fit cache belongs to different data")
    split = read_json(encoded / "protocol.json")["split"]
    if read_json(stage_source / "training-plan.json")["parent_protocol_sha256"] != sha(
        encoded / "protocol.json"
    ):
        raise ValueError("Stage control used different original records")
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        candidate="Disagreement-gated raw-evidence review, development only",
        split=split,
        folds=[split["train"][index::4] for index in range(4)],
        data_manifest_sha256=sha(data / "data-freeze.json"),
        encoding_manifest_sha256=sha(encoded / "encoding-freeze.json"),
        stage_manifest_sha256=sha(stage_source / "calibration-freeze.json"),
        source_bindings=source_bindings(),
        neural_widths=[16, 32],
        tree_depths=[2, 4],
        base_visual_width=16,
        base_numeric_depth=2,
        neural_seed=20261020,
        tree_seed=20261007,
        epochs=12,
        thresholds=THRESHOLDS,
        ridge_alpha=[1, 10],
        stacking_C=[1, 10],
        fixed_weights=[0.25, 0.5, 0.75],
        benefit="OOF numeric correctness minus OOF visual-numeric correctness, at each threshold",
        feedback="6 probability/margin features plus36 raw numeric values/availability; ablation removes36",
        revision="supported disagreement and predicted gain>0; switch to existing numeric score once; agreements unchanged",
        crossfit="all calendar dates of a held record excluded from both source models and numerical scaling; fit gate on stride3 OOF queries only",
        selection="same dailyF1+eventF1 first maximum; 38 learned chances, fixed blend57, rule/PELT9 each",
        common_decoder="14 positive observed queries, bridge only<=7 raw missing days, no imputation",
        gate=[
            "daily gain vs fixed blend and score-only review",
            "event nondecrease vs both",
            "positive net correction vs both",
            "nuisance FP no higher than fixed blend",
            "mixed recall no lower than fixed blend",
            "visual necessity vs raw84",
            "not dominated by generic same-evidence stacker",
        ],
        prior="Utility learning, stacking and OOF already exist, including this repository; new claim requires actual increment from permission-limited raw evidence",
        next="Only all checks permits new-record confirmation; failure stops this version, no seed/alpha/threshold rescue",
        test_truth_opened=False,
        new_api_calls=0,
        scope="48/24 exposed development population",
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(stage_source / "stage-xgboost.json", out / "stage-xgboost.json")
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    banks = load_banks(encoded, split)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    for cid, bank in banks.items():
        if not np.array_equal(
            bank["activity"], [int(row["activity_label"]) for row in labels[cid]]
        ):
            raise ValueError("Cached original activity differs: " + cid)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    costs, oof, fold_audit = [], {}, []
    for index, held in enumerate(protocol["folds"]):
        training_ids = [cid for cid in split["train"] if cid not in held]
        scaling = fit_numeric_scaling(
            np.concatenate([banks[cid]["numbers"][banks[cid]["selected"]] for cid in training_ids])
        )
        write_json(out / f"fold{index}-scaling.json", scaling)
        visual = fit_neural(
            banks, training_ids, scaling, "visual_numeric", 16, out / f"fold{index}-visual", costs
        )
        numeric = fit_tree(banks, training_ids, "xgb_raw84", 2, out / f"fold{index}-numeric", costs)
        for cid in held:
            selected = banks[cid]["selected"]
            left = probabilities(
                visual, neural_inputs(banks[cid], scaling, "visual_numeric")[selected]
            )
            right = numeric.predict_proba(banks[cid]["numbers"][selected])[:, 1]
            oof[cid] = dict(
                visual=left.tolist(),
                numeric=right.tolist(),
                original_days=np.flatnonzero(selected).tolist(),
                actual=banks[cid]["activity"][selected].tolist(),
                fold=index,
            )
        fold_audit.append(
            dict(
                fold=index,
                excluded=held,
                training_ids=training_ids,
                overlap=sorted(set(held) & set(training_ids)),
            )
        )
        print("cross-fitted", index, len(training_ids), len(held), flush=True)
    write_json(out / "out-of-fold.json", oof)
    write_json(out / "crossfit-audit.json", fold_audit)
    scaling = fit_numeric_scaling(
        np.concatenate([banks[cid]["numbers"][banks[cid]["selected"]] for cid in split["train"]])
    )
    if scaling != read_json(encoded / "numeric-scaling.json"):
        raise ValueError("Full TRAIN scaling differs from parent cache")
    write_json(out / "numeric-scaling.json", scaling)
    stage = XGBClassifier()
    stage.load_model(out / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    trials, choices, predictions, saved_probability, review_models, stacking_models = (
        [],
        {},
        {},
        {},
        {},
        {},
    )
    all_candidates = {}

    def evaluate(name, parameter, threshold, probability, supports=None):
        decoded = {
            cid: decode(
                probability[cid],
                threshold,
                banks[cid],
                codes[cid],
                None if supports is None else supports[cid],
            )
            for cid in split["validation"]
        }
        score = aggregate_scores(
            [evaluate_case(labels[cid], value) for cid, value in decoded.items()]
        )
        outputs = {cid: value.to_dict() for cid, value in decoded.items()}
        row = dict(
            method=name,
            parameter=parameter,
            threshold=threshold,
            objective=calibration_objective(score),
            validation=score,
        )
        trials.append(row)
        all_candidates[(name, parameter, threshold)] = (outputs, probability)

    def choose(name):
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        outputs, probability = all_candidates[(name, choice["parameter"], choice["threshold"])]
        predictions[name] = outputs
        saved_probability[name] = {cid: value.tolist() for cid, value in probability.items()}
        for key in [key for key in all_candidates if key[0] == name]:
            del all_candidates[key]

    base = {}
    for name in ("visual_numeric", "pure_visual", "numeric_mlp"):
        for width in (16, 32):
            model = fit_neural(
                banks, split["train"], scaling, name, width, out / f"{name}-width{width}", costs
            )
            probability = {
                cid: probabilities(model, neural_inputs(banks[cid], scaling, name))
                for cid in split["validation"]
            }
            if name == "visual_numeric" and width == 16:
                base["visual"] = probability
            for threshold in THRESHOLDS:
                evaluate(name, width, threshold, probability)
        choose(name)
        print("calibrated", name, flush=True)
    for name in ("xgb_raw84", "xgb_raw12", "xgb_chart384"):
        supports = {cid: tree_inputs(banks[cid], name)[1] for cid in split["validation"]}
        for depth in (2, 4):
            model = fit_tree(
                banks, split["train"], name, depth, out / f"{name}-depth{depth}", costs
            )
            probability = {
                cid: model.predict_proba(tree_inputs(banks[cid], name)[0])[:, 1]
                for cid in split["validation"]
            }
            if name == "xgb_raw84" and depth == 2:
                base["numeric"] = probability
            for threshold in THRESHOLDS:
                evaluate(name, depth, threshold, probability, supports)
        choose(name)
        print("calibrated", name, flush=True)
    write_json(
        out / "fixed-source-probabilities.json",
        {
            source: {cid: values.tolist() for cid, values in probability.items()}
            for source, probability in base.items()
        },
    )
    for weight in protocol["fixed_weights"]:
        probability = {
            cid: (1 - weight) * base["visual"][cid] + weight * base["numeric"][cid]
            for cid in split["validation"]
        }
        for threshold in THRESHOLDS:
            evaluate("fixed_blend", weight, threshold, probability)
    choose("fixed_blend")
    actual = np.concatenate([oof[cid]["actual"] for cid in split["train"]])
    visual = np.concatenate([oof[cid]["visual"] for cid in split["train"]])
    numeric = np.concatenate([oof[cid]["numeric"] for cid in split["train"]])
    numbers = np.concatenate(
        [
            numeric_input(banks[cid]["numbers"], scaling)[banks[cid]["selected"]]
            for cid in split["train"]
        ]
    )
    validation_numbers = {
        cid: numeric_input(banks[cid]["numbers"], scaling) for cid in split["validation"]
    }
    review_counts = []
    for name, raw_feedback in ((MAIN, True), ("scores_only_review", False)):
        review_models[name] = {}
        for alpha in protocol["ridge_alpha"]:
            for threshold in THRESHOLDS:
                matrix = review_features(
                    visual, numeric, numbers if raw_feedback else None, threshold
                )
                model = fit_review(matrix, visual, numeric, actual, threshold, alpha)
                review_models[name][f"{alpha}:{threshold}"] = model
                probability = {}
                for cid in split["validation"]:
                    features = review_features(
                        base["visual"][cid],
                        base["numeric"][cid],
                        validation_numbers[cid] if raw_feedback else None,
                        threshold,
                    )
                    value, switched = revise_disagreements(
                        base["visual"][cid],
                        base["numeric"][cid],
                        banks[cid]["supported"],
                        review_benefit(model, features),
                        threshold,
                    )
                    probability[cid] = value
                    review_counts.append(
                        dict(
                            method=name,
                            alpha=alpha,
                            threshold=threshold,
                            case_id=cid,
                            switched=int(switched.sum()),
                        )
                    )
                evaluate(name, alpha, threshold, probability)
        choose(name)
        print("calibrated", name, flush=True)
    for name, raw_feedback in (("generic_raw_stacker", True), ("generic_score_stacker", False)):
        # Use the same OOF sources, with ordinary activity supervision. The margin
        # reference is fixed .5 because a stacker predicts its own posterior.
        matrix = review_features(visual, numeric, numbers if raw_feedback else None, 0.5)
        scaler = StandardScaler().fit(matrix)
        stacking_models[name] = {}
        for regularization in protocol["stacking_C"]:
            model = LogisticRegression(
                C=regularization, class_weight="balanced", max_iter=1000, random_state=20261007
            ).fit(scaler.transform(matrix), actual)
            saved = dict(
                mean=scaler.mean_.tolist(),
                scale=scaler.scale_.tolist(),
                coefficient=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]),
            )
            stacking_models[name][str(regularization)] = saved
            probability = {
                cid: model.predict_proba(
                    scaler.transform(
                        review_features(
                            base["visual"][cid],
                            base["numeric"][cid],
                            validation_numbers[cid] if raw_feedback else None,
                            0.5,
                        )
                    )
                )[:, 1]
                for cid in split["validation"]
            }
            for threshold in THRESHOLDS:
                evaluate(name, regularization, threshold, probability)
        choose(name)
        print("calibrated", name, flush=True)
    features = {cid: load_case_features(data, cid)[1] for cid in split["validation"]}
    for name in ("robust_rule", "pelt_rule"):
        for setting in (1.0, 2.0, 3.0) if name == "robust_rule" else (10.0, 40.0, 160.0):
            probability, supports = {}, {}
            for cid, feature in features.items():
                if name == "robust_rule":
                    values = np.where(
                        feature.signal_to_noise >= setting, feature.progressive_rate, -1.0
                    )
                    supports[cid] = feature.supported
                else:
                    rates = pelt_segment_rates(feature, setting)
                    values = np.minimum(rates, 2 * feature.progressive_rate)
                    values[feature.signal_to_noise < 1] = -1.0
                    supports[cid] = feature.supported & np.isfinite(rates)
                probability[cid] = np.where(np.isfinite(values), values, -1.0)
            for threshold in (0.03, 0.06, 0.12):
                evaluate(name, setting, threshold, probability, supports)
        choose(name)
    write_json(out / "review-models.json", review_models)
    write_json(out / "stacking-models.json", stacking_models)
    write_json(out / "selection.json", choices)
    write_json(out / "validation-trials.json", trials)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "review-counts.csv", review_counts)
    write_csv(out / "training-costs.csv", costs)
    validation_labels = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation_labels}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]])
        for cid in validation_labels
    }
    results, summary, records = score_target(validation_labels, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", records)
    table = {row["method"]: row for row in summary}
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
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

    main = table[MAIN]
    controls = ["fixed_blend", "scores_only_review"]
    net = {
        name: table[name]["fp"] + table[name]["fn"] - main["fp"] - main["fn"] for name in controls
    }
    checks = dict(
        daily_gain=all(main["f1"] > table[name]["f1"] for name in controls),
        event_nondecrease=all(
            main["event_f1_iou03"] >= table[name]["event_f1_iou03"] for name in controls
        ),
        net_gain=min(net.values()) > 0,
        nuisance_fp=subset(MAIN, negative)["fp"] <= subset("fixed_blend", negative)["fp"],
        mixed_recall=subset(MAIN, mixed)["recall"] >= subset("fixed_blend", mixed)["recall"],
        visual_necessity=main["f1"] > table["xgb_raw84"]["f1"]
        or (
            main["event_f1_iou03"] > table["xgb_raw84"]["event_f1_iou03"]
            and main["f1"] >= table["xgb_raw84"]["f1"] - 0.01
        ),
        generic_stack_not_dominating=not (
            table["generic_raw_stacker"]["f1"] >= main["f1"]
            and table["generic_raw_stacker"]["event_f1_iou03"] >= main["event_f1_iou03"]
        ),
    )
    write_json(
        out / "decision.json",
        dict(
            decision="PROCEED_DISAGREEMENT_CONFIRMATION"
            if all(checks.values())
            else "STOP_DISAGREEMENT_REVIEW",
            checks=checks,
            net_correct=net,
            nuisance_fp={name: subset(name, negative)["fp"] for name in [MAIN, *controls]},
            mixed_recall={name: subset(name, mixed)["recall"] for name in [MAIN, *controls]},
            test_truth_opened=False,
            new_api_calls=0,
        ),
    )
    if source_bindings() != protocol["source_bindings"]:
        raise ValueError("Cross-fit review source changed during experiment")
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "stage-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(
        args.data.resolve(), args.encoded.resolve(), args.stage_source.resolve(), args.out.resolve()
    )
