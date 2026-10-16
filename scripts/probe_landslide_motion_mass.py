"""Check incremental query-local movement information before another visual method."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_motion_mass import (
    GLOBAL_NAMES,
    LOCAL_NAMES,
    SCALES,
    motion_mass_descriptors,
)

MAIN = "query_motion_mass"
ARMS = ("mass_raw84", "global_motion_mass", MAIN, "raw_motion_profile", "all_motion_information")


def matrices_for_arm(banks, evidence, name):
    matrices = {}
    for cid, bank in banks.items():
        parts = [bank["numbers"]]
        if name == "global_motion_mass":
            parts.append(evidence[cid]["global_distribution"])
        elif name == MAIN:
            parts.append(evidence[cid]["query_distribution"])
        elif name == "raw_motion_profile":
            parts.append(evidence[cid]["raw_profile"])
        elif name == "all_motion_information":
            parts.extend(
                (
                    bank["chart_descriptors"],
                    evidence[cid]["raw_profile"],
                    evidence[cid]["query_distribution"],
                )
            )
        matrices[cid] = np.column_stack(parts)
    return matrices


def probe(data, encoded, parent, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (parent, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    split = read_json(encoded / "protocol.json")["split"]
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Original models use a different train/validation split")
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_motion_mass.py",
        ROOT / "tests/test_landslide_motion_mass.py",
        ROOT / "src/gnss_sim/landslide_feature_discovery.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "scripts/score_landslide_claim_review.py",
    ]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protocol = dict(
        candidate="Query-local displacement mass (numerical information probe)",
        split=split,
        scope="48TRAIN/24exposedVALIDATION; reserved24 inputs/references not loaded",
        hypothesis="Ordered movement concentration distinguishes dispersed motion, concentrated offsets, and movement outside the query; localization must add value over global distribution summaries",
        scales=SCALES,
        global_features=GLOBAL_NAMES,
        query_features=LOCAL_NAMES,
        observation="13 original14-day block medians requiring8 complete NEU; adjacent supported edges only; daily-noise units; 5/7/13 block subsets",
        unavailable="At least half adjacent edges and both query edges; missing target observations produceNaN; no new activity support, no interpolation or hard movement cutoff",
        arms=ARMS,
        training="All30659 originally supported known TRAIN dates; XGB120trees,depths2/4,lr.05,seed20261007,n_jobs2;19thresholds;38 opportunities per arm,190new total",
        selection="First max dailyF1+eventF1; all arms same frozen calendar decoder and numerical stage head",
        gate=[
            "main dailyF1 above raw84 AND global distribution",
            "main eventF1>=global AND >=raw84-.01",
            "main positive net correct days vs global",
            "nuisance-only FP<=raw84",
            "mixed deformation recall>=raw84",
        ],
        limitations="Numeric probe cannot establish Agent/VLM contribution; strongest all-info tree must be reported, not renamed as novel visual method",
        failure="Stop fixed descriptor version if gate fails; no feature/scale/seed/threshold rescue; no reserved validation",
        parent_manifest_sha256=sha(parent / "freeze.json"),
        encoded_manifest_sha256=sha(encoded / "encoding-freeze.json"),
        data_manifest_sha256=sha(data / "data-freeze.json"),
        source_bindings=bindings,
        test_truth_opened=False,
        new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    banks = load_banks(encoded, split)
    evidence, feature_costs = {}, []
    (out / "evidence").mkdir()
    for cid in banks:
        started = time.perf_counter()
        case, features = load_case_features(data, cid)
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        evidence[cid] = motion_mass_descriptors(raw, features.noise)
        np.savez_compressed(out / "evidence" / (cid + ".npz"), **evidence[cid])
        feature_costs.append(dict(case_id=cid, seconds=time.perf_counter() - started))
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(parent / "stage-xgboost.json")
    codes = {
        cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1
        for cid in split["validation"]
    }
    actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
    mask = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (actual >= 0)
    trials, choices, predictions, saved_probability, costs = [], {}, {}, {}, []
    for name in ARMS:
        matrices = matrices_for_arm(banks, evidence, name)
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        candidates, probabilities = {}, {}
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training[mask], actual[mask])
            model.save_model(out / f"{name}-depth{depth}.json")
            probabilities[depth] = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            for threshold in THRESHOLDS:
                decoded = {
                    cid: decode(probabilities[depth][cid], threshold, banks[cid], codes[cid])
                    for cid in split["validation"]
                }
                score = aggregate_scores(
                    [evaluate_case(labels[cid], decoded[cid]) for cid in decoded]
                )
                trials.append(
                    dict(
                        method=name,
                        parameter=depth,
                        threshold=threshold,
                        objective=calibration_objective(score),
                        validation=score,
                    )
                )
                candidates[depth, threshold] = {
                    cid: output.to_dict() for cid, output in decoded.items()
                }
            costs.append(
                dict(
                    method=name,
                    depth=depth,
                    seconds=time.perf_counter() - started,
                    samples=int(mask.sum()),
                    dimensions=training.shape[1],
                )
            )
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        choices[name] = choice
        predictions[name] = candidates[choice["parameter"], choice["threshold"]]
        saved_probability[name] = {
            cid: value.tolist() for cid, value in probabilities[choice["parameter"]].items()
        }
        print("Calibrated", name, flush=True)
    parent_predictions = read_json(parent / "predictions.json")
    if predictions["mass_raw84"] != parent_predictions["xgb_raw84"]:
        raise ValueError("Re-fitted raw84 does not reproduce the original fair control")
    predictions.update(parent_predictions)
    write_json(out / "selection.json", choices)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_json(out / "trials.json", trials)
    write_csv(out / "feature-costs.csv", feature_costs)
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
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in validation if families[cid] == family]
            )
            family_rows.append(
                dict(method=name, family=family, **score["daily"], event_f1=score["events"]["f1"])
            )
    write_csv(out / "family-summary.csv", family_rows)
    indexed = {row["method"]: row for row in summary}

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

    raw, global_control, main = indexed["mass_raw84"], indexed["global_motion_mass"], indexed[MAIN]
    net = global_control["fp"] + global_control["fn"] - main["fp"] - main["fn"]
    checks = dict(
        daily_increment=main["f1"] > max(raw["f1"], global_control["f1"]),
        event_nonloss=main["event_f1_iou03"] >= global_control["event_f1_iou03"]
        and main["event_f1_iou03"] >= raw["event_f1_iou03"] - 0.01,
        net_correction=net > 0,
        nuisance_nonloss=nuisance_fp(MAIN) <= nuisance_fp("mass_raw84"),
        mixed_nonloss=mixed_recall(MAIN) >= mixed_recall("mass_raw84"),
    )
    write_json(
        out / "decision.json",
        dict(
            status="ELIGIBLE_INFORMATION_PENDING_REPLAY"
            if all(checks.values())
            else "STOP_MOTION_MASS_DESCRIPTOR",
            checks=checks,
            net_vs_global=net,
            nuisance_fp={name: nuisance_fp(name) for name in ARMS},
            mixed_recall={name: mixed_recall(name) for name in ARMS},
            numerical_only=True,
            test_truth_opened=False,
        ),
    )
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen descriptor source changed: " + relative)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(read_json(out / "decision.json"), flush=True)
    print(
        [
            {k: indexed[name][k] for k in ("method", "f1", "event_f1_iou03", "fp", "fn")}
            for name in ARMS
        ],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.data.resolve(), args.encoded.resolve(), args.parent.resolve(), args.out.resolve())
