"""Test one nuisance-conditioned evidence feature on the existing grouped split.

This is a numerical mechanism pilot. It must pass before adding expensive visual
review, and does not establish a new Agent/VLM contribution or unseen accuracy.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, calibration_objective
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_nuisance_evidence import (
    record_nuisance_evidence,
    render_nuisance_evidence,
)
from gnss_sim.landslide_review_utility import correction_utility

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
TRAINING = BASE / "motion-probe-v2"
DEVELOPMENT = BASE / "core-lockbox-data-v1"
OLD = BASE / "core-lockbox-analysis-v1/primary_process"
ARMS = ("numeric_raw84", "numeric_ncda", "numeric_exclusive_rivals")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty experiment table")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_matrix(raw_matrix, evidence, arm):
    if arm == "numeric_raw84":
        return raw_matrix
    column = 0 if arm == "numeric_ncda" else 1
    return np.column_stack((raw_matrix, evidence[:, column]))


def predict(probability, support, codes, stage_support, threshold):
    activity = activity_from_mask(probability >= threshold, support)
    return learned_stages(codes, activity, stage_support)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(TRAINING / "protocol.json")["split"]
    split = {group: split[group] for group in ("train", "validation")}
    split["development"] = read_json(DEVELOPMENT / "protocol.json")["case_ids"]
    write_json(out / "protocol.json", {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "24 original train, 12 original validation, 24 previously exposed development; not independent evaluation",
        "definition": "Target continuous deformation; instrument offsets and spikes are nuisances and may coexist with deformation",
        "split": split, "arms": list(ARMS),
        "change": "Exactly one shared-nuisance penalized Huber fit difference added to raw84 XGBoost",
        "evidence": "Original calendar 14-day targets; 168-day context; raw14 medians >=8 completeNEU, >=6 supported blocks, gaps<=7; H0 constant/step, H1 ramp/step+ramp; ramp contains query and lasts>=28d; bounded20 IRLS, Huber1.5, complexity=(3*columns+searchedknots)*log(3*blocks); divide cost difference by3*blocks",
        "information": "All arms see the same original diagnostics and raw42/84; ablation replaces just the new scalar with exclusive step-vs-ramp evidence; no visual, labels or latent motion in evidence path",
        "training": "120 trees, learning_rate.05, seed20261007, n_jobs2; same complete known supported train dates; depths2/4 and thresholds.35/.5/.65, six chances per arm",
        "selection": "First maximum validation daily F1+event F1; all choices freeze before development labels are loaded",
        "support": "Identical original observed short/42/84 union; unavailable evidence is NaN and does not remove scoring dates; same frozen numerical stage head",
        "continue_gate": "On exposed development: NCDA dailyF1>numeric_raw84; eventF1>=numeric_raw84; net correction>0; stationary/offset-only FP<raw84; NCDA F1>=exclusive rivals and at least one strict F1/event advantage. All outcomes reported, no test threshold or parameter rescue",
        "next": "Only after pilot benefit: fixed vs adaptive bounded visual review with original diagrams and same-information numeric controls; new diverse data with frozen population/split",
        "new_api_calls": 0, "independent_reference": False,
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_nuisance_evidence.py",
               ROOT / "tests/test_landslide_nuisance_evidence.py",
               ROOT / "src/gnss_sim/landslide_observation_view.py",
               ROOT / "scripts/score_landslide_claim_review.py",
               ROOT / "src/gnss_sim/landslide_evaluation.py"]
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    for path in sources:
        destination = out / "source" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
    for group, ids in split.items():
        source = DEVELOPMENT if group == "development" else TRAINING
        for cid in ids:
            for path in (source / "inputs" / cid).iterdir():
                if path.is_file():
                    bindings[str(path.relative_to(ROOT))] = sha(path)
    for path in (TRAINING / "reference-labels.csv", TRAINING / "stage-xgboost.json",
                 OLD / "reference-labels.json", OLD / "predictions.json"):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    write_json(out / "source-bindings.json", bindings)
    write_json(out / "pre-fit-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    head = XGBClassifier()
    head.load_model(TRAINING / "stage-xgboost.json")
    records = {}
    timings = []
    for group, ids in split.items():
        source = DEVELOPMENT if group == "development" else TRAINING
        for cid in ids:
            started = time.perf_counter()
            case, feature = load_case_features(source, cid)
            short = raw_motion_evidence(case, feature)
            long = raw_motion_evidence(case, feature, 84)
            raw_matrix = np.column_stack((feature.matrix, *[item[key] for item in (short, long) for key in ("rate", "signal_to_noise", "direction")]))
            support = feature.observed & (np.isfinite(feature.progressive_rate) | np.isfinite(short["rate"]) | np.isfinite(long["rate"]))
            raw_seconds = time.perf_counter() - started
            values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
            started = time.perf_counter()
            evidence, diagnostics = record_nuisance_evidence(values, feature.noise)
            evidence_seconds = time.perf_counter() - started
            records[cid] = {"case": case, "feature": feature, "raw": raw_matrix, "evidence": evidence, "support": support, "codes": head.predict(feature.matrix).astype(int) + 1}
            np.savez_compressed(out / (cid + "-features.npz"), raw=raw_matrix, evidence=evidence, support=support)
            write_json(out / (cid + "-evidence.json"), diagnostics)
            # A fixed training index illustrates the evidence without selecting a
            # high-scoring example or changing any reference after viewing it.
            if group == "train" and cid in split["train"][:2]:
                diagnostic = diagnostics[len(diagnostics) // 2]
                if diagnostic["status"] == "available":
                    render_nuisance_evidence(values, diagnostic, out / (cid + "-observation.png"))
            timings.append({"case_id": cid, "group": group, "raw_seconds": raw_seconds, "evidence_seconds": evidence_seconds, "evidence_available_days": int(np.isfinite(evidence[:, 0]).sum()), "original_days": len(values)})
            print(group, cid, f"evidence {evidence_seconds:.3f}s", flush=True)
    labels = {cid: [] for cid in split["train"] + split["validation"]}
    with (TRAINING / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    target = np.array([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    eligible = np.concatenate([records[cid]["support"] for cid in split["train"]]) & (target >= 0)
    selections = {}
    trials = []
    predictions = {arm: {} for arm in ARMS}
    probabilities = {arm: {} for arm in ARMS}
    learning_started = time.perf_counter()
    for arm in ARMS:
        train_x = np.concatenate([evidence_matrix(records[cid]["raw"], records[cid]["evidence"], arm) for cid in split["train"]])
        candidates = {}
        for depth in (2, 4):
            model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05, n_jobs=2, random_state=20261007)
            model.fit(train_x[eligible], target[eligible])
            candidates[depth] = model
            for threshold in (.35, .5, .65):
                scores = []
                for cid in split["validation"]:
                    record = records[cid]
                    probability = model.predict_proba(evidence_matrix(record["raw"], record["evidence"], arm))[:, 1]
                    prediction = predict(probability, record["support"], record["codes"], record["feature"].supported, threshold)
                    scores.append(evaluate_case(labels[cid], prediction))
                score = aggregate_scores(scores)
                trials.append({"arm": arm, "depth": depth, "threshold": threshold, "objective": calibration_objective(score), "validation": score})
        choice = max((row for row in trials if row["arm"] == arm), key=lambda row: row["objective"])
        selections[arm] = choice
        model = candidates[choice["depth"]]
        model.save_model(out / (arm + "-model.json"))
        for cid in split["development"]:
            record = records[cid]
            probability = model.predict_proba(evidence_matrix(record["raw"], record["evidence"], arm))[:, 1]
            probabilities[arm][cid] = probability.tolist()
            predictions[arm][cid] = predict(probability, record["support"], record["codes"], record["feature"].supported, choice["threshold"]).to_dict()
    write_json(out / "validation-trials.json", trials)
    write_json(out / "selection.json", selections)
    write_json(out / "new-predictions.json", predictions)
    write_json(out / "probabilities.json", probabilities)
    write_csv(out / "evidence-timings.csv", timings)
    write_json(out / "prediction-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    learning_seconds = time.perf_counter() - learning_started
    # Only now load the complete exposed development labels and family metadata.
    development_labels = read_json(OLD / "reference-labels.json")
    observed = {cid: records[cid]["feature"].observed for cid in split["development"]}
    weak = {cid: np.array([bool(int(row.get("weak_signal", 0))) for row in development_labels[cid]]) for cid in split["development"]}
    historical = read_json(OLD / "predictions.json")
    predictions.update({"historical__" + name: value for name, value in historical.items()})
    results, summaries, per_case = score_target(development_labels, observed, weak, predictions, {}, {})
    actual = {cid: np.array([int(row["activity_label"]) for row in development_labels[cid]]) for cid in split["development"]}
    for row in summaries:
        if row["method"] in ARMS:
            mask = {cid: (actual[cid] >= 0) & records[cid]["support"] for cid in actual}
            row["auprc_common_support"] = float(average_precision_score(np.concatenate([actual[cid][mask[cid]] for cid in actual]), np.concatenate([np.array(probabilities[row["method"]][cid])[mask[cid]] for cid in actual])))
        else:
            row["auprc_common_support"] = None
    correction_rows = []
    for arm in ARMS[1:]:
        for cid in actual:
            before = np.array(predictions["numeric_raw84"][cid]["activity"])
            after = np.array(predictions[arm][cid]["activity"])
            correction_rows.append({"method": arm, "case_id": cid, **correction_utility(actual[cid], before, after, np.ones(len(before), dtype=bool))})
    negative_ids = [cid for cid in actual if not np.any(actual[cid] == 1)]
    table = {row["method"]: row for row in summaries}
    negative_fp = {arm: sum(int((np.array(predictions[arm][cid]["activity"]) == 1).sum()) for cid in negative_ids) for arm in ARMS}
    net = sum(row["net_correct"] for row in correction_rows if row["method"] == "numeric_ncda")
    baseline, main, exclusive = [table[arm] for arm in ARMS]
    checks = {
        "daily_f1_increase": baseline["f1"] < main["f1"],
        "event_f1_nondecrease": baseline["event_f1_iou03"] <= main["event_f1_iou03"],
        "net_correction_positive": net > 0,
        "negative_record_fp_decrease": negative_fp["numeric_ncda"] < negative_fp["numeric_raw84"],
        "shared_nuisance_advantage": main["f1"] >= exclusive["f1"] and (main["f1"] > exclusive["f1"] or main["event_f1_iou03"] > exclusive["event_f1_iou03"]),
    }
    decision = {"decision": "PROCEED_NCDA_VISUAL_MECHANISM" if all(checks.values()) else "STOP_NCDA_SCALAR_PILOT", "checks": checks, "negative_record_ids": negative_ids, "negative_record_fp": negative_fp, "net_correct": net, "new_api_calls": 0, "learning_and_prediction_seconds": learning_seconds, "interpretation": "Exposed numerical development only; no new visual or independent-reference claim"}
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summaries)
    write_json(out / "decision.json", decision)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per-case.csv", per_case)
    write_csv(out / "corrections.csv", correction_rows)
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Experiment source or input changed during execution")
    write_json(out / "delivery-manifest.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
