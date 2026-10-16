"""Test component-preserving evidence before paying for another visual mechanism."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS, prediction
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_axis_evidence import axis_persistence_evidence, pool_axis_evidence
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case


def probe(data, encoding, previous, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoding, "encoding-freeze.json"),
        (previous, "calibration-freeze.json"),
    ):
        for relative, expected in read_json(folder / manifest).items():
            if sha(folder / relative) != expected:
                raise ValueError("Upstream experiment changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(encoding / "protocol.json")["split"]
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_axis_evidence.py",
        ROOT / "tests/test_landslide_axis_evidence.py",
        ROOT / "scripts/train_landslide_deformation_quality.py",
    ]
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Same new48train/24exposedvalidation; all24 reserved references remain unopened",
        "hypothesis": "Preserve signed axiswise noise-normalized successive displacement instead of pooling axes prematurely",
        "derivation": "For each axis and21/42/84 spacing, sign(delta_before*delta_after)*min(abs(delta_before),abs(delta_after))/daily_difference_noise; same disjoint14-day medians >=8 complete observations; unavailable staysNaN",
        "inputs": "Old18raw84 in every arm. Add9 axis scores or3 positive-part norm pools of the EXACT same scores. No new sensor or generator components. No new activity support",
        "arms": ["raw84", "pooled_evidence", "axis_evidence"],
        "training": "All known supported training dates; same XGB120trees lr.05 seed20261007 n_jobs2; depths2/4 times19 frozen thresholds=38 opportunities per arm; first maximum dailyF1+eventF1",
        "stage": "Unchanged new48-trained numerical stage model; activity before phases; same14-day minimum run and all known positive dates in denominator",
        "gate": "Axis F1>raw and pooled; eventF1>=raw; net corrections>0; pure nuisance FP<=raw; mixed deformation recall>=raw. All six required before another visual probe. This numerical pilot cannot establish Agent/VLM contribution",
        "failure": "Stop this exact version, preserve results, no seed/scale/threshold/subgroup rescue; never open reserved references to rescue validation",
        "prior_exposure": "All24 validation used in prior DQ-VN model selection; two scored original images inspected. This is development, not independent confirmation",
        "upstream": {
            str(folder): sha(folder / manifest)
            for folder, manifest in (
                (data, "data-freeze.json"),
                (encoding, "encoding-freeze.json"),
                (previous, "calibration-freeze.json"),
            )
        },
        "test_reference_opened": False,
        "new_api_calls": 0,
    }
    write_json(out / "protocol.json", protocol)
    for source in sources:
        destination = out / "source" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    banks, features, evidence, timings = {}, {}, {}, []
    for group, ids in split.items():
        for cid in ids:
            started = time.perf_counter()
            with np.load(encoding / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            case, features[cid] = load_case_features(data, cid)
            values = np.array(
                [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
            )
            evidence[cid] = axis_persistence_evidence(values, features[cid].noise)
            destination = out / "evidence" / (cid + ".npz")
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                destination, axis=evidence[cid], pooled=pool_axis_evidence(evidence[cid])
            )
            timings.append(
                {"case_id": cid, "group": group, "seconds": time.perf_counter() - started}
            )
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(previous / "stage-xgboost.json")
    codes = {
        cid: stage.predict(features[cid].matrix).astype(int) + 1 for cid in split["validation"]
    }
    actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
    supported = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (actual >= 0)
    trials, selections, predictions, probabilities, training_costs = [], {}, {}, {}, []
    for name in protocol["arms"]:
        matrices = {}
        for cid, bank in banks.items():
            if name == "raw84":
                matrices[cid] = bank["numbers"]
            elif name == "pooled_evidence":
                matrices[cid] = np.column_stack(
                    (bank["numbers"], pool_axis_evidence(evidence[cid]))
                )
            else:
                matrices[cid] = np.column_stack((bank["numbers"], evidence[cid]))
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        outputs, probability_bank = {}, {}
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training[supported], actual[supported])
            model.save_model(out / (name + f"-depth{depth}.json"))
            probability_bank[depth] = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            for threshold in THRESHOLDS:
                output = {
                    cid: prediction(probability_bank[depth][cid], banks[cid], codes[cid], threshold)
                    for cid in split["validation"]
                }
                score = aggregate_scores(
                    [evaluate_case(labels[cid], output[cid]) for cid in output]
                )
                trials.append(
                    {
                        "method": name,
                        "depth": depth,
                        "threshold": threshold,
                        "objective": calibration_objective(score),
                        "validation": score,
                    }
                )
                outputs[depth, threshold] = {cid: value.to_dict() for cid, value in output.items()}
            training_costs.append(
                {"method": name, "depth": depth, "seconds": time.perf_counter() - started}
            )
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        selections[name] = choice
        predictions[name] = outputs[choice["depth"], choice["threshold"]]
        probabilities[name] = {
            cid: value.tolist() for cid, value in probability_bank[choice["depth"]].items()
        }
        print("Completed", name, flush=True)
    # The raw control must reproduce the already-frozen same-population model.
    prior = read_json(previous / "predictions.json")
    if predictions["raw84"] != prior["xgb_raw84"]:
        raise ValueError("Raw84 same-protocol control failed exact prediction replay")
    write_json(out / "probabilities.json", probabilities)
    write_json(out / "predictions.json", predictions)
    write_json(out / "trials.json", trials)
    write_json(out / "selection.json", selections)
    write_csv(out / "evidence-costs.csv", timings)
    write_csv(out / "training-costs.csv", training_costs)
    write_json(
        out / "prediction-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
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
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["group"] == "validation"
    }

    def subset(method, family_set):
        return aggregate_scores(
            [
                results[method]["per_case"][cid]
                for cid in split["validation"]
                if families[cid] in family_set
            ]
        )["daily"]

    negative = {
        name: subset(name, {"quiet", "step_only", "spike_only"})["fp"] for name in predictions
    }
    mixed = {
        name: subset(name, {"deformation_step", "deformation_spike"})["recall"]
        for name in predictions
    }
    table = {row["method"]: row for row in summary}
    raw, axis, pooled = [table[name] for name in ("raw84", "axis_evidence", "pooled_evidence")]
    net = raw["fp"] + raw["fn"] - axis["fp"] - axis["fn"]
    checks = {
        "daily_f1_over_raw": axis["f1"] > raw["f1"],
        "daily_f1_over_pooled": axis["f1"] > pooled["f1"],
        "events_nondecrease": axis["event_f1_iou03"] >= raw["event_f1_iou03"],
        "net_correction_positive": net > 0,
        "nuisance_fp_nondecrease_avoided": negative["axis_evidence"] <= negative["raw84"],
        "mixed_recall_nondecrease": mixed["axis_evidence"] >= mixed["raw84"],
    }
    decision = {
        "decision": "PROCEED_AXIS_GUIDED_VISUAL_PROBE"
        if all(checks.values())
        else "STOP_AXIS_EVIDENCE_PILOT",
        "checks": checks,
        "net_correct": net,
        "negative_fp": negative,
        "mixed_recall": mixed,
        "test_truth_opened": False,
        "new_api_calls": 0,
        "interpretation": "Numerical development gate only, not a new Agent/VLM contribution",
    }
    write_json(out / "decision.json", decision)
    for source in sources:
        if sha(source) != sha(out / "source" / source.name):
            raise ValueError("Source changed during pilot: " + str(source))
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe(args.data.resolve(), args.encoding.resolve(), args.previous.resolve(), args.out.resolve())
