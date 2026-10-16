"""Freeze a grouped P227 residual-background benchmark and recalibrate baselines."""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_field_probe import daily_coordinates, load_ngl_records
from prepare_landslide_motion_probe import FAMILIES, controlled_case
from run_landslide_comparison import (
    ROOT,
    calibration_objective,
    fit_baselines,
    observation_features,
    predict_baselines,
    save_json,
    sha,
)
from run_landslide_window_optimization import read_json
from xgboost import XGBClassifier

from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_evaluation import STAGES, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_semisynthetic import add_injection, residual_background

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
SOURCE = BASE / "field-data-v1/ngl-source-v1/P227.tenv3"
BACKGROUND_GROUPS = {"train": [0, 1, 6, 7], "validation": [2, 3], "mechanism_test": [4, 5, 8, 9]}
SEED = 2026101349
DAYS = 730


def fit_long_controls(features, evidence, labels, split, head, out):
    """Give every long-evidence competitor the same validation opportunities."""
    codes = {cid: head.predict(value.matrix).astype(int) + 1 for cid, value in features.items()}
    supports, matrices = {}, {}
    for cid, feature in features.items():
        h42, h84 = evidence[cid]
        matrix42 = np.column_stack((feature.matrix, h42["rate"], h42["signal_to_noise"], h42["direction"]))
        matrices[cid] = {"xgboost_raw_persistence": matrix42, "xgboost_42_matched_support": matrix42,
            "xgboost_84": np.column_stack((matrix42, h84["rate"], h84["signal_to_noise"], h84["direction"]))}
        supports[cid] = feature.observed & (np.isfinite(feature.progressive_rate)
            | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))

    def score(gates):
        return aggregate_scores([evaluate_case(labels[cid], learned_stages(
            codes[cid], gate, features[cid].supported)) for cid, gate in gates.items()])

    trials, choices, selected_models = [], {}, {}
    for method in ("raw_multiscale_rule", "raw_rule_84"):
        for rate in (.03, .06, .12):
            for snr in (1., 2., 3.):
                gates = {}
                for cid in split["validation"]:
                    feature, (h42, h84) = features[cid], evidence[cid]
                    if method == "raw_multiscale_rule":
                        gate = raw_gate(feature, h42, rate, snr)
                    else:
                        positive = (feature.progressive_rate >= rate) & (feature.signal_to_noise >= snr)
                        positive |= (h42["rate"] >= rate) & (h42["signal_to_noise"] >= snr)
                        positive |= (h84["rate"] >= rate) & (h84["signal_to_noise"] >= snr)
                        gate = activity_from_mask(positive, supports[cid])
                    gates[cid] = gate
                measured = score(gates)
                trials.append({"method": method, "rate": rate, "snr": snr,
                               "objective": calibration_objective(measured), "validation": measured})
        choices[method] = max((row for row in trials if row["method"] == method), key=lambda row: row["objective"])
    train_y = np.array([int(row["activity_label"]) for cid in split["train"] for row in labels[cid]])
    for method in ("xgboost_raw_persistence", "xgboost_42_matched_support", "xgboost_84"):
        train_x = np.concatenate([matrices[cid][method] for cid in split["train"]])
        eligible = np.concatenate([features[cid].supported if method == "xgboost_raw_persistence"
            else supports[cid] for cid in split["train"]]) & (train_y >= 0)
        models = {}
        for depth in (2, 4):
            model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                                  n_jobs=2, random_state=20261007)
            model.fit(train_x[eligible], train_y[eligible])
            models[depth] = model
            probabilities = {cid: model.predict_proba(matrices[cid][method])[:, 1] for cid in split["validation"]}
            for threshold in (.35, .5, .65):
                gates = {cid: activity_from_mask(probabilities[cid] >= threshold,
                    features[cid].supported if method == "xgboost_raw_persistence" else supports[cid])
                    for cid in split["validation"]}
                measured = score(gates)
                trials.append({"method": method, "depth": depth, "threshold": threshold,
                    "objective": calibration_objective(measured), "validation": measured})
        choices[method] = max((row for row in trials if row["method"] == method), key=lambda row: row["objective"])
        selected_models[method] = models[choices[method]["depth"]]
        selected_models[method].save_model(out / f"{method}.json")
    save_json(out / "long-validation-trials.json", trials)
    save_json(out / "long-selection.json", choices)
    return choices, selected_models


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    retrieval = read_json(SOURCE.parent / "current-retrieval-v2.json")
    expected = next(row["sha256"] for row in retrieval if row.get("file") == "P227.tenv3" and row.get("status") == 200)
    if sha(SOURCE) != expected:
        raise ValueError("Official source has changed")
    split = {name: [f"case_{2001 + block * 6 + family:04d}" for block in groups for family in range(6)]
             for name, groups in BACKGROUND_GROUPS.items()}
    protocol = {"schema": "gnss-p227-background-benchmark-v1", "seed": SEED, "days": DAYS,
        "source": str(SOURCE), "source_sha256": expected, "split": split, "background_groups": BACKGROUND_GROUPS,
        "families": list(FAMILIES), "background_start": "2006-02-17", "background_count": 10,
        "scope": "Known injected sustained-motion recovery in observed detrended residuals; not total field motion or expert references",
        "preprocessing": "Per730-day block observed-date OLS removes linear trends from ALL N/E/U before injection; no interpolation or residual rescaling",
        "grouping": "Six family variants share each observed background. No background dates cross splits. Four held-out background blocks, not24 independent sites",
        "seen_background_policy": "Blocks6/7 cover the2018-2020 field diagnostic and are train only; no existing model predictions or labels choose held-out blocks",
        "fitting": "Original robust9/PELT9/XGB6 trials; raw42/84 rules9 each; each extended XGB6; same train24/validation12; no old fitted model reused",
        "test": "Test inputs are generated only after all numerical choices/models freeze. No candidate/threshold/repetition switch from results",
        "method": "Original21/42 raw conflicts,14-day count,0.03 rate,2 noise proxy,2 reviews; primary pure-image conflict49 scope",
        "scope_check": "Report both repetitions versus whole-target; reducedFP with noTP or weak loss supports replication; otherwise explicitly not replicated. No retuning",
        "cost": "Two repeats on24 test inputs; maximum336 new API requests, no retry; same stage head; no competitor-fit tool"}
    save_json(out / "protocol.json", protocol)
    for folder in (ROOT / "scripts", ROOT / "src/gnss_sim"):
        for path in folder.glob("*.py"):
            target = out / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    save_json(out / "pre-generation-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    records = load_ngl_records(SOURCE)
    start_date = date(2006, 2, 17)
    backgrounds, mapping = {}, {}
    # Fitting background preprocessing never sees injection or method outputs.
    for block in range(10):
        calendar = [start_date + timedelta(days=block * DAYS + day) for day in range(DAYS)]
        residual, fit = residual_background(daily_coordinates(records, calendar))
        backgrounds[block] = (calendar, residual)
        mapping[str(block)] = {"start": str(calendar[0]), "stop_inclusive": str(calendar[-1]),
            "split": next(name for name, groups in BACKGROUND_GROUPS.items() if block in groups), **fit}
    save_json(out / "background-mapping.json", mapping)
    request = LandslideRequest(seed=SEED, count=60, days=DAYS)
    features, evidence, labels = {}, {}, {}

    def generate(block):
        calendar, residual = backgrounds[block]
        for family_index in range(6):
            index = block * 6 + family_index
            _, _, truth = controlled_case(request, index)
            cid = f"case_{2001 + index:04d}"
            latent = np.array(truth["latent_displacement_mm"])
            artifact = np.zeros_like(latent)
            offset = None
            if truth["family"] == "instrument_step":
                rng = np.random.default_rng(np.random.SeedSequence([SEED, index, 0xF14]))
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                offset = float(rng.uniform(12, 30)) * direction
                artifact[truth["instrument_step_day"]:] = offset
            observed = add_injection(residual, latent, artifact)
            case = LandslideInput(case_id=cid, dates=calendar, displacement_mm=observed)
            folder = out / "inputs" / cid
            folder.mkdir(parents=True)
            (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
            motions = {window: derive_motion(case, window) for window in (31, 61, 91)}
            for window, motion in motions.items():
                (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
            features[cid] = observation_features(case, motions)
            evidence[cid] = (raw_motion_evidence(case, features[cid]), raw_motion_evidence(case, features[cid], 84))
            activity = np.array(truth["speed_mm_day"]) > 0
            stage = np.zeros(DAYS, dtype=int)
            weak = np.zeros(DAYS, dtype=bool)
            episodes = np.full(DAYS, "", dtype=object)
            for phase in truth["schedule"]:
                left, right = phase["start"], phase["stop"]
                stage[left:right], weak[left:right], episodes[left:right] = phase["stage"], phase["weak"], phase["episode"]
            # Use the scoring module's actual business labels, not display codes.
            names = {code: name for name, code in STAGES.items()}
            labels[cid] = [{"case_id": cid, "day_index": day, "date": str(calendar[day]),
                "activity_label": int(activity[day]) if observed[day] is not None else -1,
                "feature": names[stage[day]] if activity[day] and observed[day] is not None else "",
                "episode_id": str(episodes[day]) if activity[day] and observed[day] is not None else "",
                "weak_signal": int(weak[day] and observed[day] is not None)} for day in range(DAYS)]
            truth.update({"scope": protocol["scope"], "background_group": block,
                "input_case_id": cid, "injected_artifact_offset_mm": None if offset is None else offset.tolist(),
                "background_source_sha256": expected, "injected_artifact_mm": artifact.tolist()})
            reference_folder = out / "mechanism-reference" / cid
            reference_folder.mkdir(parents=True)
            save_json(reference_folder / "truth.json", truth)
            print(f"Background input {cid} group{block} {truth['family']}", flush=True)

    for block in BACKGROUND_GROUPS["train"] + BACKGROUND_GROUPS["validation"]:
        generate(block)
    choices, activity_head, stage_head = fit_baselines(features, labels, split, out)
    long_choices, long_models = fit_long_controls(features, evidence, labels, split, stage_head, out)
    save_json(out / "fitting-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.glob("*.json") if path.name != "fitting-freeze.json"})
    for block in BACKGROUND_GROUPS["mechanism_test"]:
        generate(block)
    numeric = predict_baselines(features, split["mechanism_test"], choices, activity_head, stage_head)
    numeric.pop("ungated_rule")
    for cid in split["mechanism_test"]:
        feature, (h42, h84) = features[cid], evidence[cid]
        codes = stage_head.predict(feature.matrix).astype(int) + 1
        support = feature.observed & (np.isfinite(feature.progressive_rate) | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))
        matrix42 = np.column_stack((feature.matrix, h42["rate"], h42["signal_to_noise"], h42["direction"]))
        matrix84 = np.column_stack((matrix42, h84["rate"], h84["signal_to_noise"], h84["direction"]))
        for method, choice in long_choices.items():
            if method == "raw_multiscale_rule":
                gate = raw_gate(feature, h42, choice["rate"], choice["snr"])
            elif method == "raw_rule_84":
                positive = (feature.progressive_rate >= choice["rate"]) & (feature.signal_to_noise >= choice["snr"])
                positive |= (h42["rate"] >= choice["rate"]) & (h42["signal_to_noise"] >= choice["snr"])
                positive |= (h84["rate"] >= choice["rate"]) & (h84["signal_to_noise"] >= choice["snr"])
                gate = activity_from_mask(positive, support)
            else:
                matrix = matrix84 if method == "xgboost_84" else matrix42
                gate = activity_from_mask(long_models[method].predict_proba(matrix)[:, 1] >= choice["threshold"],
                    feature.supported if method == "xgboost_raw_persistence" else support)
            numeric.setdefault(method, {})[cid] = learned_stages(codes, gate, feature.supported)
    save_json(out / "numeric-predictions.json", {method: {cid: value.to_dict() for cid, value in cases.items()}
        for method, cases in numeric.items()})
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(labels.values()))[0]))
        writer.writeheader()
        for rows in labels.values():
            writer.writerows(rows)
    save_json(out / "manifest.json", {"status": "complete_test_predictions_not_scored", "split": split,
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
