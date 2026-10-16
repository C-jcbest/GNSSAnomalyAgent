"""Test bounded visual labeling of observation-derived temporal partitions."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, load_labels, save_json, summarize
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_dual_reference_report import full_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_prompt,
)
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_partition_review import (
    numeric_piece_activity,
    parse_piece_activity,
    partition_review_prompt,
    rate_partition,
    uniform_partition,
)
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
PARENT = BASE / "signed-evidence-development-v1"
PRIOR = BASE / "signed-evidence-development-analysis-v1"
ORIGINAL = BASE / "sequential-followup-confirmation-comparison-v1/run"
REFERENCES = BASE / "sequential-followup-confirmation-analysis-v1"
DATA = BASE / "sequential-followup-confirmation-data-v1"
TRAINING = {
    "primary_ai": BASE / "ai-observable-calibration-v3",
    "secondary_process": BASE / "dual-reference-process-calibration-v1",
}
INTERRUPTED = BASE / "partition-review-development-v1"
SOURCES = [
    Path(__file__),
    ROOT / "src/gnss_sim/landslide_partition_review.py",
    ROOT / "src/gnss_sim/landslide_baselines.py",
    ROOT / "src/gnss_sim/landslide_adaptive_review.py",
    ROOT / "src/gnss_sim/landslide_visual.py",
    ROOT / "scripts/probe_landslide_revision_scope.py",
    ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py",
]


def calibrate_piece_controls(training, penalty, out):
    """Give numerical aggregation the same partitions and existing validation grid."""
    verify_bindings(training, "training-freeze.json")
    protocol = read_json(training / "protocol.json")
    split = protocol["split"]
    inputs = {
        cid: load_case_features(training, cid) for cid in split["validation"] + split["train"]
    }
    labels = load_labels(
        training / "reference-labels.csv", {cid: pair[0] for cid, pair in inputs.items()}
    )
    model = XGBClassifier()
    model.load_model(training / "primary-baselines/activity-xgboost.json")
    probabilities = {
        cid: model.predict_proba(feature.matrix)[:, 1] for cid, (_, feature) in inputs.items()
    }
    partitions = {}
    for cid, (_, feature) in inputs.items():
        length = len(feature.observed)
        partitions[cid] = {
            "pelt": rate_partition(pelt_segment_rates(feature, penalty), 0, length),
            "uniform30": uniform_partition((length + 29) // 30, 0, length),
        }
    trials, choices = [], {}
    for kind in ("pelt", "uniform30"):
        for threshold in (0.35, 0.5, 0.65):
            predictions = {}
            for cid in split["validation"]:
                feature = inputs[cid][1]
                activity = numeric_piece_activity(
                    probabilities[cid], feature.observed, partitions[cid][kind], threshold
                )
                predictions[cid] = ActivityPrediction(activity, np.where(activity == 0, 0, -1))
            result = summarize(labels, predictions, split["validation"])
            trials.append(
                {
                    "kind": kind,
                    "threshold": threshold,
                    "objective": calibration_objective(result),
                    "validation": result,
                }
            )
        choices[kind] = max(
            (row for row in trials if row["kind"] == kind), key=lambda row: row["objective"]
        )
    out.mkdir(exist_ok=False)
    write_json(out / "validation-trials.json", trials)
    write_json(out / "choices.json", choices)
    write_json(
        out / "provenance.json",
        {
            "training": str(training),
            "split": split,
            "training_freeze_sha256": sha(training / "training-freeze.json"),
            "activity_model_sha256": sha(training / "primary-baselines/activity-xgboost.json"),
            "reference_sha256": sha(training / "reference-labels.csv"),
            "new_model_fitting": False,
            "new_validation_trials": 6,
            "method": "Observed mean of frozen daily XGBoost probabilities per identical temporal proposal; independent three-threshold validation selection for each geometry. No developer-test tuning.",
        },
    )
    return model, choices


def prepare(out):
    verify_manifest(PRIOR)
    verify_bindings(PARENT, "prediction-freeze.json")
    config = read_json(PARENT / "protocol.json")
    out.mkdir(exist_ok=False)
    (out / "source").mkdir()
    for path in SOURCES:
        (out / "source" / path.name).write_bytes(path.read_bytes())
    shutil.copytree(PARENT / "inputs", out / "inputs")
    shutil.copytree(PARENT / "images", out / "images")
    for name in ("selections.json", "stage-xgboost.json"):
        shutil.copyfile(PARENT / name, out / name)
    shutil.copyfile(PRIOR / "primary_ai/predictions.json", out / "prior-predictions.json")
    shutil.copyfile(PARENT / "used-requests.json", out / "prior-used-requests.json")
    penalty = read_json(TRAINING["primary_ai"] / "primary-baselines/frozen-selection.json")["pelt"][
        "penalty"
    ]
    numeric = {}
    controls = {}
    for target, training in TRAINING.items():
        model, choices = calibrate_piece_controls(
            training, penalty, out / (target + "-calibration")
        )
        numeric[target], controls[target] = {}, choices
        for cid in config["case_ids"]:
            _, feature = load_case_features(out, cid)
            numeric[target][cid] = model.predict_proba(feature.matrix)[:, 1].tolist()
    write_json(out / "numeric-probabilities.json", numeric)
    write_json(out / "numeric-control-choices.json", controls)
    choices = read_json(out / "selections.json")
    partitions, full_partitions, plan = {}, {}, []
    order = np.random.default_rng(20261012)
    for cid in config["case_ids"]:
        case, feature = load_case_features(out, cid)
        rates = pelt_segment_rates(feature, penalty)
        full_partitions[cid] = {
            "pelt": rate_partition(rates, 0, len(feature.observed)),
            "uniform30": uniform_partition(
                (len(feature.observed) + 29) // 30, 0, len(feature.observed)
            ),
        }
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            selection = choices[f"{cid}-r{rep}"]
            adaptive = {row["index"] for row in selection["selected"]}
            for index in sorted(adaptive | set(selection["fixed_indices"])):
                window = selection["priorities"][index]
                start, stop = window["start"], window["stop"]
                pelt = rate_partition(rates, start, stop)
                geometries = {
                    "pelt": pelt,
                    "uniform_matched": uniform_partition(len(pelt), start, stop),
                    "uniform30": uniform_partition((stop - start + 29) // 30, start, stop),
                }
                partitions[f"{cid}-{index}"] = geometries
                arms = (
                    ["pelt", "uniform_matched", "uniform30", "no_numeric", "free_norm"]
                    if index in adaptive
                    else ["pelt"]
                )
                for arm in order.permutation(arms):
                    if arm == "free_norm":
                        prompt = review_prompt(feature, long, start, stop, True)
                    else:
                        geometry = "pelt" if arm == "no_numeric" else str(arm)
                        prompt = partition_review_prompt(
                            feature, long, geometries[geometry], arm != "no_numeric"
                        )
                    plan.append(
                        {
                            "request_id": f"partition-probe-{cid}-window-{index:02d}-{arm}-r{rep}",
                            "case_id": cid,
                            "repeat": rep,
                            "arm": str(arm),
                            **window,
                            "prompt": prompt,
                            "images": [f"{cid}-raw.png", f"{cid}-window-{index:02d}.png"],
                        }
                    )
    write_json(out / "partitions.json", partitions)
    write_json(out / "full-partitions.json", full_partitions)
    write_json(out / "query-plan.json", plan)
    # An HTTP 400 run is not a mechanism experiment. Preserve its already
    # completed free-interval controls without issuing those questions again.
    carried = {}
    if out != INTERRUPTED and (INTERRUPTED / "aborted-transport.json").exists():
        verify_bindings(INTERRUPTED, "freeze.json")
        for row in plan:
            receipt_path = INTERRUPTED / "requests" / (row["request_id"] + ".response.json")
            if row["arm"] != "free_norm" or not receipt_path.exists():
                continue
            receipt = read_json(receipt_path)
            if receipt["status"] != "returned_json":
                continue
            expected_images = {name: sha(out / "images" / name) for name in row["images"]}
            if receipt["prompt"] != row["prompt"] or receipt["image_sha256"] != expected_images:
                raise ValueError("Carried control must match the frozen question and pixels")
            carried[row["request_id"]] = receipt
    write_json(out / "carried-controls.json", carried)
    write_json(
        out / "protocol.json",
        {
            "date": "2026-10-11",
            "scope": "All12 developer-exposed0157-0168; two repeats; no new labels/data. Development only.",
            "case_ids": config["case_ids"],
            "model": config["model"],
            "policy": config["policy"],
            "primary": "partition_pelt_r{repeat}__conflict_scope",
            "penalty": penalty,
            "mechanism": "Existing PELT constant-rate/support runs define temporal pieces; split spans above30days, matching existing numeric reporting resolution. VLM labels every whole piece as sustained activity/stationary/unknown rather than inventing interval endpoints. Mixed pieces remain unknown.",
            "feasibility": "Before protocol freeze, original PELT yielded1-5 native runs per180day window;0163 tail was one entire180day piece. Chose existing30day resolution before calls or new scores, not a parameter scan. Feasibility source/outputs retained.",
            "fixed": "Original global answers,raw21/42 conflicts,14 admission,budget2,180day targets,49 authority,pixels,norm evidence,stage head. No signed facts,positive protection,abstention restoration or new numerical activity rule in the primary.",
            "controls": "All68 prior methods (incl six strong target-matched numerical baselines); contemporaneous free-interval norm feedback; uniform30 and same-count uniform piece labeling; PELT without numeric feedback; fixed/first/same-K positions/whole-target authority. Same geometry numerical probability means independently recalibrated on each target's original6/6 split, six validation trials per target.",
            "gate": "Each repeat primaryF1 strictly above fresh free-interval norm; meanF1 and summed net-correct strictly above matched uniform,uniform30,no-numeric and fixed whole-target; meanF1/net-correct not below cheaper first-only; mean weakR and eventF1(.3) not below fresh free norm. Any failure stops candidate; process scores cannot rescue AI-primary failure; no prompt,block,penalty,threshold,repeat or subset rescue.",
            "maximum_new_requests": len(plan),
            "carried_valid_control_receipts": len(carried),
            "pending_new_requests": len(plan) - len(carried),
            "transport_repair": "Prior attempt interrupted for provider HTTP400: prompt lacked literal JSON. Only add JSON to partition output instruction; no scoring or mechanism selection. Carry all exact successful free_norm controls; no repeat inference.",
            "repetitions": 2,
            "retries": 0,
            "hard_timeout_seconds": 1800,
            "request_policy": "temperature.1/top_p.3/max_tokens2048/no_thinking; once-random arm order seed20261012; execute all planned questions once",
            "risks": "Proposed boundaries can reflect centered smoothing, missing fit spans or step artifacts. A mixed temporal piece may lose true activity. Identical adjacent rates merge original PELT edges. These are offline proposals, not exact physical boundaries or independent sensor evidence.",
            "reference": "Frozen author-AI visual/numeric reference and separate process sensitivity; author already saw prior labels/results; not expert/field gold standard.",
        },
    )
    bindings = {str(path): sha(path) for path in SOURCES}
    for path in (
        PRIOR / "manifest.json",
        REFERENCES / "manifest.json",
        PARENT / "freeze.json",
        PARENT / "prediction-freeze.json",
        ORIGINAL / "freeze.json",
        ORIGINAL / "prediction-freeze.json",
    ):
        bindings[str(path)] = sha(path)
    for training in TRAINING.values():
        bindings[str(training / "training-freeze.json")] = sha(training / "training-freeze.json")
    write_json(out / "source-bindings.json", bindings)
    write_json(
        out / "freeze.json",
        {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    save_json(out / "state.json", {"status": "prepared", "completed": 0, "total": len(plan)})
    print(
        {
            "status": "prepared",
            "new_requests": len(plan),
            "penalty": penalty,
            "calibration_trials": 12,
            "new_model_training": False,
        },
        flush=True,
    )


def verify(out):
    verify_bindings(out, "freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Partition experiment bound source changed: " + name)
    for training in TRAINING.values():
        verify_bindings(training, "training-freeze.json")


def review_from_receipt(row, payload, feature, partitions):
    if row["arm"] == "free_norm":
        return parse_window_activity(payload, feature.observed, row["start"], row["stop"])
    kind = "pelt" if row["arm"] == "no_numeric" else row["arm"]
    return parse_piece_activity(
        payload, feature.observed, partitions[f"{row['case_id']}-{row['index']}"][kind]
    )


def build_predictions(out, reviews, target):
    predictions = read_json(PRIOR / target / "predictions.json")
    used = read_json(out / "prior-used-requests.json")
    selections = read_json(out / "selections.json")
    partitions = read_json(out / "partitions.json")
    full = read_json(out / "full-partitions.json")
    probabilities = read_json(out / "numeric-probabilities.json")[target]
    thresholds = read_json(out / "numeric-control-choices.json")[target]
    head = XGBClassifier()
    head.load_model(TRAINING[target] / "stage-xgboost.json")
    config = read_json(out / "protocol.json")
    for cid in config["case_ids"]:
        case, feature = load_case_features(out, cid)
        codes = head.predict(feature.matrix).astype(int) + 1
        probability = np.array(probabilities[cid])
        for kind in ("pelt", "uniform30"):
            activity = numeric_piece_activity(
                probability, feature.observed, full[cid][kind], thresholds[kind]["threshold"]
            )
            predictions.setdefault(f"numeric_piece_{kind}", {})[cid] = learned_stages(
                codes, activity, feature.supported
            ).to_dict()
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            choice = selections[f"{cid}-r{rep}"]
            selected = choice["selected"]
            fixed = [choice["priorities"][index] for index in choice["fixed_indices"]]
            count = len(selected)
            first = (
                [min(selected, key=lambda row: (-row["conflict_days"], row["start"]))]
                if selected
                else []
            )
            targets = {
                "pelt": ("pelt", selected),
                "no_numeric": ("no_numeric", selected),
                "uniform_matched": ("uniform_matched", selected),
                "uniform30": ("uniform30", selected),
                "free_norm": ("free_norm", selected),
                "fixed": ("pelt", fixed),
                "first": ("pelt", first),
                "same_count_early": ("pelt", fixed[:count]),
                "same_count_late": ("pelt", fixed[len(fixed) - count :] if count else []),
            }
            authority = conflict_neighbourhood(
                initial, feature, long, ReviewPolicy(**config["policy"])
            )[1]
            for stem, (arm, windows) in targets.items():
                keys = [
                    f"partition-probe-{cid}-window-{row['index']:02d}-{arm}-r{rep}"
                    for row in windows
                ]
                local = [
                    dict(reviews[key], activity=np.array(reviews[key]["activity"])) for key in keys
                ]
                reviewed = combine_reviews(initial, local, feature.observed).activity
                scope = target_scopes(authority, feature.observed, windows)[0]
                variants = {
                    "conflict_scope": scoped_activity(initial, reviewed, scope, feature.observed)
                }
                if stem in (
                    "pelt",
                    "no_numeric",
                    "uniform_matched",
                    "uniform30",
                    "free_norm",
                    "fixed",
                ):
                    variants["whole_target"] = reviewed
                for variant, activity in variants.items():
                    method = f"partition_{stem}_r{rep}__{variant}"
                    predictions.setdefault(method, {})[cid] = learned_stages(
                        codes, activity, feature.supported
                    ).to_dict()
                    used.setdefault(method, {})[cid] = [f"{cid}-direct-r{rep}"] + keys
            numeric_reviews = []
            for window in selected:
                pieces = partitions[f"{cid}-{window['index']}"]["pelt"]
                numeric_reviews.append(
                    {
                        "start": window["start"],
                        "stop": window["stop"],
                        "status": "success",
                        "activity": numeric_piece_activity(
                            probability, feature.observed, pieces, thresholds["pelt"]["threshold"]
                        ),
                    }
                )
            numeric_reviewed = combine_reviews(initial, numeric_reviews, feature.observed).activity
            scope = target_scopes(authority, feature.observed, selected)[0]
            method = f"numeric_piece_selected_r{rep}"
            predictions.setdefault(method, {})[cid] = learned_stages(
                codes,
                scoped_activity(initial, numeric_reviewed, scope, feature.observed),
                feature.supported,
            ).to_dict()
            used.setdefault(method, {})[cid] = [f"{cid}-direct-r{rep}"]
    return predictions, used


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only execute a never-started prepared plan; no repeats")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    partitions = read_json(out / "partitions.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], len(plan))
    carried = read_json(out / "carried-controls.json")
    for request_id in carried:
        for suffix in (".started.json", ".response.json"):
            shutil.copyfile(
                INTERRUPTED / "requests" / (request_id + suffix),
                out / "requests" / (request_id + suffix),
            )
    reviews, failures = {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "completed": 0, "total": len(plan)})
    for index, row in enumerate(plan):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            save_json(
                out / "state.json", {"status": "timeout", "completed": index, "total": len(plan)}
            )
            raise RuntimeError("Registered wall-time limit exceeded; do not restart")
        _, feature = load_case_features(out, row["case_id"])
        activity = np.full(len(feature.observed), -1, dtype=int)
        status = "success"
        try:
            if row["request_id"] in carried:
                payload = json.loads(carried[row["request_id"]]["output"])
            else:
                payload = transport.ask(
                    row["request_id"],
                    row["prompt"],
                    [out / "images" / name for name in row["images"]],
                )
            activity = review_from_receipt(row, payload, feature, partitions)
        except (RuntimeError, ValueError) as error:
            status = "failed"
            failures[row["request_id"]] = str(error)
        reviews[row["request_id"]] = {
            "start": row["start"],
            "stop": row["stop"],
            "activity": activity.tolist(),
            "status": status,
        }
        save_json(out / "reviews.json", reviews)
        save_json(out / "failures.json", failures)
        save_json(
            out / "state.json",
            {
                "status": "running",
                "completed": index + 1,
                "total": len(plan),
                "seconds": time.monotonic() - started,
            },
        )
        print(f"Partition review {index + 1}/{len(plan)}: {row['request_id']} {status}", flush=True)
    predictions, used = build_predictions(out, reviews, "primary_ai")
    process, process_used = build_predictions(out, reviews, "secondary_process")
    if used != process_used:
        raise ValueError("Target changes must not change actual visual request attribution")
    write_json(out / "predictions.json", predictions)
    write_json(out / "process-predictions.json", process)
    write_json(out / "used-requests.json", used)
    verify(out)
    write_json(
        out / "prediction-freeze.json",
        {
            name: sha(out / name)
            for name in (
                "reviews.json",
                "failures.json",
                "predictions.json",
                "process-predictions.json",
                "used-requests.json",
            )
        },
    )
    save_json(
        out / "state.json",
        {
            "status": "predictions_complete",
            "completed": len(plan),
            "total": len(plan),
            "seconds": time.monotonic() - started,
        },
    )


def decision_for_target(labels, observed, predictions, summaries):
    values = {row["method"]: row for row in summaries}
    comparisons, corrections = [], []
    for stem, variant in (
        ("free_norm", "conflict_scope"),
        ("uniform_matched", "conflict_scope"),
        ("uniform30", "conflict_scope"),
        ("no_numeric", "conflict_scope"),
        ("fixed", "whole_target"),
        ("first", "conflict_scope"),
    ):
        differences, weak_differences, event_differences, net = [], [], [], 0
        for rep in (0, 1):
            main = f"partition_pelt_r{rep}__conflict_scope"
            control = f"partition_{stem}_r{rep}__{variant}"
            differences.append(values[main]["f1"] - values[control]["f1"])
            weak_differences.append(values[main]["weak_recall"] - values[control]["weak_recall"])
            event_differences.append(
                values[main]["event_f1_iou03"] - values[control]["event_f1_iou03"]
            )
            for cid, rows in labels.items():
                counts = correction_utility(
                    np.array([row["activity_label"] for row in rows]),
                    np.array(predictions[control][cid]["activity"]),
                    np.array(predictions[main][cid]["activity"]),
                    observed[cid],
                )
                net += counts["net_correct"]
                corrections.append({"case_id": cid, "main": main, "control": control, **counts})
        if stem == "free_norm":
            passed = (
                all(delta > 0 for delta in differences)
                and np.mean(weak_differences) >= 0
                and np.mean(event_differences) >= 0
            )
        elif stem == "first":
            passed = np.mean(differences) >= 0 and net >= 0
        else:
            passed = np.mean(differences) > 0 and net > 0
        comparisons.append(
            {
                "control": stem,
                "f1_difference_by_repeat": differences,
                "mean_f1_difference": float(np.mean(differences)),
                "mean_weak_recall_difference": float(np.mean(weak_differences)),
                "mean_event_f1_difference": float(np.mean(event_differences)),
                "net_correct_both_repeats": net,
                "passed": bool(passed),
            }
        )
    return {
        "passed": all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
    }, corrections


def score(out, destination):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete all predictions before scoring")
    destination.mkdir(exist_ok=False)
    receipts = {}
    for source in (ORIGINAL, PARENT, out):
        extra = verify_receipts(source)
        if set(extra) & set(receipts):
            raise ValueError("Shared actual receipt ids collide")
        receipts.update(extra)
    reviews = read_json(out / "reviews.json")
    partitions = read_json(out / "partitions.json")
    for row in read_json(out / "query-plan.json"):
        receipt = receipts[row["request_id"]]
        _, feature = load_case_features(out, row["case_id"])
        try:
            if receipt["status"] != "returned_json":
                raise ValueError("Failed response")
            activity = review_from_receipt(row, json.loads(receipt["output"]), feature, partitions)
            status = "success"
        except ValueError:
            activity = np.full(len(feature.observed), -1, dtype=int)
            status = "failed"
        if (
            reviews[row["request_id"]]["activity"] != activity.tolist()
            or reviews[row["request_id"]]["status"] != status
        ):
            raise ValueError("Review does not reconstruct from actual receipt")
    authored = read_json(DATA / "codex-review/activity-decisions.json")[
        "weak_interiors_authored_before_predictions"
    ]
    observed = {
        cid: load_case_features(out, cid)[1].observed
        for cid in read_json(out / "protocol.json")["case_ids"]
    }
    tables, decisions, correction_rows = {}, {}, []
    for target, filename in (
        ("primary_ai", "predictions.json"),
        ("secondary_process", "process-predictions.json"),
    ):
        predictions, used = build_predictions(out, reviews, target)
        if predictions != read_json(out / filename) or used != read_json(
            out / "used-requests.json"
        ):
            raise ValueError("All target predictions and request costs must replay exactly")
        labels = read_json(REFERENCES / target / "reference-labels.json")
        weak = {}
        for cid, rows in labels.items():
            if target == "primary_ai":
                mask = np.zeros(len(rows), bool)
                for start, stop in authored.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
            else:
                weak[cid] = np.array(
                    [row["weak_signal"] == 1 and row["activity_label"] == 1 for row in rows]
                )
        results, summaries, records = score_target(
            labels, observed, weak, predictions, used, receipts
        )
        folder = destination / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        decisions[target], corrections = decision_for_target(
            labels, observed, predictions, summaries
        )
        correction_rows.extend(dict(row, target=target) for row in corrections)
        tables[target] = summaries
    decisions["candidate"] = (
        "CONTINUE_TO_NEW_CONFIRMATION" if decisions["primary_ai"]["passed"] else "STOPPED"
    )
    decisions["default_changed"] = False
    write_json(destination / "decision.json", decisions)
    write_csv(destination / "corrections.csv", correction_rows)
    new_receipts = verify_receipts(out)
    carried = read_json(out / "carried-controls.json")
    fresh_receipts = {key: row for key, row in new_receipts.items() if key not in carried}
    write_json(
        destination / "cost.json",
        {
            "new_launches": len(list((out / "requests").glob("*.started.json"))) - len(carried),
            "new_receipts": len(fresh_receipts),
            "all_plan_receipts": len(new_receipts),
            "carried_controls": len(carried),
            "reported_tokens": sum(
                row.get("usage", {}).get("total_tokens", 0) for row in fresh_receipts.values()
            ),
            "seconds": read_json(out / "state.json")["seconds"],
            "failures": len(read_json(out / "failures.json")),
            "historical_receipts_reused": len(receipts) - len(fresh_receipts),
            "new_records": 0,
            "new_model_training": False,
            "new_validation_trials": 12,
        },
    )
    text = "# PELT候选时间段的视觉分类：完整开发试验\n\n## Material Passport\n\n状态：所有推理完成；确定性预测重建与独立逐日计数通过。参考：已暴露作者AI与过程敏感性。不是专家或现场独立验证。\n\n"
    text += f"冻结继续条件结果：**{decisions['candidate']}**。\n\n"
    for target, summaries in tables.items():
        text += f"## {target}\n\n" + full_table(summaries) + "\n\n"
    (destination / "report.md").write_text(text, encoding="utf-8")
    write_json(
        destination / "manifest.json",
        {
            "files": {
                path.relative_to(destination).as_posix(): sha(path)
                for path in destination.rglob("*")
                if path.is_file()
            }
        },
    )
    print(
        {
            "decision": decisions["candidate"],
            "method_target_scores": sum(len(rows) for rows in tables.values()),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    elif args.action == "run":
        execute(args.out.resolve())
    else:
        if args.analysis is None:
            parser.error("score requires --analysis")
        score(args.out.resolve(), args.analysis.resolve())
