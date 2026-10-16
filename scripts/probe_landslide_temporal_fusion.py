"""Calibrate a minimal temporal evidence fusion and compare every frozen method."""

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
from probe_landslide_partition_review import DATA, ORIGINAL, PARENT, REFERENCES, TRAINING
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, load_labels, save_json, summarize
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_dual_reference_report import full_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_temporal_fusion import decode_activity_scores, fused_motion_scores
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
PRIOR_RUN = BASE / "partition-review-development-v2"
PRIOR_ANALYSIS = BASE / "partition-review-development-analysis-v2"
TRANSITION_COST = 7 * np.log(2)
SOURCES = [
    Path(__file__),
    ROOT / "src/gnss_sim/landslide_temporal_fusion.py",
    ROOT / "src/gnss_sim/landslide_visual.py",
    ROOT / "src/gnss_sim/landslide_evaluation.py",
    ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py",
    ROOT / "scripts/run_landslide_visual_comparison.py",
]


def prepare(out):
    verify_manifest(PRIOR_ANALYSIS)
    verify_bindings(PRIOR_RUN, "prediction-freeze.json")
    for training in TRAINING.values():
        verify_bindings(training, "training-freeze.json")
    out.mkdir(exist_ok=False)
    (out / "source").mkdir()
    for path in SOURCES:
        shutil.copyfile(path, out / "source" / path.name)
    shutil.copytree(PRIOR_RUN / "inputs", out / "inputs")
    shutil.copytree(PRIOR_RUN / "images", out / "images")
    train_protocol = read_json(TRAINING["primary_ai"] / "protocol.json")
    validation = train_protocol["split"]["validation"]
    for cid in validation:
        shutil.copytree(TRAINING["primary_ai"] / "inputs" / cid, out / "inputs" / cid)
        render_raw(load_case_features(out, cid)[0], out / "images" / (cid + "-raw.png"))
    config = read_json(PRIOR_RUN / "protocol.json")
    plan = []
    for rep in (0, 1):
        for cid in validation:
            case, _ = load_case_features(out, cid)
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200]."
            )
            plan.append(
                {
                    "request_id": f"temporal-calibration-{cid}-r{rep}",
                    "case_id": cid,
                    "repeat": rep,
                    "prompt": prompt,
                    "images": [cid + "-raw.png"],
                }
            )
    probabilities, thresholds = {}, {}
    for target, training in TRAINING.items():
        if read_json(training / "protocol.json")["split"] != train_protocol["split"]:
            raise ValueError("Both numerical targets must use the same original record split")
        model = XGBClassifier()
        model.load_model(training / "primary-baselines/activity-xgboost.json")
        probabilities[target] = {
            cid: model.predict_proba(load_case_features(out, cid)[1].matrix)[:, 1].tolist()
            for cid in validation + config["case_ids"]
        }
        thresholds[target] = read_json(training / "primary-baselines/frozen-selection.json")[
            "xgboost"
        ]["threshold"]
    write_json(out / "numeric-probabilities.json", probabilities)
    write_json(out / "base-thresholds.json", thresholds)
    write_json(out / "query-plan.json", plan)
    write_json(
        out / "protocol.json",
        {
            "date": "2026-10-11",
            "case_ids": config["case_ids"],
            "validation_ids": validation,
            "training_split": train_protocol["split"],
            "model": config["model"],
            "new_requests": len(plan),
            "new_model_fitting": False,
            "repeats": 2,
            "retries": 0,
            "transition_cost": float(TRANSITION_COST),
            "transition_basis": "Fixed engineering cost7*ln2: under illustrative odds2, an interior14-day motion span offsets its two transition costs. This is neither a minimum-duration guarantee nor a physical threshold. No duration/penalty scan.",
            "unary": "supported numerical logit(p)-logit(frozen threshold), plus weight times signed VLM state. Unknown visual contributes zero; unsupported numerical contributes zero. VLM votes are not probabilities.",
            "unknown": "Missing and jointly unsupported dates split chains; no gap interpolation. Equal globally optimal state costs remain unknown, tolerance1e-12 only for arithmetic ties. This is not posterior uncertainty or physical confidence.",
            "calibration": "Mean original dailyF1+eventF1 across two validation repeats only. Joint and no-temporal arms independently choose VLM weight .5/1/2 with frozen numerical threshold. Numeric-temporal and numeric-point independently choose original .35/.5/.65 thresholds. Twelve validation trials per reference target;24 total. No developer-test selection.",
            "gate": "Each repeat jointF1 strictly above numeric-temporal; meanF1 and net-correct strictly above no-temporal and direct VLM; mean weakR and eventF1(.3) not below numeric-temporal. Any failure stops, no parameter/prompt/repeat/subset rescue. All102 prior methods and strong numerical controls retained; process target cannot rescue primary failure.",
            "scope": "Exposed0157-0168 development only; six original0049-0060 validation records for weights. No new test records/labels. Test initial VLM answers reused exactly; no re-inference. Native stage head unchanged.",
            "hard_timeout_seconds": 600,
        },
    )
    bindings = {str(path): sha(path) for path in SOURCES}
    for path in (
        PRIOR_ANALYSIS / "manifest.json",
        PRIOR_RUN / "prediction-freeze.json",
        REFERENCES / "manifest.json",
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
        {"status": "prepared", "new_validation_calls": len(plan), "test_calls": 0, "trials": 24},
        flush=True,
    )


def verify(out):
    verify_bindings(out, "freeze.json")
    verify_manifest(PRIOR_ANALYSIS)
    verify_manifest(REFERENCES)
    verify_bindings(PRIOR_RUN, "prediction-freeze.json")
    for path, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Bound fusion source changed")
    for training in TRAINING.values():
        verify_bindings(training, "training-freeze.json")


def activity_for(feature, probability, visual, threshold, weight, cost):
    scores, available = fused_motion_scores(
        probability, feature.supported, visual, feature.observed, threshold, weight
    )
    return decode_activity_scores(scores, available, cost)


def calibrate(out, visual):
    config = read_json(out / "protocol.json")
    ids = config["validation_ids"]
    features = {cid: load_case_features(out, cid)[1] for cid in ids}
    probability = read_json(out / "numeric-probabilities.json")
    thresholds = read_json(out / "base-thresholds.json")
    choices, trials = {}, []
    for target, training in TRAINING.items():
        all_cases = {
            cid: load_case_features(training, cid)[0]
            for cid in config["training_split"]["train"] + ids
        }
        labels = load_labels(training / "reference-labels.csv", all_cases)
        choices[target] = {}
        for arm in ("joint", "no_temporal", "numeric_temporal", "numeric_point"):
            for candidate in (0.35, 0.5, 0.65) if arm.startswith("numeric") else (0.5, 1.0, 2.0):
                numeric = arm.startswith("numeric")
                threshold = candidate if numeric else thresholds[target]
                weight = 0 if numeric else candidate
                cost = config["transition_cost"] if arm in ("joint", "numeric_temporal") else 0
                repetitions = []
                for rep in (0, 1):
                    predictions = {}
                    for cid in ids:
                        activity = activity_for(
                            features[cid],
                            np.array(probability[target][cid]),
                            np.array(visual[f"{cid}-r{rep}"]),
                            threshold,
                            weight,
                            cost,
                        )
                        predictions[cid] = ActivityPrediction(
                            activity, np.where(activity == 0, 0, -1)
                        )
                    repetitions.append(summarize(labels, predictions, ids))
                trials.append(
                    {
                        "target": target,
                        "arm": arm,
                        "threshold": threshold,
                        "weight": weight,
                        "cost": cost,
                        "objective": float(
                            np.mean([calibration_objective(row) for row in repetitions])
                        ),
                        "validation_by_repeat": repetitions,
                    }
                )
            choices[target][arm] = max(
                (row for row in trials if row["target"] == target and row["arm"] == arm),
                key=lambda row: row["objective"],
            )
    return choices, trials


def build_predictions(out, choices, target):
    predictions = read_json(PRIOR_ANALYSIS / target / "predictions.json")
    used = read_json(PRIOR_RUN / "used-requests.json")
    probabilities = read_json(out / "numeric-probabilities.json")[target]
    head = XGBClassifier()
    head.load_model(TRAINING[target] / "stage-xgboost.json")
    for cid in read_json(out / "protocol.json")["case_ids"]:
        _, feature = load_case_features(out, cid)
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            visual = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            for arm in (
                "joint",
                "no_temporal",
                "numeric_temporal",
                "numeric_point",
                "visual_temporal",
            ):
                selected = choices[target]["joint" if arm == "visual_temporal" else arm]
                probability = np.array(probabilities[cid])
                support = feature.supported
                if arm == "visual_temporal":
                    scores, available = fused_motion_scores(
                        probability,
                        np.zeros_like(support),
                        visual,
                        feature.observed,
                        selected["threshold"],
                        selected["weight"],
                    )
                    activity = decode_activity_scores(scores, available, selected["cost"])
                else:
                    activity = activity_for(
                        feature,
                        probability,
                        visual,
                        selected["threshold"],
                        selected["weight"],
                        selected["cost"],
                    )
                name = f"temporal_{arm}" if arm.startswith("numeric") else f"temporal_{arm}_r{rep}"
                prediction = learned_stages(codes, activity, support).to_dict()
                if (
                    arm.startswith("numeric")
                    and name in predictions
                    and predictions[name].get(cid, prediction) != prediction
                ):
                    raise ValueError("No-visual control must not depend on visual repetition")
                predictions.setdefault(name, {})[cid] = prediction
                used.setdefault(name, {})[cid] = (
                    [] if arm.startswith("numeric") else [f"{cid}-direct-r{rep}"]
                )
    return predictions, used


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only execute a never-started calibration; no retries")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], len(plan))
    visual, failures = {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "completed": 0})
    for index, row in enumerate(plan):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            save_json(out / "state.json", {"status": "timeout", "completed": index})
            raise RuntimeError("Frozen timeout exceeded; do not restart")
        _, feature = load_case_features(out, row["case_id"])
        activity = np.full(len(feature.observed), -1, dtype=int)
        try:
            payload = transport.ask(
                row["request_id"], row["prompt"], [out / "images" / name for name in row["images"]]
            )
            activity = parse_activity(payload, feature.observed)
        except (RuntimeError, ValueError) as error:
            failures[row["request_id"]] = str(error)
        visual[f"{row['case_id']}-r{row['repeat']}"] = activity.tolist()
        save_json(out / "visual-validation.json", visual)
        save_json(out / "failures.json", failures)
        save_json(
            out / "state.json", {"status": "running", "completed": index + 1, "total": len(plan)}
        )
        print(f"Fusion validation {index + 1}/{len(plan)}: {row['request_id']}", flush=True)
    choices, trials = calibrate(out, visual)
    write_json(out / "choices.json", choices)
    write_json(out / "validation-trials.json", trials)
    write_json(
        out / "calibration-freeze.json",
        {
            name: sha(out / name)
            for name in (
                "choices.json",
                "validation-trials.json",
                "visual-validation.json",
                "failures.json",
            )
        },
    )
    for target in TRAINING:
        predictions, used = build_predictions(out, choices, target)
        write_json(out / (target + "-predictions.json"), predictions)
        write_json(out / (target + "-used.json"), used)
    verify(out)
    write_json(
        out / "prediction-freeze.json",
        {
            name: sha(out / name)
            for name in (
                "primary_ai-predictions.json",
                "secondary_process-predictions.json",
                "primary_ai-used.json",
                "secondary_process-used.json",
            )
        },
    )
    save_json(
        out / "state.json",
        {
            "status": "predictions_complete",
            "completed": len(plan),
            "seconds": time.monotonic() - started,
        },
    )


def score(out, destination):
    verify(out)
    verify_bindings(out, "calibration-freeze.json")
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete calibration and predictions before scoring")
    destination.mkdir(exist_ok=False)
    validation_receipts = verify_receipts(out)
    saved_visual = read_json(out / "visual-validation.json")
    for row in read_json(out / "query-plan.json"):
        receipt = validation_receipts[row["request_id"]]
        _, feature = load_case_features(out, row["case_id"])
        if row["prompt"] != receipt["prompt"] or list(receipt["image_sha256"]) != row["images"]:
            raise ValueError("Validation question changed")
        try:
            if receipt["status"] != "returned_json":
                raise ValueError("Failed transport")
            activity = parse_activity(json.loads(receipt["output"]), feature.observed)
        except ValueError:
            activity = np.full(len(feature.observed), -1, dtype=int)
        if activity.tolist() != saved_visual[f"{row['case_id']}-r{row['repeat']}"]:
            raise ValueError("Validation visual answer cannot be reconstructed")
    choices, trials = calibrate(out, saved_visual)
    if choices != read_json(out / "choices.json") or trials != read_json(
        out / "validation-trials.json"
    ):
        raise ValueError("Parameter selection must reproduce from validation only")
    receipts = {}
    for source in (ORIGINAL, PARENT, PRIOR_RUN):
        extra = verify_receipts(source)
        if receipts.keys() & extra.keys():
            raise ValueError("Prior receipts collide")
        receipts.update(extra)
    config = read_json(out / "protocol.json")
    observed = {cid: load_case_features(out, cid)[1].observed for cid in config["case_ids"]}
    authored = read_json(DATA / "codex-review/activity-decisions.json")[
        "weak_interiors_authored_before_predictions"
    ]
    decisions, corrections, tables = {}, [], {}
    for target in TRAINING:
        predictions, used = build_predictions(out, choices, target)
        if predictions != read_json(out / (target + "-predictions.json")) or used != read_json(
            out / (target + "-used.json")
        ):
            raise ValueError("All predictions and cost attribution must replay exactly")
        labels = read_json(REFERENCES / target / "reference-labels.json")
        weak = {}
        for cid, rows in labels.items():
            actual = np.array([row["activity_label"] for row in rows])
            mask = np.zeros(len(rows), bool)
            if target == "primary_ai":
                for start, stop in authored.get(cid, []):
                    mask[start:stop] = True
            else:
                mask = np.array([row["weak_signal"] == 1 for row in rows])
            weak[cid] = mask & (actual == 1)
        results, summaries, records = score_target(
            labels, observed, weak, predictions, used, receipts
        )
        folder = destination / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        values = {row["method"]: row for row in summaries}
        comparisons = []
        for control in (
            "temporal_numeric_temporal",
            "temporal_no_temporal_r{}",
            "direct_visual_r{}",
        ):
            differences, weak_deltas, event_deltas, net = [], [], [], 0
            for rep in (0, 1):
                main = f"temporal_joint_r{rep}"
                opponent = control.format(rep)
                differences.append(values[main]["f1"] - values[opponent]["f1"])
                weak_deltas.append(values[main]["weak_recall"] - values[opponent]["weak_recall"])
                event_deltas.append(
                    values[main]["event_f1_iou03"] - values[opponent]["event_f1_iou03"]
                )
                for cid, reference in labels.items():
                    counts = correction_utility(
                        np.array([row["activity_label"] for row in reference]),
                        np.array(predictions[opponent][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]),
                        observed[cid],
                    )
                    net += counts["net_correct"]
                    corrections.append(
                        {
                            "target": target,
                            "case_id": cid,
                            "main": main,
                            "control": opponent,
                            **counts,
                        }
                    )
            if control == "temporal_numeric_temporal":
                passed = (
                    all(delta > 0 for delta in differences)
                    and np.mean(weak_deltas) >= 0
                    and np.mean(event_deltas) >= 0
                )
            else:
                passed = np.mean(differences) > 0 and net > 0
            comparisons.append(
                {
                    "control": control,
                    "f1_deltas": differences,
                    "mean_f1_delta": float(np.mean(differences)),
                    "mean_weak_delta": float(np.mean(weak_deltas)),
                    "mean_event_delta": float(np.mean(event_deltas)),
                    "net_correct": net,
                    "passed": bool(passed),
                }
            )
        decisions[target] = {
            "passed": all(row["passed"] for row in comparisons),
            "comparisons": comparisons,
        }
        tables[target] = summaries
    decisions["candidate"] = (
        "CONTINUE_TO_NEW_CONFIRMATION" if decisions["primary_ai"]["passed"] else "STOPPED"
    )
    write_json(destination / "decision.json", decisions)
    write_csv(destination / "corrections.csv", corrections)
    write_json(
        destination / "cost.json",
        {
            "new_launches": len(list((out / "requests").glob("*.started.json"))),
            "new_receipts": len(validation_receipts),
            "new_validation_calls": len(validation_receipts),
            "new_test_calls": 0,
            "reported_tokens": sum(
                row.get("usage", {}).get("total_tokens", 0) for row in validation_receipts.values()
            ),
            "seconds": read_json(out / "state.json")["seconds"],
            "failures": len(read_json(out / "failures.json")),
            "historical_receipts_reused": len(receipts),
            "new_model_fitting": False,
            "validation_trials": len(trials),
        },
    )
    text = "# 连续运动证据与VLM区间的时间解码\n\n## Material Passport\n\n已暴露开发记录；原6训练/6验证选择；不属于独立专家或现场确认。\n\n"
    text += f"冻结决定：**{decisions['candidate']}**。\n\n"
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
    parser.add_argument("--out", required=True, type=Path)
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
