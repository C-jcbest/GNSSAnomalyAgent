"""Score shared visual responses against two predeclared, target-matched references."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from check_landslide_reference_process_agreement import classify_day, summarize
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import load_labels, predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "dual-reference-confirmation-data-v1"
COMPARISON = BASE / "dual-reference-confirmation-comparison-v2"
TRAINING = BASE / "dual-reference-process-calibration-v1"


def prepare_process(out):
    """Predict from the process-target models without opening new process labels."""
    verify_bindings(DATA, "data-freeze.json")
    verify_bindings(TRAINING, "training-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    ids = read_json(DATA / "protocol.json")["case_ids"]
    models = {}
    for name, filename in {
        "activity": "primary-baselines/activity-xgboost.json",
        "stage": "stage-xgboost.json",
        "persistence": "raw-baselines/activity-xgboost-raw-persistence.json",
    }.items():
        models[name] = XGBClassifier()
        models[name].load_model(TRAINING / filename)
    cases, features = {}, {}
    for cid in ids:
        cases[cid], features[cid] = load_case_features(DATA, cid)
    started = time.perf_counter()
    numerical = predict_baselines(features, ids,
        read_json(TRAINING / "primary-baselines/frozen-selection.json"),
        models["activity"], models["stage"])
    numerical.pop("ungated_rule")
    choices = read_json(TRAINING / "raw-baselines/frozen-selection.json")
    numerical.update({name: {} for name in choices})
    codes = {}
    for cid in ids:
        feature = features[cid]
        long = raw_motion_evidence(cases[cid], feature)
        matrix = np.column_stack((feature.matrix, long["rate"], long["signal_to_noise"], long["direction"]))
        rule = choices["raw_multiscale_rule"]
        gates = {
            "raw_multiscale_rule": raw_gate(feature, long, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                models["persistence"].predict_proba(matrix)[:, 1]
                >= choices["xgboost_raw_persistence"]["threshold"], feature.supported),
        }
        codes[cid] = models["stage"].predict(feature.matrix).astype(int) + 1
        for method, activity in gates.items():
            numerical[method][cid] = learned_stages(codes[cid], activity, feature.supported)
    write_json(out / "numeric-predictions.json", {
        method: {cid: value.to_dict() for cid, value in records.items()}
        for method, records in numerical.items()})
    write_json(out / "stage-codes.json", {cid: value.tolist() for cid, value in codes.items()})
    write_json(out / "provenance.json", {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "test_process_labels_read": False,
        "training_freeze_sha256": sha(TRAINING / "training-freeze.json"),
        "data_freeze_sha256": sha(DATA / "data-freeze.json"),
        "six_controls_seconds": time.perf_counter() - started,
        "timing_scope": "Six predictions plus persistence derivation; excludes feature loading and fitting",
        "new_remote_requests": 0,
    })
    write_json(out / "prediction-freeze.json", {
        path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})


def score_target(labels, observed, weak, predictions, used, receipts):
    """Apply the common task, checking day counts independently with sklearn."""
    results, summaries, records = {}, [], []
    for method, cases in predictions.items():
        if set(cases) != set(labels):
            raise ValueError("Every method must score every record")
        per_case, bounds03, bounds05, scores05 = {}, {}, {}, []
        actual_all, predicted_all, actual_stage, predicted_stage = [], [], [], []
        weak_total = weak_detected = unknown_positive = unknown_negative = unknown_abstained = 0
        for cid, value in cases.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            if np.any(prediction.activity[~observed[cid]] != -1) or np.any(prediction.stage[~observed[cid]] != -1):
                raise ValueError("Missing observation gained a decision")
            per_case[cid] = evaluate_case(labels[cid], prediction)
            scores05.append(evaluate_case(labels[cid], prediction, .5))
            bounds03[cid] = boundary_errors(labels[cid], prediction.activity, .3)
            bounds05[cid] = boundary_errors(labels[cid], prediction.activity, .5)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            actual_all.extend(actual[known].tolist())
            predicted_all.extend(prediction.activity[known].tolist())
            stages = stage_reference_arrays(labels[cid], prediction)
            interior = stages["stage_interior"]
            actual_stage.extend(stages["actual_stage"][interior].tolist())
            predicted_stage.extend(prediction.stage[interior].tolist())
            found = int((weak[cid] & (prediction.activity == 1)).sum())
            weak_detected += found
            weak_total += int(weak[cid].sum())
            unknown = observed[cid] & (actual == -1)
            unknown_positive += int((unknown & (prediction.activity == 1)).sum())
            unknown_negative += int((unknown & (prediction.activity == 0)).sum())
            unknown_abstained += int((unknown & (prediction.activity == -1)).sum())
            records.append({"method": method, "case_id": cid, **per_case[cid]["daily"],
                "weak_days": int(weak[cid].sum()), "weak_detected_days": found})
        aggregate = aggregate_scores(list(per_case.values()))
        p, r, f, _ = precision_recall_fscore_support(actual_all, predicted_all, labels=[1], zero_division=0)
        for metric, independent in (("precision", p[0]), ("recall", r[0]), ("f1", f[0])):
            if not np.isclose(aggregate["daily"][metric], independent):
                raise ValueError("Independent day counter disagrees")
        ip, ir, sf, support = precision_recall_fscore_support(
            actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)
        interior_score = {"precision_ASD": ip.tolist(), "recall_ASD": ir.tolist(),
            "f1_ASD": sf.tolist(), "support_ASD": support.tolist(), "macro_f1": float(sf.mean()),
            "coverage": float(np.mean(np.array(predicted_stage) > 0))}
        event05 = aggregate_scores(scores05)["events"]
        pairs = [pair for row in bounds03.values() for pair in row["matched_pairs"]]
        if len(pairs) != aggregate["events"]["tp"] or sum(len(row["matched_pairs"])
                for row in bounds05.values()) != event05["tp"]:
            raise ValueError("Event and boundary counts differ")
        results[method] = {"aggregate": aggregate, "per_case": per_case, "events_iou05": event05,
            "internal_stages": interior_score, "boundaries_iou03": bounds03, "boundaries_iou05": bounds05}
        keys = [key for values in used.get(method, {}).values() for key in values]
        summaries.append({"method": method, **aggregate["daily"],
            "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": event05["f1"],
            "onset_core_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs])) if pairs else None,
            "offset_core_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs])) if pairs else None,
            "stage_end_to_end_macro_f1": aggregate["stage_macro_f1"],
            "stage_internal_macro_f1": interior_score["macro_f1"],
            "stage_internal_coverage": interior_score["coverage"], "coverage": aggregate["coverage"],
            "weak_days": weak_total, "weak_detected_days": weak_detected,
            "weak_recall": weak_detected / weak_total,
            "unknown_reference_predicted_positive": unknown_positive,
            "unknown_reference_predicted_negative": unknown_negative,
            "unknown_reference_abstained": unknown_abstained,
            "requests": len(keys), "reported_tokens": sum(receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys)})
    return results, summaries, records


def comparisons(labels, observed, predictions, results, summaries):
    table = {row["method"]: row for row in summaries}
    corrections, intervals, registration = [], [], []
    for rep in (0, 1):
        main = f"adaptive_no_feedback_r{rep}__conflict_scope"
        controls = [f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
            f"same_count_fixed_early_r{rep}__conflict_scope", f"same_count_fixed_late_r{rep}__conflict_scope",
            f"adaptive_no_feedback_r{rep}__whole_target", f"adaptive_no_feedback_r{rep}__matched_center_scope",
            f"adaptive_numeric_feedback_r{rep}__conflict_scope", "xgboost", "xgboost_raw_persistence", "raw_multiscale_rule"]
        for control in controls:
            counts = [correction_utility(np.array([int(row["activity_label"]) for row in labels[cid]]),
                np.array(predictions[control][cid]["activity"]), np.array(predictions[main][cid]["activity"]), observed[cid])
                for cid in labels]
            total = {key: sum(row[key] for row in counts)
                for key in ("eligible_days", "gained_correct", "lost_correct", "net_correct")}
            corrections.append({"control": control, "treatment": main, **total})
            intervals.append({"control": control, "treatment": main, **paired_record_interval(results, control, main)})
            if control.startswith("same_count_fixed_"):
                conditions = {"f1_gain": table[main]["f1"] > table[control]["f1"],
                    "weak_recall_not_lower": table[main]["weak_recall"] >= table[control]["weak_recall"],
                    "positive_net_correct": total["net_correct"] > 0}
                registration.append({"control": control, "treatment": main, **conditions,
                    "passed": all(conditions.values()), "net_correct": total["net_correct"]})
    return corrections, intervals, registration


def analyze(process_predictions, out):
    primary = COMPARISON / "analysis"
    verify_manifest(primary)
    verify_bindings(process_predictions, "prediction-freeze.json")
    verify_bindings(DATA, "data-freeze.json")
    verify_bindings(TRAINING, "training-freeze.json")
    verify_manifest(DATA / "codex-review", "delivery-manifest.json")
    if read_json(COMPARISON / "run/state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all visual predictions before opening process test labels")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    write_json(out / "before-process-label-read.json", {
        "opened_after_primary_reference_and_all_predictions_frozen": True,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "primary_prediction_freeze_sha256": sha(primary / "prediction-freeze.json"),
        "secondary_prediction_freeze_sha256": sha(process_predictions / "prediction-freeze.json"),
        "reference_delivery_sha256": sha(DATA / "codex-review/delivery-manifest.json"),
        "data_protocol_sha256": sha(DATA / "protocol.json"),
    })
    ai, observed, _ = scoring_inputs(DATA / "codex-review")
    case_features = {cid: load_case_features(DATA, cid) for cid in ai}
    cases = {cid: value[0] for cid, value in case_features.items()}
    loaded = load_labels(DATA / "mechanism-sealed/reference-labels.csv", cases)
    process = {cid: [dict(row, activity_label=int(row["activity_label"]), weak_signal=int(row["weak_signal"]))
        for row in rows] for cid, rows in loaded.items()}
    agreement = []
    for cid in ai:
        for day, (ai_row, process_row) in enumerate(zip(ai[cid], process[cid], strict=True)):
            activity_category, stage_category = classify_day(ai_row, process_row, observed[cid][day])
            agreement.append({"case_id": cid, "day_index": day, "date": process_row["date"],
                "observed": bool(observed[cid][day]), "ai_activity": ai_row["activity_label"],
                "process_activity": process_row["activity_label"], "ai_feature": ai_row["feature"],
                "process_feature": process_row["feature"], "activity_comparison": activity_category,
                "stage_comparison": stage_category})
    write_csv(out / "reference-agreement.csv", agreement)
    write_json(out / "reference-agreement.json", summarize(agreement))
    original = read_json(primary / "predictions.json")
    secondary = read_json(process_predictions / "numeric-predictions.json")
    codes = read_json(process_predictions / "stage-codes.json")
    for method, records in original.items():
        if method in secondary:
            continue
        secondary[method] = {}
        for cid, value in records.items():
            feature = case_features[cid][1]
            prediction = learned_stages(np.array(codes[cid]), np.array(value["activity"]), feature.supported)
            prediction.status = value["status"]
            if prediction.status == "failed":
                prediction.stage[:] = -1
            secondary[method][cid] = prediction.to_dict()
            if secondary[method][cid]["activity"] != value["activity"]:
                raise ValueError("Target change cannot change shared visual activity")
    if set(secondary) != set(original) or len(original) != 34:
        raise ValueError("Keep all original34 methods in both targets")
    authored = read_json(DATA / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
    weak_ai, weak_process = {}, {}
    for cid in ai:
        weak_ai[cid] = np.zeros(len(ai[cid]), dtype=bool)
        for start, stop in authored.get(cid, []):
            weak_ai[cid][start:stop] = True
        weak_ai[cid] &= np.array([row["activity_label"] for row in ai[cid]]) == 1
        weak_process[cid] = np.array([row["weak_signal"] == 1 and row["activity_label"] == 1 for row in process[cid]])
    receipts = {path.name.removesuffix(".response.json"): read_json(path)
        for path in (COMPARISON / "run/requests").glob("*.response.json")}
    used = read_json(primary / "used-requests.json")
    decisions = {}
    for target, labels, weak, predictions in (
        ("primary_ai", ai, weak_ai, original), ("secondary_process", process, weak_process, secondary)):
        folder = out / target
        folder.mkdir()
        results, summaries, records = score_target(labels, observed, weak, predictions, used, receipts)
        corrections, intervals, registration = comparisons(labels, observed, predictions, results, summaries)
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "reference-labels.json", labels)
        write_json(folder / "results.json", results)
        write_json(folder / "bootstrap.json", intervals)
        write_json(folder / "registration.json", registration)
        for filename, rows in (("summary.csv", summaries), ("per-record.csv", records), ("corrections.csv", corrections)):
            write_csv(folder / filename, rows)
        decisions[target] = {"passed_comparisons": sum(row["passed"] for row in registration),
            "total_registered_comparisons": len(registration), "all_passed": all(row["passed"] for row in registration)}
        if target == "primary_ai":
            with (primary / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                archived = {row["method"]: row for row in csv.DictReader(stream)}
            for row in summaries:
                for key in row:
                    if key == "method":
                        continue
                    expected = archived[row["method"]][key]
                    if row[key] is None:
                        equal = expected == ""
                    else:
                        equal = np.isclose(row[key], float(expected))
                    if not equal:
                        raise ValueError("Original primary metric replay differs: " + key)
    write_json(out / "decision.json", decisions)
    write_json(out / "audit.json", {"status": "passed", "cases": len(ai), "methods_per_target": 34,
        "targets": 2, "primary_summary_replayed": True, "sklearn_activity_internal_stage": True,
        "matched_target_numerical_calibrations": True, "visual_activity_identical_across_targets": True,
        "sealed_labels_read_only_after_all_prediction_freezes": True, "new_remote_requests": 0,
        "independent_reviewer": False, "formal_reference_verified": False})
    write_json(out / "manifest.json", {"files": {
        path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    print(decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-process", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--process-predictions", type=Path)
    args = parser.parse_args()
    if args.action == "prepare-process":
        prepare_process(args.out.resolve())
    else:
        if args.process_predictions is None:
            parser.error("analyze requires --process-predictions")
        analyze(args.process_predictions.resolve(), args.out.resolve())
