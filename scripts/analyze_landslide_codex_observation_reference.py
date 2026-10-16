"""Evaluate unchanged same-record predictions against the frozen Codex development reference."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from sklearn.metrics import precision_recall_fscore_support
from verify_landslide_codex_observation_review import verify as verify_reference

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_review_utility import correction_utility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
STAGE_CODES = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}
WEAK_INTERIORS = {"case_0051": [(220, 390)], "case_0054": [(470, 650)],
                  "case_0057": [(225, 420)], "case_0060": [(480, 640)]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_hashes(directory, manifest_name, key=None):
    bindings = read_json(directory / manifest_name)
    if key is not None:
        bindings = bindings[key]
    for name, digest in bindings.items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen source changed: " + str(directory / name))


def scoring_inputs(reference):
    daily = read_json(reference / "compiled/daily-reference.json")
    reviews = read_json(reference / "compiled/activity-reviews.json")
    labels, observations, weak = {}, {}, {}
    for review in reviews:
        case_id = review["case_id"]
        case = LandslideInput.model_validate_json((reference / case_id / "input.json").read_bytes())
        rows = [dict(row, episode_id="") for row in daily if row["case_id"] == case_id]
        episode = 0
        for span in review["spans"]:
            if span["state"] != "activity":
                continue
            episode += 1
            for day in range(span["start"], span["stop"]):
                rows[day]["episode_id"] = f"{case_id}-interior-{episode:02d}"
        labels[case_id] = rows
        observations[case_id] = np.array([value is not None for value in case.displacement_mm])
        weak_mask = np.zeros(len(rows), dtype=bool)
        for start, stop in WEAK_INTERIORS.get(case_id, []):
            if any(rows[day]["activity_label"] == 0 for day in range(start, stop)):
                raise ValueError("Descriptive weak subset extends beyond confirmed activity")
            weak_mask[start:stop] = True
        weak[case_id] = weak_mask & (np.array([row["activity_label"] for row in rows]) == 1)
    return labels, observations, weak


def stage_reference_arrays(labels, prediction):
    actual = np.array([STAGE_CODES.get(row["feature"], -1) for row in labels])
    activity = np.array([row["activity_label"] for row in labels])
    actual[activity == 0] = 0
    known = actual >= 0
    interior = actual > 0
    confusion = [[int(np.count_nonzero((actual == expected) & (prediction.stage == code)))
                  for code in (-1, 0, 1, 2, 3)] for expected in (1, 2, 3)]
    return {"confusion_rows_ASD_columns_unknown_none_ASD": confusion,
            "actual_stage": actual, "stage_known": known, "stage_interior": interior}


def comparisons(predictions, labels, observed):
    records = []
    pairs = []
    for repeat in (0, 1):
        main = f"adaptive_no_feedback_r{repeat}__conflict_scope"
        controls = (f"direct_visual_r{repeat}", f"fixed_image_review_r{repeat}__whole_target",
                    f"adaptive_no_feedback_r{repeat}__whole_target",
                    f"adaptive_no_feedback_r{repeat}__matched_center_scope",
                    f"adaptive_numeric_feedback_r{repeat}__conflict_scope",
                    "xgboost_raw_persistence", "raw_multiscale_rule")
        pairs.extend((control, main) for control in controls)
    for control, treatment in pairs:
        for case_id, rows in labels.items():
            actual = np.array([row["activity_label"] for row in rows])
            before = np.array(predictions[control][case_id]["activity"])
            after = np.array(predictions[treatment][case_id]["activity"])
            gained = (before != 1) & (after == 1)
            removed = (before == 1) & (after != 1)
            records.append({"control": control, "treatment": treatment, "case_id": case_id,
                            **correction_utility(actual, before, after, observed[case_id]),
                            "gained_tp": int(np.count_nonzero(gained & (actual == 1))),
                            "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                            "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                            "gained_fp": int(np.count_nonzero(gained & (actual == 0)))})
    aggregates = []
    for control, treatment in pairs:
        selected = [row for row in records if row["control"] == control and row["treatment"] == treatment]
        aggregate = {"control": control, "treatment": treatment}
        for key in ("eligible_days", "gained_correct", "lost_correct", "net_correct",
                    "gained_tp", "lost_tp", "removed_fp", "gained_fp"):
            aggregate[key] = sum(row[key] for row in selected)
        aggregates.append(aggregate)
    return records, aggregates


def analyze(out):
    if out.exists():
        raise FileExistsError("Use a new analysis directory; keep prior results frozen")
    reference = BASE / "codex-observation-review-v1"
    run = BASE / "confirmation-v1"
    scope = BASE / "revision-scope-probe-v1"
    native = BASE / "confirmation-native-stages-v1"
    verify_reference(reference, BASE / "independent-review-pack-v1")
    verify_hashes(reference, "delivery-manifest.json", "files")
    verify_hashes(scope, "manifest.json", "files")
    verify_hashes(run, "prediction-freeze.json")
    verify_hashes(run, "freeze.json")
    if sha(native / "predictions.json") != read_json(native / "prediction-freeze.json")["sha256"]:
        raise ValueError("Native stage predictions changed")
    predictions = read_json(scope / "confirmation-v1/predictions.json")
    native_predictions = read_json(native / "predictions.json")
    if set(predictions) & set(native_predictions):
        raise ValueError("Native supplement replaces an existing method")
    predictions.update(native_predictions)
    labels, observed, weak = scoring_inputs(reference)
    case_ids = list(labels)
    if len(predictions) != 30 or any(set(cases) != set(case_ids) for cases in predictions.values()):
        raise ValueError("Require all 30 same-record arms, including six strong numerical controls")
    for case_id in case_ids:
        if sha(reference / case_id / "input.json") != sha(run / "inputs" / case_id / "input.json"):
            raise ValueError("Reference and methods use different raw observations")
    if read_json(native / "protocol.json")["activity_protocol_sha256"] != sha(run / "protocol.json"):
        raise ValueError("Native stages use another activity experiment")
    original_predictions = read_json(run / "predictions.json")
    for method, cases in native_predictions.items():
        activity_method = method.replace("_native_stage", "")
        for case_id, prediction in cases.items():
            if prediction["activity"] != original_predictions[activity_method][case_id]["activity"]:
                raise ValueError("Native supplement changed its original activity gate")

    costs = read_json(run / "costs.json")
    costs.update(read_json(native / "costs.json"))
    source_paths = [reference / "delivery-manifest.json", reference / "compiled/daily-reference.json",
                    scope / "manifest.json", scope / "confirmation-v1/predictions.json",
                    native / "predictions.json", run / "costs.json", native / "costs.json",
                    run / "protocol.json", run / "stage-xgboost.json"]
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "source.py")
    write_json(out / "pre-score-freeze.json", {
        "scope": "Posthoc development/reference sensitivity, not independent or field validation",
        "methods": list(predictions), "case_ids": case_ids,
        "reference": "Frozen Codex AI-assisted raw-first review; prior method results exposed",
        "weak_subset_half_open": WEAK_INTERIORS,
        "unknown_reference": "Exclude reference unknown from known-day metrics; separately report predictions on all observed unknown days",
        "unknown_prediction": "Retain as missed positives; no true-negative credit",
        "stages": "591 definite ASD observation days; 5223 confirmed stationary days also enter end-to-end stage FP denominators",
        "events": "Existing one-to-one masked IoU .3/.5, reference interior IDs; boundary errors relative to conservative cores, not physical endpoints",
        "training": "Unchanged common train24/validation12 models and tuning; no retraining or calibration on these 12 evaluated AI references",
        "limitations": "References share numerical evidence with numerical methods; boundary/motion evidence is non-independent; semantic sensitivity only",
        "new_model_requests": 0, "models_changed": False, "original_scores_changed": False,
        "source_bindings": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "scorer_sha256": sha(ROOT / "src/gnss_sim/landslide_evaluation.py"),
        "boundary_scorer_sha256": sha(ROOT / "scripts/analyze_landslide_adaptive_review.py"),
    })
    results, summary, per_record, day_rows = {}, [], [], []
    for method, cases in predictions.items():
        per_case, boundaries, boundaries05 = {}, {}, {}
        actual_all, predicted_all, actual_internal, predicted_internal = [], [], [], []
        stage_actual_all, stage_predicted_all = [], []
        unknown_counts = Counter()
        weak_detected, weak_total = 0, 0
        confusion = np.zeros((3, 5), dtype=int)
        for case_id in case_ids:
            serialized = cases[case_id]
            prediction = ActivityPrediction(
                np.array(serialized["activity"]), np.array(serialized["stage"]),
                status=serialized["status"], gated=serialized["gated"], stage_status=serialized["stage_status"])
            if np.any(prediction.activity[~observed[case_id]] != -1) or np.any(prediction.stage[~observed[case_id]] != -1):
                raise ValueError("Prediction invented a missing-day label")
            rows = labels[case_id]
            per_case[case_id] = evaluate_case(rows, prediction)
            boundaries[case_id] = boundary_errors(rows, prediction.activity, .3)
            boundaries05[case_id] = boundary_errors(rows, prediction.activity, .5)
            actual = np.array([row["activity_label"] for row in rows])
            known = actual >= 0
            unknown = observed[case_id] & ~known
            counts = {f"unknown_reference_predicted_{code}": int(np.count_nonzero(unknown & (prediction.activity == code)))
                      for code in (-1, 0, 1)}
            unknown_counts.update(counts)
            actual_all.extend(actual[known].tolist())
            predicted_all.extend((prediction.activity[known] == 1).astype(int).tolist())
            stage = stage_reference_arrays(rows, prediction)
            confusion += np.array(stage["confusion_rows_ASD_columns_unknown_none_ASD"])
            stage_actual_all.extend(stage["actual_stage"][stage["stage_known"]].tolist())
            stage_predicted_all.extend(prediction.stage[stage["stage_known"]].tolist())
            actual_internal.extend(stage["actual_stage"][stage["stage_interior"]].tolist())
            predicted_internal.extend(prediction.stage[stage["stage_interior"]].tolist())
            detected = int(np.count_nonzero(weak[case_id] & (prediction.activity == 1)))
            weak_detected += detected
            weak_total += int(weak[case_id].sum())
            per_record.append({"method": method, "case_id": case_id, **per_case[case_id]["daily"],
                               "weak_days": int(weak[case_id].sum()), "weak_detected": detected, **counts})
            for day, row in enumerate(rows):
                day_rows.append({"method": method, **row, "activity_prediction": int(prediction.activity[day]),
                                 "stage_prediction": int(prediction.stage[day]),
                                 "observed": bool(observed[case_id][day]), "weak_subset": bool(weak[case_id][day])})
        aggregate = aggregate_scores(list(per_case.values()))
        p, r, f1, _ = precision_recall_fscore_support(actual_all, predicted_all, labels=[1], zero_division=0)
        for key, value in (("precision", p[0]), ("recall", r[0]), ("f1", f1[0])):
            if not np.isclose(aggregate["daily"][key], value):
                raise ValueError("Independent sklearn activity verification failed")
        _, _, end_f1, _ = precision_recall_fscore_support(
            stage_actual_all, stage_predicted_all, labels=[1, 2, 3], zero_division=0)
        ip, ir, inf1, support = precision_recall_fscore_support(
            actual_internal, predicted_internal, labels=[1, 2, 3], zero_division=0)
        if not np.isclose(aggregate["stage_macro_f1"], np.mean(end_f1)):
            raise ValueError("Independent sklearn stage verification failed")
        pairs = [pair for result in boundaries.values() for pair in result["matched_pairs"]]
        if len(pairs) != aggregate["events"]["tp"]:
            raise ValueError("Boundary and event match counts disagree")
        iou05 = aggregate_scores([evaluate_case(labels[cid], ActivityPrediction(
            np.array(cases[cid]["activity"]), np.array(cases[cid]["stage"]),
            status=cases[cid]["status"], gated=cases[cid]["gated"], stage_status=cases[cid]["stage_status"]), .5)
            for cid in case_ids])["events"]
        if sum(len(result["matched_pairs"]) for result in boundaries05.values()) != iou05["tp"]:
            raise ValueError("IoU .5 boundary match counts disagree")
        internal = {"precision_ASD": ip.tolist(), "recall_ASD": ir.tolist(), "f1_ASD": inf1.tolist(),
                    "support_ASD": support.tolist(), "macro_f1": float(np.mean(inf1)),
                    "definite_stage_coverage": float(np.mean(np.array(predicted_internal) > 0))}
        results[method] = {"aggregate": aggregate, "per_case": per_case,
                           "events_iou05": iou05, "internal_stages": internal,
                           "confusion_rows_ASD_columns_unknown_none_ASD": confusion.tolist(),
                           "boundaries_iou03": boundaries, "boundaries_iou05": boundaries05}
        cost = costs.get(method.split("__")[0])
        summary.append({"method": method, **aggregate["daily"],
                        "events_iou03_f1": aggregate["events"]["f1"], "events_iou05_f1": iou05["f1"],
                        "onset_core_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs])) if pairs else None,
                        "offset_core_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs])) if pairs else None,
                        "matched_events_iou03": len(pairs), "stage_end_to_end_macro_f1": aggregate["stage_macro_f1"],
                        "stage_internal_macro_f1": internal["macro_f1"], "stage_internal_coverage": internal["definite_stage_coverage"],
                        "activity_coverage": aggregate["coverage"], "weak_days": weak_total,
                        "weak_detected_days": weak_detected, "weak_recall": weak_detected / weak_total,
                        **dict(unknown_counts), "archived_requests": cost["requests"] if cost else 0,
                        "archived_reported_tokens": cost["tokens"] if cost else 0,
                        "compute_cost_scope": "Archived shared deployment receipts" if cost else "No VLM; CPU/training cost not measured here"})
    records, corrections = comparisons(predictions, labels, observed)
    write_json(out / "results.json", results)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-record.csv", per_record)
    write_csv(out / "daily-comparison.csv", day_rows)
    write_csv(out / "corrections.csv", corrections)
    write_csv(out / "per-record-corrections.csv", records)
    write_json(out / "validation-audit.json", {
        "status": "passed", "cases": 12, "methods": 30, "daily_prediction_rows": len(day_rows),
        "known_activity_days": len(actual_all), "observed_unknown_reference_days": sum(unknown_counts.values()),
        "stage_internal_support_ASD": support.tolist(), "weak_subset_days": weak_total,
        "original_reference_and_predictions_unchanged": True, "sklearn_activity_and_stage_verified": True,
        "event_boundary_matches_agree": True, "new_remote_model_requests": 0,
        "formal_reference_verified": False, "reference_use": "development_only"})
    selected = [row for row in summary if row["method"] in (
        "direct_visual_r0", "direct_visual_r1", "fixed_image_review_r0__whole_target",
        "fixed_image_review_r1__whole_target", "adaptive_no_feedback_r0__conflict_scope",
        "adaptive_no_feedback_r1__conflict_scope", "xgboost_raw_persistence", "raw_multiscale_rule")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    positions = np.arange(len(selected))
    for panel, metric, title in zip(axes, ("f1", "weak_recall", "fp"),
                                    ("Activity F1", "Reviewed weak subset recall", "Confirmed-stationary FP"), strict=True):
        panel.barh(positions, [row[metric] for row in selected], color="#327b88")
        panel.set_yticks(positions, [row["method"] for row in selected], fontsize=8)
        panel.set_title(title)
        panel.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    figure.suptitle("Frozen Codex AI-assisted development reference; NOT independent validation")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=150)
    plt.close(figure)
    write_json(out / "manifest.json", {"reference_use": "development_only", "formal_reference_verified": False,
        "files": {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out.resolve())
