"""Compare frozen methods on new records after the author AI reference is frozen."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from analyze_landslide_review_utility_v2 import verify_receipts
from fit_landslide_raw_persistence_baselines import raw_gate
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_adaptive_review import prepare as prepare_run
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from verify_landslide_codex_observation_review import verify as verify_reference
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
)
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity

ROOT = Path(__file__).resolve().parents[1]


def verify_bindings(directory, name):
    for filename, expected in read_json(directory / name).items():
        if sha(directory / filename) != expected:
            raise ValueError("Frozen evidence changed: " + filename)


def prepare(data, training, out):
    reference = data / "codex-review"
    verify_reference(reference, data / "raw-review-pack")
    verify_bindings(data, "data-freeze.json")
    verify_bindings(training, "training-freeze.json")
    if (reference / "delivery-manifest.json").exists():
        verify_manifest(reference, "delivery-manifest.json")
    else:
        write_json(reference / "delivery-manifest.json", {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "specific_new_predictions_seen_during_review": False,
            "specific_new_generation_labels_seen_during_review": False,
            "generic_design_and_historical_results_known": True,
            "independent_reviewer": False,
            "actual_raw_images_inspected": 24, "actual_auxiliary_images_inspected": 12,
            "compiled_images_visually_checked": ["case_0136", "case_0142", "case_0144"],
            "files": {p.relative_to(reference).as_posix(): sha(p)
                      for p in sorted(reference.rglob("*")) if p.is_file()},
        })
    if out.exists():
        raise FileExistsError("Preserve previous experiments and use a new directory")
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "source.py")
    write_json(out / "source-bindings.json", {
        "data": str(data), "training": str(training),
        "data_freeze_sha256": sha(data / "data-freeze.json"),
        "training_freeze_sha256": sha(training / "training-freeze.json"),
        "reference_delivery_sha256": sha(reference / "delivery-manifest.json"),
        "experiment_protocol_sha256": sha(data / "protocol.json"),
    })
    source = out / "source-bundle"
    source.mkdir()
    labels, _, _ = scoring_inputs(reference)
    ids = list(labels)
    write_csv(source / "reference-labels.csv", [row for cid in ids for row in labels[cid]])
    shutil.copytree(data / "inputs", source / "inputs")
    shutil.copyfile(training / "stage-xgboost.json", source / "stage-xgboost.json")
    models = {}
    for name, path in {
        "activity": training / "primary-baselines/activity-xgboost.json",
        "stage": training / "stage-xgboost.json",
        "persistence": training / "raw-baselines/activity-xgboost-raw-persistence.json",
    }.items():
        models[name] = XGBClassifier()
        models[name].load_model(path)
    features = {cid: load_case_features(source, cid)[1] for cid in ids}
    started = time.perf_counter()
    choices = read_json(training / "primary-baselines/frozen-selection.json")
    numerical = predict_baselines(features, ids, choices, models["activity"], models["stage"])
    numerical.pop("ungated_rule")
    raw_choices = read_json(training / "raw-baselines/frozen-selection.json")
    numerical.update({name: {} for name in raw_choices})
    for cid in ids:
        case, feature = load_case_features(source, cid)
        long = raw_motion_evidence(case, feature)
        matrix = np.column_stack((feature.matrix, long["rate"], long["signal_to_noise"], long["direction"]))
        rule = raw_choices["raw_multiscale_rule"]
        gates = {
            "raw_multiscale_rule": raw_gate(feature, long, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                models["persistence"].predict_proba(matrix)[:, 1]
                >= raw_choices["xgboost_raw_persistence"]["threshold"], feature.supported),
        }
        codes = models["stage"].predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            numerical[method][cid] = learned_stages(codes, gate, feature.supported)
    write_json(source / "numeric-predictions.json", {
        method: {cid: value.to_dict() for cid, value in records.items()}
        for method, records in numerical.items()})
    write_json(source / "numeric-cost.json", {
        "six_controls_total_seconds": time.perf_counter() - started,
        "scope": "Prediction of six controls, includes second feature loading and persistence derivation; excludes initial feature extraction and model fitting; no per-method timing claim",
    })
    write_json(source / "manifest.json", {
        "reference_use": "new_data_same_author_ai_reference",
        "formal_reference_verified": False,
        "files": {p.relative_to(source).as_posix(): sha(p)
                  for p in sorted(source.rglob("*")) if p.is_file()},
    })
    config = read_json(ROOT / "artifacts/landslide-adaptive-review-2026-10-09/confirmation-v1/protocol.json")
    config.update(
        date=read_json(data / "protocol.json")["date"], source_bundle=source.relative_to(ROOT).as_posix(), case_ids=ids,
        scope="New independently generated records; same-session author AI observable reference frozen before any predictions. No independent expert or field claim.",
        stage_policy="Frozen author-AI-target common stage head and six numerical controls; six training records and six separate validation records; calibration completed before these test observations were generated.",
        primary_method="adaptive_no_feedback__conflict_scope",
        revision_neighbourhood_days=49,
        freeze="Original policy,49-day scope,prompts,model,two repeats fixed before generation; report all outcomes and controls; no threshold/candidate/seed/repeat selection.",
    )
    write_json(out / "run-config.json", config)
    prepare_run(out / "run-config.json", out / "run")
    print("Prepared frozen new-record comparison; numerical predictions follow reference freeze", flush=True)


def replay(run, receipts):
    config = read_json(run / "protocol.json")
    policy = ReviewPolicy(**config["policy"])
    cached = read_json(run / "predictions.json")
    choices = read_json(run / "selections.json")
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    predictions = read_json(run / "numeric-predictions.json")
    used, scopes = {}, {}
    for cid in config["case_ids"]:
        case, feature = load_case_features(run, cid)
        codes = head.predict(feature.matrix).astype(int) + 1
        long = raw_motion_evidence(case, feature)
        for rep in range(config["repetitions"]):
            direct = f"direct_visual_r{rep}"
            initial = np.array(cached[direct][cid]["activity"])
            predictions.setdefault(direct, {})[cid] = cached[direct][cid]
            used.setdefault(direct, {})[cid] = [f"{cid}-direct-r{rep}"]
            choice = choices[f"{cid}-r{rep}"]
            windows = [(row["start"], row["stop"]) for row in choice["priorities"]]
            priorities, selected = review_priorities(initial, feature, long, windows, policy)
            if priorities != choice["priorities"] or selected != choice["selected"]:
                raise ValueError("Original adaptive schedule does not replay")
            fixed = [priorities[index] for index in choice["fixed_indices"]]
            count = len(selected)
            targets_by_arm = {
                "fixed_image_review": fixed,
                "adaptive_no_feedback": selected,
                "adaptive_numeric_feedback": selected,
                "same_count_fixed_early": fixed[:count],
                "same_count_fixed_late": fixed[len(fixed) - count:] if count else [],
            }
            neighbourhood = conflict_neighbourhood(initial, feature, long, policy)[1]
            for stem, targets in targets_by_arm.items():
                arm = "numeric" if stem == "adaptive_numeric_feedback" else "image"
                keys = [f"{cid}-direct-r{rep}"]
                local = []
                for target in targets:
                    key = f"{cid}-window-{target['index']:02d}-{arm}-r{rep}"
                    receipt = receipts[key]
                    try:
                        activity = parse_window_activity(json.loads(receipt["output"]), feature.observed,
                                                         target["start"], target["stop"])
                        status = "success"
                    except (ValueError, KeyError):
                        activity = np.full(len(initial), -1)
                        status = "failed"
                    local.append({"start": target["start"], "stop": target["stop"],
                                  "activity": activity.tolist(), "status": status})
                    keys.append(key)
                reviewed = combine_reviews(initial, local, feature.observed)
                if stem in {"fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"}:
                    if reviewed.activity.tolist() != cached[f"{stem}_r{rep}"][cid]["activity"]:
                        raise ValueError("Archived local replies do not reconstruct")
                authority, center, details = target_scopes(neighbourhood, feature.observed, targets)
                variants = {"whole_target": None, "conflict_scope": authority}
                if stem in {"fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"}:
                    variants["matched_center_scope"] = center
                for variant, scope in variants.items():
                    activity = reviewed.activity.copy()
                    if scope is not None:
                        activity = scoped_activity(initial, activity, scope, feature.observed)
                    method = f"{stem}_r{rep}__{variant}"
                    prediction = learned_stages(codes, activity, feature.supported)
                    predictions.setdefault(method, {})[cid] = prediction.to_dict()
                    used.setdefault(method, {})[cid] = keys
                    scopes.setdefault(method, {})[cid] = {
                        "targets": details, "editable_days": np.flatnonzero(scope).tolist()
                        if scope is not None else [day for target in targets
                                                  for day in range(target["start"], target["stop"])
                                                  if feature.observed[day]],
                    }
    return predictions, used, scopes


def analyze(out):
    bindings = read_json(out / "source-bindings.json")
    data = Path(bindings["data"])
    training = Path(bindings["training"])
    reference = data / "codex-review"
    verify_bindings(training, "training-freeze.json")
    verify_bindings(data, "data-freeze.json")
    verify_manifest(reference, "delivery-manifest.json")
    verify_manifest(out / "source-bundle")
    run = out / "run"
    verify_freeze(run)
    verify_bindings(run, "prediction-freeze.json")
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Analyze only the complete frozen inference run")
    target = out / "analysis"
    target.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), target / "source.py")
    receipts = verify_receipts(run)
    predictions, used, scopes = replay(run, receipts)
    write_json(target / "predictions.json", predictions)
    write_json(target / "used-requests.json", used)
    write_json(target / "scopes.json", scopes)
    write_json(target / "prediction-freeze.json", {
        name: sha(target / name) for name in ("predictions.json", "used-requests.json", "scopes.json")})
    labels, observed, _ = scoring_inputs(reference)
    weak_spans = read_json(reference / "activity-decisions.json")["weak_interiors_authored_before_predictions"]
    weak = {}
    for cid, rows in labels.items():
        weak[cid] = np.zeros(len(rows), dtype=bool)
        for start, stop in weak_spans.get(cid, []):
            weak[cid][start:stop] = True
        weak[cid] &= np.array([row["activity_label"] for row in rows]) == 1
    results, summaries, daily, per_record = {}, [], [], []
    for method, cases in predictions.items():
        per_case, boundaries, boundaries05 = {}, {}, {}
        actual_all, predicted_all, actual_internal, predicted_internal = [], [], [], []
        confusion = np.zeros((3, 5), dtype=int)
        weak_total = weak_detected = unknown_positive = unknown_negative = unknown_abstained = 0
        for cid, value in cases.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            if np.any(prediction.activity[~observed[cid]] != -1) or np.any(prediction.stage[~observed[cid]] != -1):
                raise ValueError("Missing day gained a prediction")
            rows = labels[cid]
            per_case[cid] = evaluate_case(rows, prediction)
            boundaries[cid] = boundary_errors(rows, prediction.activity, .3)
            boundaries05[cid] = boundary_errors(rows, prediction.activity, .5)
            actual = np.array([row["activity_label"] for row in rows])
            known = actual >= 0
            actual_all.extend(actual[known].tolist())
            predicted_all.extend(prediction.activity[known].tolist())
            stage = stage_reference_arrays(rows, prediction)
            interior = stage["stage_interior"]
            actual_internal.extend(stage["actual_stage"][interior].tolist())
            predicted_internal.extend(prediction.stage[interior].tolist())
            confusion += np.array(stage["confusion_rows_ASD_columns_unknown_none_ASD"])
            found = int((weak[cid] & (prediction.activity == 1)).sum())
            weak_detected += found
            weak_total += int(weak[cid].sum())
            unknown = (actual == -1) & observed[cid]
            unknown_positive += int((unknown & (prediction.activity == 1)).sum())
            unknown_negative += int((unknown & (prediction.activity == 0)).sum())
            unknown_abstained += int((unknown & (prediction.activity == -1)).sum())
            per_record.append({"method": method, "case_id": cid, **per_case[cid]["daily"],
                               "weak_days": int(weak[cid].sum()), "weak_detected_days": found})
            daily.extend({"method": method, **row, "predicted_activity": int(prediction.activity[day]),
                          "predicted_stage": int(prediction.stage[day]), "observed": bool(observed[cid][day])}
                         for day, row in enumerate(rows))
        aggregate = aggregate_scores(list(per_case.values()))
        p, r, f1, _ = precision_recall_fscore_support(actual_all, predicted_all, labels=[1], zero_division=0)
        if not all(np.isclose(aggregate["daily"][name], value)
                   for name, value in (("precision", p[0]), ("recall", r[0]), ("f1", f1[0]))):
            raise ValueError("Independent activity counter disagrees")
        ip, ir, internal_f1, support = precision_recall_fscore_support(
            actual_internal, predicted_internal, labels=[1, 2, 3], zero_division=0)
        pairs = [pair for value in boundaries.values() for pair in value["matched_pairs"]]
        scores05 = aggregate_scores([evaluate_case(labels[cid], ActivityPrediction(
            np.array(value["activity"]), np.array(value["stage"])), .5) for cid, value in cases.items()])
        if len(pairs) != aggregate["events"]["tp"] or sum(len(value["matched_pairs"])
                for value in boundaries05.values()) != scores05["events"]["tp"]:
            raise ValueError("Event and boundary match counts disagree")
        internal = {"precision_ASD": ip.tolist(), "recall_ASD": ir.tolist(),
                    "f1_ASD": internal_f1.tolist(), "support_ASD": support.tolist(),
                    "macro_f1": float(internal_f1.mean()),
                    "coverage": float(np.mean(np.array(predicted_internal) > 0))}
        results[method] = {"aggregate": aggregate, "per_case": per_case,
                           "events_iou05": scores05["events"], "internal_stages": internal,
                           "boundaries_iou03": boundaries, "boundaries_iou05": boundaries05,
                           "confusion_rows_ASD_columns_unknown_none_ASD": confusion.tolist()}
        keys = [key for values in used.get(method, {}).values() for key in values]
        summaries.append({"method": method, **aggregate["daily"],
            "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": scores05["events"]["f1"],
            "onset_core_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs])) if pairs else None,
            "offset_core_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs])) if pairs else None,
            "matched_events": len(pairs), "stage_end_to_end_macro_f1": aggregate["stage_macro_f1"],
            "stage_internal_macro_f1": internal["macro_f1"], "stage_internal_coverage": internal["coverage"],
            "coverage": aggregate["coverage"], "weak_days": weak_total,
            "weak_detected_days": weak_detected, "weak_recall": weak_detected / weak_total,
            "unknown_reference_predicted_positive": unknown_positive,
            "unknown_reference_predicted_negative": unknown_negative,
            "unknown_reference_abstained": unknown_abstained,
            "requests": len(keys), "reported_tokens": sum(receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys),
            "receipt_seconds": sum(receipts[key]["seconds"] for key in keys)})
    corrections, bootstrap = [], []
    for rep in (0, 1):
        main = f"adaptive_no_feedback_r{rep}__conflict_scope"
        controls = [f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
                    f"same_count_fixed_early_r{rep}__conflict_scope",
                    f"same_count_fixed_late_r{rep}__conflict_scope",
                    f"adaptive_no_feedback_r{rep}__whole_target",
                    f"adaptive_no_feedback_r{rep}__matched_center_scope",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope",
                    "xgboost_raw_persistence", "raw_multiscale_rule"]
        for control in controls:
            counts = [correction_utility(np.array([row["activity_label"] for row in labels[cid]]),
                        np.array(predictions[control][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]), observed[cid]) for cid in labels]
            combined = {key: sum(row[key] for row in counts)
                        for key in ("eligible_days", "gained_correct", "lost_correct", "net_correct")}
            combined["utility_fraction"] = combined["net_correct"] / combined["eligible_days"]
            corrections.append({"control": control, "treatment": main, **combined})
            bootstrap.append({"control": control, "treatment": main,
                              **paired_record_interval(results, control, main)})
    write_json(target / "results.json", results)
    write_json(target / "bootstrap.json", bootstrap)
    for name, rows in (("summary.csv", summaries), ("per-record.csv", per_record),
                       ("daily-comparison.csv", daily), ("corrections.csv", corrections)):
        write_csv(target / name, rows)
    selected = [row for row in summaries if row["method"] in {
        "direct_visual_r0", "direct_visual_r1", "fixed_image_review_r0__whole_target",
        "fixed_image_review_r1__whole_target", "adaptive_no_feedback_r0__conflict_scope",
        "adaptive_no_feedback_r1__conflict_scope", "raw_multiscale_rule", "xgboost_raw_persistence"}]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, metric, title in zip(axes, ("f1", "weak_recall", "fp"),
                                   ("Activity F1", "Frozen weak-interior recall", "Stationary FP"), strict=True):
        axis.barh(range(len(selected)), [row[metric] for row in selected], color="#327b88")
        axis.set_yticks(range(len(selected)), [row["method"] for row in selected], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    figure.suptitle("New controlled records; author AI reference frozen before predictions")
    figure.tight_layout()
    figure.savefig(target / "comparison.png", dpi=150)
    plt.close(figure)
    write_json(target / "audit.json", {
        "status": "passed", "cases": len(labels), "methods": len(predictions), "daily_rows": len(daily),
        "known_activity_days": len(actual_all), "stage_support_ASD": support.tolist(),
        "weak_subset_days": weak_total, "verified_request_payloads": len(receipts),
        "requests_started": len(list((run / "requests").glob("*.started.json"))),
        "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
        "shared_activity_stage_missing_contracts": True, "sklearn_activity_internal_stage": True,
        "schedule_response_replay": True, "event_boundary_agreement": True,
        "new_data_independent_rng": True, "independent_reviewer": False,
        "formal_reference_verified": False, "test_selection": False,
    })
    write_json(target / "manifest.json", {"files": {
        p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob("*")) if p.is_file()}})
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.data is None or args.training is None:
            parser.error("prepare requires --data and --training")
        prepare(args.data.resolve(), args.training.resolve(), args.out.resolve())
    else:
        analyze(args.out.resolve())
