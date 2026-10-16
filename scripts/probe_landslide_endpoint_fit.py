"""Freeze raw endpoint repairs before scoring against the two existing targets."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from diagnose_landslide_feedback_contribution import compare_cases
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_endpoint_fit import apply_endpoint_fits, fit_activity_endpoints
from gnss_sim.landslide_evaluation import ActivityPrediction, event_masks
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
PRIOR = BASE / "displacement-comparison-analysis-v2"
TARGETS = ("primary_process", "secondary_developer")
RULES = ("selective", "no_step_rival", "unconditional_fit")


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def export(path, value):
    write_json(path, native(value))


def method_name(base, rule):
    return base + "__endpoint_" + rule


def anchors(predictions):
    visual = [
        name
        for rep in (0, 1)
        for name in (
            f"direct_visual_r{rep}",
            f"adaptive_no_feedback_r{rep}__conflict_scope",
            f"fixed_image_review_r{rep}__whole_target",
            f"same_count_fixed_early_r{rep}__conflict_scope",
        )
    ]
    numerical = [name for name in predictions if not any(f"_r{rep}" in name for rep in (0, 1))]
    return visual + numerical


def event_extents(activity):
    return [
        (int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask)[-1]) + 1)
        for mask in event_masks(np.asarray(activity))
    ]


def validate_repair(original, repaired, observed):
    before = np.array(original["activity"])
    after = np.array(repaired["activity"])
    stages = np.array(repaired["stage"])
    if not np.array_equal(after[before == -1], before[before == -1]):
        raise ValueError("Endpoint repair filled original unknown dates")
    if np.any(after[~observed] != -1) or np.any(stages[~observed] != -1):
        raise ValueError("Endpoint repair filled raw missing dates")
    ActivityPrediction(
        after, stages, repaired["status"], repaired["gated"], repaired["stage_status"]
    ).validate(len(after))
    old_events, new_events = event_masks(before), event_masks(after)
    if len(old_events) != len(new_events) or any(
        not np.any(left & right & (before == 1) & (after == 1))
        for left, right in zip(old_events, new_events, strict=True)
    ):
        raise ValueError("Endpoint repair changed event identity or topology")


def prepare(out):
    verify_manifest(PRIOR)
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_endpoint_fit.py",
        ROOT / "src/gnss_sim/landslide_motion_competition.py",
        ROOT / "src/gnss_sim/landslide_displacement_comparison_view.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
        ROOT / "tests/test_landslide_endpoint_fit.py",
        ROOT / "scripts/score_landslide_core_lockbox.py",
        ROOT / "scripts/analyze_landslide_adaptive_review.py",
        ROOT / "scripts/diagnose_landslide_feedback_contribution.py",
    ]
    heads = {
        "primary_process": DATA / "stage-xgboost.json",
        "secondary_developer": BASE / "ai-observable-calibration-v3/stage-xgboost.json",
    }
    originals = {target: read_json(PRIOR / target / "predictions.json") for target in TARGETS}
    plan = {target: anchors(predictions) for target, predictions in originals.items()}
    export(
        out / "protocol.json",
        {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Complete exposed24 process/fixed12 author development; no independent or field confirmation",
            "change": "Local raw stationary-to-linear/linear-to-stationary endpoint fit; no event discovery or deletion",
            "anchors": plan,
            "rules": RULES,
            "evidence": "Endpoint +/-98 original days,14-day medians >=8 completeNEU; >=6 blocks; knots +/-49 integer days, >=2 blocks each shoulder, event >=28 days",
            "selection": "Each alternating-block heldout fold must beat stationary,linear drift,single step; full/two fold knot span <=14 days;1e-12 numerical ties rejected",
            "calendar": "All original unknown and raw gaps retained; never cross >7-day abstention; fixed known stationary separator between adjacent events;record-censored endpoints retained",
            "stages": "Keep original stage where activity unchanged; regate changed dates with the same target's frozen XGBoost head and original support",
            "calibration": "No fitting to reference, training, validation selection or threshold search; all original target-matched baseline models retained",
            "primary": "Direct visual r0/r1 selective endpoint repair",
            "gate": {
                "process": "Mean repeat F1 difference >0;total net correct>0;mean weakRecall nondecrease;>=2 benefiting records each repeat;pooled same-reference/same-prediction-identity common-match endpoint MAE strictly decreases with at least one common match per repeat",
                "ablation": "Direct selective mean F1 strictly better than unconditional_fit;no_step_rival effects fully reported, not a mandatory win",
                "author": "Mean repeat F1 nondecrease;total net correct>0;mean weakRecall nondecrease;author endpoints are conservative interiors, not precise physical endpoints",
            },
            "failure": "Stop this endpoint candidate if any registered gate fails;no parameter/rival/fold/prompt/seed/repeat/subset tuning or replacement of the direct anchor",
            "next_if_pass": "Freeze unseen-record confirmation and compare same repair on fixed/adaptive/numerical anchors; development alone is not a contribution claim",
            "cost": "Zero new API;deployment request/token inherited from each anchor;actual endpoint CPU recorded separately, not forced equal in replay",
            "old_decisions": ["NOT_CONFIRMED", "STOP_FEEDBACK_PROMOTION", "STOP_AXIS_WITNESS_BEFORE_CALLS", "STOP_COMPARISON_DISPLAY"],
        },
    )
    source_folder = out / "source"
    source_folder.mkdir()
    for path in sources:
        shutil.copyfile(path, source_folder / path.name)
    bound = sources + list(heads.values())
    bound.extend(path for path in PRIOR.rglob("*") if path.is_file())
    case_ids = sorted(set().union(*(set(p["direct_visual_r0"]) for p in originals.values())))
    bound.extend(path for cid in case_ids for path in (DATA / "inputs" / cid).iterdir() if path.is_file())
    # Bind reference bytes without loading their labels into the prediction path.
    bound.extend(BASE / "core-lockbox-analysis-v1" / target / "reference-labels.json" for target in TARGETS)
    bound.append(DATA / "codex-review/activity-decisions.json")
    export(out / "source-bindings.json", {str(path): sha(path) for path in bound})
    pairs = {cid: load_case_features(DATA, cid) for cid in case_ids}
    timings, checked_days = [], 0
    for target in TARGETS:
        head = XGBClassifier()
        head.load_model(heads[target])
        codes = {cid: head.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in originals[target]["direct_visual_r0"]}
        additions, evidence = {}, {}
        fit_cache = {}
        for base in plan[target]:
            for cid, original in originals[target][base].items():
                case, feature = pairs[cid]
                activity = np.array(original["activity"])
                key = (cid, tuple(activity))
                started = time.perf_counter()
                reused = key in fit_cache
                if not reused:
                    fit_cache[key] = fit_activity_endpoints(case, feature.noise, activity)
                fits = fit_cache[key]
                fit_seconds = time.perf_counter() - started
                evidence.setdefault(base, {})[cid] = {"fits": fits, "rules": {}}
                for rule in RULES:
                    started = time.perf_counter()
                    changed, diagnostic = apply_endpoint_fits(activity, feature.observed, fits, rule)
                    application_seconds = time.perf_counter() - started
                    value = learned_stages(codes[cid], changed, feature.supported).to_dict()
                    # This is an activity intervention, not a stage-head swap.
                    unchanged = changed == activity
                    stage = np.array(value["stage"])
                    stage[unchanged] = np.array(original["stage"])[unchanged]
                    value["stage"] = stage.tolist()
                    for field in ("status", "stage_status", "gated"):
                        value[field] = original[field]
                    validate_repair(original, value, feature.observed)
                    additions.setdefault(method_name(base, rule), {})[cid] = value
                    evidence[base][cid]["rules"][rule] = diagnostic
                    checked_days += len(changed)
                    timings.append({"target": target, "base": base, "case_id": cid, "rule": rule,
                                    "fit_reused": reused, "fit_seconds": fit_seconds if rule == RULES[0] else 0,
                                    "application_seconds": application_seconds})
        folder = out / target
        export(folder / "predictions.json", additions)
        export(folder / "endpoint-evidence.json", evidence)
    write_csv(out / "cpu-timings.csv", timings)
    export(out / "contract-checks.json", {"checked_days": checked_days, "unknown_missing_stages_and_event_identity": "passed", "new_api_calls": 0})
    verify_bindings(out, "source-bindings.json")
    export(out / "prediction-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print("Frozen endpoint predictions:", checked_days, "validated days; zero new API", flush=True)


def verify(out):
    verify_bindings(out, "source-bindings.json")
    verify_bindings(out, "prediction-freeze.json")


def common_boundaries(results, predictions, labels, control, treatment):
    """Compare exactly the same reference event and surviving prediction identity."""
    records = []
    for cid in labels:
        lookups = {}
        for method in (control, treatment):
            extents = event_extents(predictions[method][cid]["activity"])
            identities = {extent: index for index, extent in enumerate(extents)}
            lookups[method] = {
                (pair["episode_id"], identities[(pair["predicted_start"], pair["predicted_end"] + 1)]): pair
                for pair in results[method]["boundaries_iou03"][cid]["matched_pairs"]
            }
        for identity in sorted(lookups[control].keys() & lookups[treatment].keys()):
            before, after = lookups[control][identity], lookups[treatment][identity]
            records.append({"case_id": cid, "episode_id": identity[0], "event_index": identity[1],
                            "before_onset_abs": abs(before["onset_error_days"]),
                            "after_onset_abs": abs(after["onset_error_days"]),
                            "before_offset_abs": abs(before["offset_error_days"]),
                            "after_offset_abs": abs(after["offset_error_days"])})
    before = [row[key] for row in records for key in ("before_onset_abs", "before_offset_abs")]
    after = [row[key] for row in records for key in ("after_onset_abs", "after_offset_abs")]
    return {"common_matches": len(records),
            "common_endpoint_mae_before": float(np.mean(before)) if before else None,
            "common_endpoint_mae_after": float(np.mean(after)) if after else None,
            "common_endpoint_mae_difference": float(np.mean(after) - np.mean(before)) if before else None}, records


def score(out, destination):
    verify(out)
    destination.mkdir(parents=True, exist_ok=False)
    comparisons, boundary_rows, corrections = [], [], []
    for target in TARGETS:
        previous = read_json(PRIOR / target / "predictions.json")
        additions = read_json(out / target / "predictions.json")
        labels = read_json(BASE / "core-lockbox-analysis-v1" / target / "reference-labels.json")
        pairs = {cid: load_case_features(DATA, cid) for cid in labels}
        if target == "primary_process":
            weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1 for row in rows]) for cid, rows in labels.items()}
        else:
            spans = read_json(DATA / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
            weak = {}
            for cid, rows in labels.items():
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in spans.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([int(row["activity_label"]) for row in rows]) == 1)
        new_results, new_summaries, new_records = score_target(labels, {cid: pair[1].observed for cid, pair in pairs.items()}, weak, additions, {}, {})
        old_summaries = csv_rows(PRIOR / target / "summary.csv")
        old_table = {row["method"]: row for row in old_summaries}
        # CPU differs by hardware/run. API deployment cost is exactly the frozen anchor cost.
        for row in new_summaries:
            base = row["method"].rsplit("__endpoint_", 1)[0]
            row["requests"] = int(old_table[base]["requests"])
            row["reported_tokens"] = int(old_table[base]["reported_tokens"])
        results = {**read_json(PRIOR / target / "results.json"), **new_results}
        predictions = {**previous, **additions}
        summaries = old_summaries + new_summaries
        records = csv_rows(PRIOR / target / "per-record.csv") + new_records
        folder = destination / target
        export(folder / "predictions.json", predictions)
        export(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        (folder / "all-methods.md").write_text(full_table(summaries), encoding="utf-8")
        table = {(row["method"], row["case_id"]): row for row in records}
        for base in read_json(out / "protocol.json")["anchors"][target]:
            for rule in RULES:
                treatment = method_name(base, rule)
                bounds, details = common_boundaries(results, predictions, labels, base, treatment)
                row = {"target": target, "base": base, "rule": rule, "control": base, "treatment": treatment,
                       **compare_cases(labels, predictions, results, table, list(labels), base, treatment), **bounds}
                comparisons.append(row)
                boundary_rows.extend({"target": target, "base": base, "rule": rule, **detail} for detail in details)
                for cid, reference in labels.items():
                    actual = np.array([int(day["activity_label"]) for day in reference])
                    corrections.append({"target": target, "base": base, "rule": rule, "case_id": cid,
                                        **correction_utility(actual, np.array(previous[base][cid]["activity"]), np.array(additions[treatment][cid]["activity"]), actual >= 0)})
        if any(results[name] != value for name, value in read_json(PRIOR / target / "results.json").items()):
            raise ValueError("An original result dictionary changed")
        if csv_rows(folder / "summary.csv")[:len(old_summaries)] != old_summaries:
            raise ValueError("Historical metrics or deployment cost changed")
    write_csv(destination / "comparisons.csv", comparisons)
    write_csv(destination / "common-boundaries.csv", boundary_rows)
    write_csv(destination / "per-record-corrections.csv", corrections)
    main = {target: [row for row in comparisons if row["target"] == target and row["base"].startswith("direct_visual") and row["rule"] == "selective"] for target in TARGETS}
    unconditional = [row for row in comparisons if row["target"] == TARGETS[0] and row["base"].startswith("direct_visual") and row["rule"] == "unconditional_fit"]

    def mean(rows, field):
        return float(np.mean([row[field] for row in rows]))

    process, author = main[TARGETS[0]], main[TARGETS[1]]
    weights = [row["common_matches"] for row in process]
    boundary_change = float(np.average([row["common_endpoint_mae_difference"] for row in process], weights=weights)) if all(weights) else None
    checks = {
        "process_mean_f1_improves": mean(process, "f1_difference") > 0,
        "process_net_correct_positive": sum(row["net_correct"] for row in process) > 0,
        "process_mean_weak_nondecrease": mean(process, "treatment_weak_recall") >= mean(process, "control_weak_recall"),
        "process_two_benefiting_records_each": all(row["benefited_records"] >= 2 for row in process),
        "process_common_boundary_improves": boundary_change is not None and boundary_change < 0,
        "selective_mean_f1_beats_unconditional": mean(process, "treatment_f1") > mean(unconditional, "treatment_f1"),
        "author_mean_f1_nondecrease": mean(author, "f1_difference") >= 0,
        "author_net_correct_positive": sum(row["net_correct"] for row in author) > 0,
        "author_mean_weak_nondecrease": mean(author, "treatment_weak_recall") >= mean(author, "control_weak_recall"),
    }
    export(destination / "decision.json", {"decision": "CONTINUE_ENDPOINT_CONFIRMATION" if all(checks.values()) else "STOP_ENDPOINT_FIT",
                                         "checks": checks, "direct_comparisons": main,
                                         "pooled_process_common_endpoint_mae_change": boundary_change,
                                         "new_api_calls": 0, "independent_confirmation": False})
    export(destination / "manifest.json", {"files": {str(path.relative_to(destination)): sha(path) for path in destination.rglob("*") if path.is_file()}})
    print(read_json(destination / "decision.json")["decision"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        if args.analysis is None:
            parser.error("score requires --analysis")
        score(args.out.resolve(), args.analysis.resolve())
