"""Probe causally available strong prefixes with complete cached visual records and ablations."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from fit_landslide_causal_direction_controls import direction_features
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_direction_review import review_directional_additions
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages

ARMS = {
    "direction_main": ("adaptive_no_feedback", "direction_and_motion"),
    "direction_fixed": ("fixed_image_review", "direction_and_motion"),
    "direction_without_history": ("adaptive_no_feedback", "without_direction"),
    "direction_without_motion": ("adaptive_no_feedback", "without_motion"),
    "direction_visual_only": ("adaptive_no_feedback", "visual_only"),
    "direction_numeric_text": ("adaptive_numeric_feedback", "direction_and_motion"),
}


def execute(source, controls, out):
    started = time.perf_counter()
    verify_freeze(source)
    for name, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / name) != expected:
            raise ValueError("Existing visual predictions or selections changed")
    if sha(controls / "predictions.json") != read_json(controls / "manifest.json")["prediction_sha256"]:
        raise ValueError("Direction control predictions changed")
    config = read_json(source / "protocol.json")
    data = ROOT / config["source_bundle"]
    calibration = read_json(data / "long-selection.json")["raw_multiscale_rule"]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "scope": "All24 exposed background development records and two cached repeats; not independent",
        "source": str(source.resolve()), "controls": str(controls.resolve()), "primary": "direction_main",
        "hypothesis": "Already-observed strong prefixes can anchor weak tails without waiting for episode end",
        "previous_version": "direction-review-development-v1; no new positives; stopped. This new availability hypothesis changes no thresholds",
        "proposal": "Novel local-visual positive runs in original49-day authority; bridge<=7 unknown "
                    "missing days for evidence only, no negative gap or missing label, no new boundaries",
        "anchor": "Most recent14 observed supporting centers in an ongoing raw42 strong run using existing calibrated "
                  "rate/SNR. Full42+7 footprint precedes proposal. Numerical anchor is not expert activity",
        "interval": "Three disjoint14-day raw blocks at first/middle/last of novel proposal>=42days; "
                    ">=8 complete observations per block, no internal>7day gap",
        "main": "Both changes cosine>=.5 to anchor, projected rate>=.03 and projected noise ratio>=2. "
                "Whole proposal accepted or left unknown; all initial positives retained",
        "parameters": ".5 fixed sixty-degree geometry, not fitted; .03/2 and duration/gap from existing rules",
        "anchor_calibration": calibration, "threshold_search": False,
        "continue_gate": "Both repeats dailyF1>direct, weakRecall>=direct, eventF1>=direct, "
                         "FP additions<=half visual-only additions, TP gains in>=2 records; "
                         "mainF1>without-history and without-motion. Otherwise stop, no best-arm rescue",
        "stages": "Same background-trained stage head gated by final activity, not native VLM ASD",
        "new_api_requests": 0, "new_records": 0,
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    original = read_json(source / "predictions.json")
    choices = read_json(source / "selections.json")
    head = XGBClassifier()
    head.load_model(source / "stage-xgboost.json")
    predictions, proposals, anchors_saved, permissions, costs, records = {}, {}, {}, {}, {}, {}
    bindings = [Path(__file__), ROOT / "src/gnss_sim/landslide_direction_review.py",
                ROOT / "scripts/fit_landslide_causal_direction_controls.py",
                ROOT / "src/gnss_sim/landslide_causal_direction_review.py", source / "predictions.json",
                source / "selections.json", source / "numeric-predictions.json", source / "stage-xgboost.json",
                data / "long-selection.json", controls / "predictions.json", controls / "frozen-selection.json"]
    policy = ReviewPolicy()
    for cid in config["case_ids"]:
        case, feature = load_case_features(source, cid)
        records[cid] = case, feature
        bindings.extend((source / "inputs" / cid).glob("*.json"))
        _, _, _, anchors, raw, long = direction_features(case, feature, calibration)
        anchors_saved[cid] = anchors
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            selection = choices[f"{cid}-r{rep}"]
            _, neighbourhood = conflict_neighbourhood(initial, feature, long, policy)
            lookup = {row["index"]: row for row in selection["priorities"]}
            direct = f"direct_visual_r{rep}"
            predictions.setdefault(direct, {})[cid] = original[direct][cid]
            direct_receipt = read_json(source / "requests" / f"{cid}-direct-r{rep}.response.json")
            costs.setdefault(direct, []).append(direct_receipt)
            for stem, (prior, variant) in ARMS.items():
                targets = selection["selected"]
                if prior == "fixed_image_review":
                    targets = [lookup[index] for index in selection["fixed_indices"]]
                scope = target_scopes(neighbourhood, feature.observed, targets)[0]
                local = np.asarray(original[f"{prior}_r{rep}"][cid]["activity"])
                activity, decisions = review_directional_additions(
                    initial, local, scope, feature.observed, raw, feature.noise, anchors, variant)
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                proposals[f"{method}-{cid}"] = decisions
                permissions[f"{method}-{cid}"] = np.flatnonzero(scope).tolist()
                if np.any(activity[initial == 1] != 1) or np.any(activity[~feature.observed] != -1):
                    raise ValueError("Protected initial activity or missing mask changed")
                if np.any((activity == 1) & (initial != 1) & (~scope | (local != 1))):
                    raise ValueError("Activity added without local visual permission")
                costs.setdefault(method, []).append(direct_receipt)
                suffix = "numeric" if prior == "adaptive_numeric_feedback" else "image"
                for target in targets:
                    request = f"{cid}-window-{target['index']:02d}-{suffix}-r{rep}"
                    costs[method].append(read_json(source / "requests" / f"{request}.response.json"))
            for stem, prior in (("old_adaptive_image", "adaptive_no_feedback"),
                                ("old_fixed_image", "fixed_image_review")):
                targets = selection["selected"] if stem == "old_adaptive_image" else [
                    lookup[index] for index in selection["fixed_indices"]]
                scope = target_scopes(neighbourhood, feature.observed, targets)[0]
                local = np.asarray(original[f"{prior}_r{rep}"][cid]["activity"])
                activity = scoped_activity(initial, local, scope, feature.observed)
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                cost_stem = "direction_main" if stem == "old_adaptive_image" else "direction_fixed"
                # Same targets and requests as the corresponding proposal arm.
                costs[method] = costs[f"{cost_stem}_r{rep}"].copy()
    numerical = read_json(source / "numeric-predictions.json")
    predictions.update({name: values for name, values in numerical.items() if name != "ungated_rule"})
    predictions.update(read_json(controls / "predictions.json"))
    for filename, value in (("predictions.json", predictions), ("proposals.json", proposals),
                            ("anchors.json", anchors_saved), ("permissions.json", permissions)):
        save_json(out / filename, value)
    save_json(out / "input-bindings.json", {str(path.resolve()): sha(path) for path in bindings})
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("predictions.json", "proposals.json", "anchors.json", "permissions.json")})
    # Ordering never makes already exposed references independent.
    reference = source / "analysis-v1/reference-subset.csv"
    labels = load_labels(reference, {cid: pair[0] for cid, pair in records.items()})
    save_json(out / "reference-binding.json", {str(reference.resolve()): sha(reference)})
    results, weak, changes, boundaries = {}, {}, {}, {}
    for method, values in predictions.items():
        per_case, per_changes, detail = {}, {}, {}
        weak_days, weak_detected = 0, 0
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]), np.asarray(value["stage"]),
                                            status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            weak_mask = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_days += int(weak_mask.sum())
            weak_detected += int((weak_mask & (prediction.activity == 1)).sum())
            detail[cid] = {str(t): boundary_errors(labels[cid], prediction.activity, t) for t in (.3, .5)}
            if method.endswith(("_r0", "_r1")):
                initial = np.asarray(original[f"direct_visual_r{method[-1]}"][cid]["activity"])
                activity = prediction.activity
                per_changes[cid] = {
                    "gained_tp": int(((actual == 1) & (initial != 1) & (activity == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (initial == 1) & (activity != 1)).sum()),
                    "added_fp": int(((actual == 0) & (initial != 1) & (activity == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (initial == 1) & (activity != 1)).sum()),
                }
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
        weak[method] = {"days": weak_days, "detected": weak_detected, "recall": weak_detected / weak_days}
        boundaries[method] = detail
        if per_changes:
            changes[method] = {"per_case": per_changes, "total": {
                key: sum(row[key] for row in per_changes.values())
                for key in ("gained_tp", "lost_tp", "added_fp", "removed_fp")}}
    stages = verify_metrics(predictions, results, labels, records)
    usage = {method: {"requests": len(receipts), "reported_tokens": sum(row["usage"]["total_tokens"] for row in receipts),
                      "shared_cache": True, "new_requests": 0} for method, receipts in costs.items()}
    decisions = []
    for rep in (0, 1):
        main, direct, visual, no_history, no_motion = [f"{stem}_r{rep}" for stem in
            ("direction_main", "direct_visual", "direction_visual_only", "direction_without_history", "direction_without_motion")]
        a, d = [results[name]["aggregate"] for name in (main, direct)]
        gains = changes[main]
        records_gained = sum(row["gained_tp"] > 0 for row in gains["per_case"].values())
        criteria = {
            "daily_gain": a["daily"]["f1"] > d["daily"]["f1"],
            "weak_nonloss": weak[main]["recall"] >= weak[direct]["recall"],
            "event_nonloss": a["events"]["f1"] >= d["events"]["f1"],
            "fp_filter": gains["total"]["added_fp"] <= .5 * changes[visual]["total"]["added_fp"],
            "multiple_gains": records_gained >= 2,
            "history_increment": a["daily"]["f1"] > results[no_history]["aggregate"]["daily"]["f1"],
            "motion_increment": a["daily"]["f1"] > results[no_motion]["aggregate"]["daily"]["f1"],
        }
        decisions.append({"repeat": rep, "passed": all(criteria.values()), "criteria": criteria,
                          "daily": a["daily"], "events": a["events"], "weak": weak[main],
                          "gained_records": records_gained, **gains["total"]})
    for filename, value in (("results.json", results), ("weak-results.json", weak), ("changes.json", changes),
                            ("boundaries.json", boundaries), ("internal-stages.json", stages), ("costs.json", usage)):
        save_json(out / filename, value)
    decision = {"allows_new_confirmation": all(row["passed"] for row in decisions), "repeats": decisions,
                "configs": len(results), "new_api_requests": 0}
    save_json(out / "decision.json", decision)
    save_json(out / "state.json", {"status": "complete", "wall_seconds_including_checks_scoring": time.perf_counter() - started})
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.controls, args.out)


if __name__ == "__main__":
    main()
