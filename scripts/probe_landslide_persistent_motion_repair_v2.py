"""Test one-sided, same-date raw/visual motion agreement on complete cached development."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_missed_motion_repair import sustained_motion_support
from gnss_sim.landslide_persistent_motion_repair import repair_persistent_motion


def execute(source, out):
    config = read_json(source / "protocol.json")
    calibration_path = ROOT / config["source_bundle"] / "long-selection.json"
    calibration = read_json(calibration_path)["raw_multiscale_rule"]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "scope": "Post-hoc continuity hypothesis on exposed development after exact-mask failures; not independent",
        "new_mechanism": "Require14 consecutive new observed agreement dates using existing activity_from_mask; "
                         "no gap filling, boundary expansion or duration tuning",
        "source": str(source.resolve()), "primary": "adaptive_no_feedback__raw_visual_additions",
        "policy": {"maximum_reviews": 2, "minimum_conflict_days": 14,
                   "minimum_rate": calibration["rate"], "minimum_signal_to_noise": calibration["snr"]},
        "calibration": "Existing common train24/validation12 raw rule selection, fixed before original test; "
                       "no new fitting, test threshold search or XGBoost substitution",
        "selection_policy": "Original0.03/2 selector and cached targets retained; only addition evidence changes",
        "selection": "Existing frozen21/42 conflict targets and identical local answers; isolate revision semantics",
        "mechanism": "Keep every initial active day; new activity requires raw support and visual positive "
                     "on that exact calendar day inside an already selected target; otherwise unresolved",
        "ablations": ["visual_additions_without_raw", "numerical_additions_without_visual",
                      "bidirectional_exact_conflict", "original_unrestricted_local"],
        "no_api": True, "no_training": True, "no_threshold_or_neighbourhood_search": True,
        "continue_gate": "Primary both repeats: dailyF1 greater than direct, weak recall not lower, "
                         "FP at least10% lower than original unrestricted local, additions true in at least2 records; "
                         "eventF1 IoU.3 also not below direct (date masking must not destroy events)",
        "limitation": "Initial false positives cannot be removed; does not prove background has no true movement",
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    bindings = [Path(__file__), ROOT / "src/gnss_sim/landslide_missed_motion_repair.py",
                source / "predictions.json", source / "selections.json", source / "stage-xgboost.json",
                source / "numeric-predictions.json", calibration_path,
                ROOT / "src/gnss_sim/landslide_persistent_motion_repair.py",
                out.parent / "calibrated-motion-repair-development-v1/predictions.json"]
    for name, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / name) != expected:
            raise ValueError("Existing predictions or selections changed")
    original = read_json(source / "predictions.json")
    choices = read_json(source / "selections.json")
    policy = ReviewPolicy(minimum_rate=calibration["rate"], minimum_signal_to_noise=calibration["snr"])
    selection_policy = ReviewPolicy()
    model = XGBClassifier()
    model.load_model(source / "stage-xgboost.json")
    predictions = dict(original)
    permissions, cases = {}, {}
    for cid in config["case_ids"]:
        case, features = load_case_features(source, cid)
        cases[cid] = case
        bindings.extend((source / "inputs" / cid).glob("*.json"))
        long_evidence = raw_motion_evidence(case, features)
        raw_support = sustained_motion_support(features, long_evidence, policy)
        stage_codes = model.predict(features.matrix).astype(int) + 1
        for rep in range(2):
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            conflicts, _ = conflict_neighbourhood(initial, features, long_evidence, selection_policy)
            selection = choices[f"{cid}-r{rep}"]
            for arm in ("fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"):
                prior_name = f"{arm}_r{rep}"
                reviewed = np.asarray(original[prior_name][cid]["activity"])
                targets = selection["selected"]
                if arm == "fixed_image_review":
                    windows = regular_windows(len(initial), 180, 90)
                    targets = [{"start": windows[index][0], "stop": windows[index][1]}
                               for index in selection["fixed_indices"]]
                covered = np.zeros(len(initial), dtype=bool)
                for target in targets:
                    covered[target["start"]:target["stop"]] = True
                permitted_raw = raw_support & covered & features.observed & (initial != 1)
                permitted_visual = covered & features.observed & (initial != 1)
                variants = {
                    "raw_visual_additions": repair_persistent_motion(initial, reviewed, permitted_raw,
                                                                  features.observed),
                    "visual_additions_without_raw": repair_persistent_motion(initial, reviewed, permitted_visual,
                                                                          features.observed),
                    "bidirectional_exact_conflict": scoped_activity(initial, reviewed,
                                                                     conflicts & covered & features.observed,
                                                                     features.observed),
                }
                numerical = initial.copy()
                numerical[activity_from_mask(permitted_raw, features.observed) == 1] = 1
                variants["numerical_additions_without_visual"] = numerical
                permission_key = f"{prior_name}-{cid}"
                permissions[permission_key] = {
                    "raw_support_days": np.flatnonzero(raw_support).tolist(),
                    "covered_days": np.flatnonzero(covered & features.observed).tolist(),
                    "raw_permitted_days": np.flatnonzero(permitted_raw).tolist(),
                    "visual_permitted_days": np.flatnonzero(permitted_visual).tolist(),
                }
                for variant, activity in variants.items():
                    if variant != "bidirectional_exact_conflict" and np.any((initial == 1) & (activity != 1)):
                        raise ValueError("One-sided repair removed initial activity")
                    name = f"{arm}__{variant}_r{rep}"
                    predictions.setdefault(name, {})[cid] = learned_stages(
                        stage_codes, activity, features.supported).to_dict()
    for method, values in read_json(source / "numeric-predictions.json").items():
        if method == "ungated_rule":
            continue
        predictions[method] = {cid: values[cid] for cid in config["case_ids"]}
    without_continuity = read_json(out.parent / "calibrated-motion-repair-development-v1/predictions.json")
    for method, values in without_continuity.items():
        if "__raw_visual_additions_r" in method:
            predictions[method.replace("__raw_visual_additions_r", "__raw_visual_additions_without_continuity_r")] = values
    save_json(out / "input-bindings.json", {str(path.resolve()): sha(path) for path in bindings})
    save_json(out / "permissions.json", permissions)
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name)
                                               for name in ("predictions.json", "permissions.json")})
    # All records were exposed previously; ordering does not confer independence.
    reference_path = source / "analysis-v1" / "reference-subset.csv"
    labels = load_labels(reference_path, cases)
    save_json(out / "reference-binding.json", {str(reference_path.resolve()): sha(reference_path)})
    results, changes, weak, boundaries = {}, {}, {}, {}
    for method, values in predictions.items():
        per_case, per_changes, weak_days, weak_detected = {}, {}, 0, 0
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]), np.asarray(value["stage"]),
                                            status=value["status"], gated=value["gated"],
                                            stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            weak_mask = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_days += int(weak_mask.sum())
            weak_detected += int((weak_mask & (prediction.activity == 1)).sum())
            if method.endswith(("_r0", "_r1")):
                rep = int(method[-1])
                before = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
                after = prediction.activity
                per_changes[cid] = {
                    "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                    "added_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum()),
                }
            boundaries.setdefault(method, {})[cid] = {str(t): boundary_errors(labels[cid], prediction.activity, t)
                                                       for t in (0.3, 0.5)}
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
        weak[method] = {"days": weak_days, "detected": weak_detected,
                        "recall": weak_detected / weak_days if weak_days else None}
        if per_changes:
            changes[method] = {"per_case": per_changes,
                               "total": {key: sum(row[key] for row in per_changes.values())
                                         for key in next(iter(per_changes.values()))}}
    save_json(out / "results.json", results)
    save_json(out / "changes.json", changes)
    save_json(out / "weak-results.json", weak)
    save_json(out / "boundaries.json", boundaries)
    decisions = []
    for rep in range(2):
        method = f"adaptive_no_feedback__raw_visual_additions_r{rep}"
        direct = f"direct_visual_r{rep}"
        unrestricted = f"adaptive_no_feedback_r{rep}"
        score = results[method]["aggregate"]["daily"]
        records = sum(row["gained_tp"] > 0 for row in changes[method]["per_case"].values())
        passed = (score["f1"] > results[direct]["aggregate"]["daily"]["f1"]
                  and weak[method]["recall"] >= weak[direct]["recall"]
                  and score["fp"] <= 0.9 * results[unrestricted]["aggregate"]["daily"]["fp"]
                  and records >= 2
                  and results[method]["aggregate"]["events"]["f1"]
                  >= results[direct]["aggregate"]["events"]["f1"])
        decisions.append({"repeat": rep, "passed": passed, "daily": score, "weak": weak[method],
                          "gained_tp_records": records, **changes[method]["total"]})
    decision = {"allows_prototype_extension": all(row["passed"] for row in decisions),
                "repeats": decisions, "new_api_requests": 0, "configs": len(results)}
    save_json(out / "decision.json", decision)
    save_json(out / "state.json", {"status": "complete"})
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.out)


if __name__ == "__main__":
    main()
