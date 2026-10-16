"""Test a history-compression admission gate using frozen complete local answers."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_scale_review import compressed_review_targets, historical_scale_gains


def execute(source, out):
    verify_freeze(source)
    for name, expected in read_json(source / "prediction-freeze.json").items():
        if sha(source / name) != expected:
            raise ValueError("Existing visual predictions or selections changed")
    config = read_json(source / "protocol.json")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {
        "scope": "Exposed complete24-record background development; zero new API calls, not independent",
        "source": str(source.resolve()), "primary": "scale_gate_image", "repetitions": 2,
        "change": "Admit a subset of the original at-most-two targets only when original missed "
                  "raw-motion support>=14 and historical raw range/local plot base span>=4 on any axis",
        "gain": "Observed global peak-to-peak / max(local peak-to-peak,6*existing noise,10mm); "
                "exclude artificial global60mm floor and reference offset; fourfold is two octaves, "
                "an engineering choice, not a proven perceptual limit",
        "no_threshold_search": True, "no_new_ranking_or_targets": True,
        "revision": "Original49-day raw conflict scope inside retained targets; no new per-day intersection",
        "stages": "Same frozen background train24 stage head, gated on final activity; not native VLM ASD",
        "continue_gate": "Both repeats dailyF1>direct; weakRecall>=direct; FP<=80% of old adaptive scope; "
                         "eventF1 IoU.3>=direct; true additions in>=2 records; local requests<old adaptive",
        "limits": "Gate design uses previously seen failure patterns; test scores are development. "
                  "Caching only tests admission, not a complete reranking across all windows",
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    policy = ReviewPolicy()
    original = read_json(source / "predictions.json")
    choices = read_json(source / "selections.json")
    stage_head = XGBClassifier()
    stage_head.load_model(source / "stage-xgboost.json")
    bindings = [Path(__file__), ROOT / "src/gnss_sim/landslide_scale_review.py",
                source / "predictions.json", source / "selections.json", source / "stage-xgboost.json",
                source / "numeric-predictions.json"]
    predictions, selections, records, permissions, costs = {}, {}, {}, {}, {}
    for cid in config["case_ids"]:
        case, feature = load_case_features(source, cid)
        records[cid] = case, feature
        bindings.extend((source / "inputs" / cid).glob("*.json"))
        windows = regular_windows(len(case.dates), 180, 90)
        gains = historical_scale_gains(case, feature.noise, windows)
        long_evidence = raw_motion_evidence(case, feature)
        codes = stage_head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            selection = choices[f"{cid}-r{rep}"]
            _, neighbourhood = conflict_neighbourhood(initial, feature, long_evidence, policy)
            target_lookup = {row["index"]: row for row in selection["priorities"]}
            adaptive = selection["selected"]
            fixed = [target_lookup[index] for index in selection["fixed_indices"]]
            main = compressed_review_targets(adaptive, gains)
            budget_order = sorted(adaptive, key=lambda row: hashlib.sha256(
                f"{cid}:{rep}:{row['index']}".encode()).hexdigest())
            matched = sorted(budget_order[:len(main)], key=lambda row: row["start"])
            selections[f"{cid}-r{rep}"] = {"gains": gains, "original": adaptive, "main": main,
                                           "matched_budget": matched, "fixed": fixed}
            arms = {
                "old_adaptive_image": (adaptive, "adaptive_no_feedback"),
                "fixed_image": (fixed, "fixed_image_review"),
                "scale_gate_image": (main, "adaptive_no_feedback"),
                "scale_gate_numeric_text": (main, "adaptive_numeric_feedback"),
                "scale_without_missed_gate": ([row for row in adaptive if gains[row["index"]]["gain"] >= 4],
                                              "adaptive_no_feedback"),
                "missed_only_image": ([row for row in adaptive if row["missed_motion_days"] >= 14],
                                      "adaptive_no_feedback"),
                "matched_budget_image": (matched, "adaptive_no_feedback"),
                "fixed_scale_gate_image": (compressed_review_targets(fixed, gains), "fixed_image_review"),
            }
            direct = f"direct_visual_r{rep}"
            predictions.setdefault(direct, {})[cid] = original[direct][cid]
            direct_receipt = read_json(source / "requests" / f"{cid}-direct-r{rep}.response.json")
            costs.setdefault(direct, []).append(direct_receipt)
            for name, (targets, prior) in arms.items():
                method = f"{name}_r{rep}"
                scope, _, _ = target_scopes(neighbourhood, feature.observed, targets)
                reviewed = np.asarray(original[f"{prior}_r{rep}"][cid]["activity"])
                activity = scoped_activity(initial, reviewed, scope, feature.observed)
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                permissions[f"{method}-{cid}"] = np.flatnonzero(scope).tolist()
                costs.setdefault(method, []).append(direct_receipt)
                suffix = "numeric" if prior == "adaptive_numeric_feedback" else "image"
                for target in targets:
                    request_id = f"{cid}-window-{target['index']:02d}-{suffix}-r{rep}"
                    costs[method].append(read_json(source / "requests" / f"{request_id}.response.json"))
    numerical = read_json(source / "numeric-predictions.json")
    predictions.update({name: values for name, values in numerical.items() if name != "ungated_rule"})
    save_json(out / "input-bindings.json", {str(path.resolve()): sha(path) for path in bindings})
    save_json(out / "selections.json", selections)
    save_json(out / "permissions.json", permissions)
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
                                              ("predictions.json", "selections.json", "permissions.json")})
    # Exposure is declared above. Saving before scoring does not confer independence.
    reference = source / "analysis-v1/reference-subset.csv"
    labels = load_labels(reference, {cid: pair[0] for cid, pair in records.items()})
    save_json(out / "reference-binding.json", {str(reference.resolve()): sha(reference)})
    results, weak, changes, boundaries = {}, {}, {}, {}
    for method, values in predictions.items():
        per_case, per_changes, detail = {}, {}, {}
        weak_total, weak_detected = 0, 0
        for cid, value in values.items():
            feature = records[cid][1]
            activity = np.asarray(value["activity"])
            prediction = learned_stages(np.asarray(value["stage"]), activity, feature.observed)
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            weak_mask = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak_mask.sum())
            weak_detected += int((weak_mask & (activity == 1)).sum())
            detail[cid] = {str(t): boundary_errors(labels[cid], activity, t) for t in (.3, .5)}
            if method.endswith(("_r0", "_r1")):
                rep = int(method[-1])
                initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
                per_changes[cid] = {
                    "gained_tp": int(((actual == 1) & (initial != 1) & (activity == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (initial == 1) & (activity != 1)).sum()),
                    "added_fp": int(((actual == 0) & (initial != 1) & (activity == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (initial == 1) & (activity != 1)).sum()),
                }
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
        weak[method] = {"days": weak_total, "detected": weak_detected, "recall": weak_detected / weak_total}
        boundaries[method] = detail
        if per_changes:
            changes[method] = {"per_case": per_changes,
                               "total": {key: sum(row[key] for row in per_changes.values())
                                         for key in ("gained_tp", "lost_tp", "added_fp", "removed_fp")}}
    internal = verify_metrics(predictions, results, labels, records)
    usage = {method: {"requests": len(receipts),
                      "reported_tokens": sum(row["usage"]["total_tokens"] for row in receipts),
                      "shared_cache": True, "new_requests": 0} for method, receipts in costs.items()}
    decisions = []
    for rep in (0, 1):
        main, direct, old = [f"{stem}_r{rep}" for stem in
                             ("scale_gate_image", "direct_visual", "old_adaptive_image")]
        a, d, o = [results[method]["aggregate"] for method in (main, direct, old)]
        records_gained = sum(row["gained_tp"] > 0 for row in changes[main]["per_case"].values())
        passed = (a["daily"]["f1"] > d["daily"]["f1"]
                  and weak[main]["recall"] >= weak[direct]["recall"]
                  and a["daily"]["fp"] <= .8 * o["daily"]["fp"]
                  and a["events"]["f1"] >= d["events"]["f1"]
                  and records_gained >= 2 and usage[main]["requests"] < usage[old]["requests"])
        decisions.append({"repeat": rep, "passed": passed, "daily": a["daily"],
                          "event": a["events"], "weak": weak[main], "gained_records": records_gained,
                          "usage": usage[main], **changes[main]["total"]})
    for filename, value in (("results.json", results), ("weak-results.json", weak), ("changes.json", changes),
                            ("boundaries.json", boundaries), ("internal-stages.json", internal), ("costs.json", usage)):
        save_json(out / filename, value)
    decision = {"allows_new_confirmation": all(row["passed"] for row in decisions), "repeats": decisions,
                "configs": len(results), "new_api_requests": 0}
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
