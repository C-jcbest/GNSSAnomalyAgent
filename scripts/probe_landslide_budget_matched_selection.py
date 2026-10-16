"""Separate record admission from window placement in the original visual method."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import evaluate_population
from analyze_landslide_review_utility_v2 import verify_receipts
from finalize_landslide_adaptive_evidence import verify_manifest
from probe_landslide_revision_scope import (
    conflict_neighbourhood,
    scoped_activity,
    target_scopes,
)
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity


def execute(run, out):
    verify_freeze(run)
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Completed main-method predictions or selections changed")
    config = read_json(run / "protocol.json")
    data = ROOT / config["source_bundle"]
    verify_manifest(data)
    analysis = run / "analysis-v1"
    verify_manifest(analysis)
    ids = config["case_ids"]
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"] for cid in ids}
    population = {family: sum(value == family for value in families.values()) for family in sorted(set(families.values()))}
    if len(ids) != 12 or len(population) != 6 or set(population.values()) != {2}:
        raise ValueError("This ablation requires all twelve original records, two per actual family")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    protocol = {
        "date": "2026-10-10", "scope": "Post-confirmation mechanism ablation on all twelve exposed controlled records; no new validation claim",
        "run": str(run.resolve()), "case_ids": ids, "actual_families": families,
        "actual_population": population, "new_model_requests": 0, "new_records": 0,
        "main_method": "Original21/42 raw-conflict selection and original49-day scope; unchanged",
        "primary_ablation": "same_count_fixed_early", "repetitions": 2,
        "count_control": "Same number K=0/1/2 of local reviews per record and repeat as the original selector",
        "position_control": "Existing two equal-spacing fixed targets; retain earliest K in the primary ablation. This prefix rule was used in the earlier utility experiment, not optimized here",
        "sensitivity": "Also retain latest K fixed targets. K=0 and K=2 coincide across the two fixed controls; K=1 differs. Both reported without choosing a winner",
        "revision_control": "Whole-target variants isolate location with the same180-day extent and K; original49-day scope variants retain the deployed method's rule. Editable observed-day counts may differ with missingness and conflict placement",
        "hypothesis": "Original adaptive placement has greater activityF1 and positive net definite correction than same-count fixed early in both repeats, with no weakRecall loss. Report comparison to late sensitivity in both repeats without replacing the primary",
        "interpretation": "These are reachable comparative conditions: different selected targets can revise both positive and negative states. No threshold tuning or output rejection rule is added. Failure leaves placement value unproven",
        "counterfactual_limits": "Cached local prompts/images do not contain other local replies; replay supports deterministic schedule ablation conditional on the saved responses. It does not simulate a new stochastic VLM repetition",
        "stage": "Same original trained stage head after final activity; no native VLM ASD claim",
        "baseline": "Nine fitted original commontrain24/validation12 rules/PELT/XGBoost, recomputed on identical records and scoring",
    }
    save_json(out / "protocol.json", protocol)
    bindings = {str(Path(__file__).resolve()): sha(Path(__file__))}
    for path in (run / "prediction-freeze.json", run / "selections.json", run / "predictions.json",
                 analysis / "predictions.json", run / "stage-xgboost.json", run / "reference-subset.csv"):
        bindings[str(path.resolve())] = sha(path)
    save_json(out / "bindings.json", bindings)
    save_json(out / "pre-score-freeze.json", {
        path.name: sha(path) for path in out.iterdir() if path.is_file()})
    original = read_json(analysis / "predictions.json")
    predictions = dict(original)
    records = {cid: load_case_features(run, cid) for cid in ids}
    numerical = numerical_predictions(data, records)
    for method, values in numerical.items():
        if values != predictions[method]:
            raise ValueError("Common numerical controls do not reconstruct")
    receipts = verify_receipts(run)
    choices = read_json(run / "selections.json")
    head = XGBClassifier()
    head.load_model(run / "stage-xgboost.json")
    used, schedules, scopes = {}, {}, {}
    for cid, (case, feature) in records.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            initial = np.array(original[f"direct_visual_r{rep}"][cid]["activity"])
            choice = choices[f"{cid}-r{rep}"]
            windows = [(row["start"], row["stop"]) for row in choice["priorities"]]
            priorities, adaptive = review_priorities(initial, feature, long, windows, ReviewPolicy())
            if priorities != choice["priorities"] or adaptive != choice["selected"]:
                raise ValueError("The original selection failed to replay")
            fixed = [priorities[index] for index in choice["fixed_indices"]]
            count = len(adaptive)
            early = fixed[:count]
            late = fixed[len(fixed) - count:] if count else []
            schedules[f"{cid}-r{rep}"] = {"adaptive": adaptive, "fixed_early": early, "fixed_late": late,
                                            "maximum_fixed": fixed, "K": count}
            neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())[1]
            for stem, targets in (("same_count_fixed_early", early), ("same_count_fixed_late", late),
                                   ("adaptive_no_feedback", adaptive)):
                keys = [f"{cid}-direct-r{rep}"]
                local = []
                for target in targets:
                    key = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                    receipt = receipts[key]
                    activity = parse_window_activity(json.loads(receipt["output"]), feature.observed,
                                                     target["start"], target["stop"])
                    local.append({"start": target["start"], "stop": target["stop"],
                                  "activity": activity.tolist(), "status": "success"})
                    keys.append(key)
                reviewed = combine_reviews(initial, local, feature.observed).activity
                authority = target_scopes(neighbourhood, feature.observed, targets)[0]
                for variant in ("whole_target", "conflict_scope"):
                    method = f"{stem}_r{rep}__{variant}"
                    activity = reviewed.copy()
                    if variant == "conflict_scope":
                        activity = scoped_activity(initial, reviewed, authority, feature.observed)
                    prediction = learned_stages(codes, activity, feature.supported).to_dict()
                    if stem == "adaptive_no_feedback":
                        if original[method][cid] != prediction:
                            raise ValueError("Original adaptive prediction no longer reconstructs")
                    else:
                        predictions.setdefault(method, {})[cid] = prediction
                    used.setdefault(method, {})[cid] = keys
                    scopes.setdefault(method, {})[cid] = np.flatnonzero(authority).tolist()
    save_json(out / "predictions.json", predictions)
    save_json(out / "schedules.json", schedules)
    save_json(out / "used-requests.json", used)
    save_json(out / "scopes.json", scopes)
    save_json(out / "prediction-freeze.json", {
        name: sha(out / name) for name in ("predictions.json", "schedules.json", "used-requests.json", "scopes.json")})
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    results, summaries, per_case = evaluate_population(predictions, labels)
    internal = verify_metrics(predictions, results, labels, records)
    changes, daily = {}, []
    for row in summaries:
        method = row["method"]
        row["internal_stage_macro_f1"] = internal[method]["macro_f1"]
        if method in used:
            keys = [key for values in used[method].values() for key in values]
            row["deployment_requests"] = len(keys)
            row["reported_tokens"] = sum(receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys)
        else:
            costs = read_json(run / "costs.json").get(method.partition("__")[0])
            if costs is None and method not in numerical:
                raise ValueError("A visual control has no attributable deployment cost")
            row["deployment_requests"] = costs["requests"] if costs is not None else 0
            row["reported_tokens"] = costs["tokens"] if costs is not None else 0
        for cid, value in predictions[method].items():
            activity = np.array(value["activity"])
            actual = np.array([int(item["activity_label"]) for item in labels[cid]])
            if method.startswith("adaptive_no_feedback") and method.endswith(("__whole_target", "__conflict_scope")):
                suffix = method.removeprefix("adaptive_no_feedback")
                for stem in ("same_count_fixed_early", "same_count_fixed_late", "fixed_image_review"):
                    control = stem + suffix
                    before = np.array(predictions[control][cid]["activity"])
                    changes.setdefault(method, {}).setdefault(control, {})[cid] = {
                        **correction_utility(actual, before, activity, actual >= 0),
                        "gained_tp": int(((actual == 1) & (before != 1) & (activity == 1)).sum()),
                        "lost_tp": int(((actual == 1) & (before == 1) & (activity != 1)).sum()),
                        "added_fp": int(((actual == 0) & (before != 1) & (activity == 1)).sum()),
                        "removed_fp": int(((actual == 0) & (before == 1) & (activity != 1)).sum()),
                    }
            for day, item in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day,
                              "actual_activity": item["activity_label"], "actual_stage": item["feature"],
                              "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
    indexed = {row["method"]: row for row in summaries}
    decisions = []
    for rep in (0, 1):
        for variant in ("whole_target", "conflict_scope"):
            main = f"adaptive_no_feedback_r{rep}__{variant}"
            early = f"same_count_fixed_early_r{rep}__{variant}"
            late = f"same_count_fixed_late_r{rep}__{variant}"
            if indexed[main]["deployment_requests"] != indexed[early]["deployment_requests"] or indexed[main]["deployment_requests"] != indexed[late]["deployment_requests"]:
                raise ValueError("Matched fixed control differs in deployed call count")
            net = sum(item["net_correct"] for item in changes[main][early].values())
            gate = {"f1_greater_than_early": indexed[main]["f1"] > indexed[early]["f1"],
                    "weak_recall_nonloss_than_early": indexed[main]["weak_recall"] >= indexed[early]["weak_recall"],
                    "net_definite_correction_positive_than_early": net > 0}
            decisions.append({"repeat": rep, "variant": variant, "gate": gate, "passed": all(gate.values()),
                              "net_vs_early": net, "f1_greater_than_late": indexed[main]["f1"] > indexed[late]["f1"]})
    for name, value in (("results.json", results), ("internal-stages.json", internal), ("changes.json", changes),
                        ("decision.json", {"comparisons": decisions, "development_only": True}),
                        ("checks.json", {"methods": len(predictions), "receipt_payloads": len(receipts),
                                         "numeric_case_replays": len(numerical) * len(records), "daily_rows": len(daily),
                                         "same_record_call_counts_verified": True, "new_api": 0})):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("per-case.csv", per_case), ("daily.csv", daily)):
        write_rows(out / name, rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stems = ("fixed_image_review", "same_count_fixed_early", "same_count_fixed_late", "adaptive_no_feedback")
    for rep in (0, 1):
        for axis, variant in zip(axes, ("whole_target", "conflict_scope"), strict=True):
            axis.bar(np.arange(4) + (rep - .5) * .36,
                     [indexed[f"{stem}_r{rep}__{variant}"]["f1"] for stem in stems], width=.36, label=f"Repeat {rep}")
            axis.set_xticks(np.arange(4), ["Full fixed", "K fixed early", "K fixed late", "Raw conflict"], rotation=15)
            axis.set_ylim(0, 1)
            axis.set_title(variant)
            axis.set_ylabel("Activity F1")
            axis.legend()
    fig.suptitle("Exposed controlled records | 36 full-fixed vs 26 other calls per repeat")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print(decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    execute(arguments.run, arguments.out)
