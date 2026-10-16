"""Explore a bounded raw-conflict policy using frozen historical local judgments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
)
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "artifacts/landslide-window-optimization-2026-10-08/run-v1"
    initial_run = ROOT / "artifacts/landslide-local-views-2026-10-07/run-v1"
    results = read_json(source / "screen-results.json")
    initial = read_json(initial_run / "predictions.json")["activity_global"]
    numeric = read_json(source / "numeric-predictions.json")
    model = XGBClassifier()
    model.load_model(source / "stage-xgboost.json")
    args.out.mkdir(parents=True, exist_ok=False)
    # Preserve the policy before examining any derived effect metric.
    policy = ReviewPolicy()
    save_json(args.out / "policy.json", vars(policy))
    cases, features, selections, methods = {}, {}, {}, {}
    for cid, values in sorted(initial.items()):
        case, feature = load_case_features(source, cid)
        cases[cid], features[cid] = case, feature
        screen = sorted([row for row in results
                         if row["case_id"] == cid and row["method"] == "window_single"],
                        key=lambda row: row["start"])
        windows = [(row["start"], row["stop"]) for row in screen]
        activity = np.array(values["activity"])
        long_evidence = raw_motion_evidence(case, feature)
        priorities, selected = review_priorities(activity, feature, long_evidence, windows, policy)
        selections[cid] = {"priorities": priorities, "selected": selected}
        selected_reviews = [screen[row["index"]] for row in selected]
        stage_codes = model.predict(feature.matrix).astype(int) + 1
        gates = {"direct_visual": ActivityPrediction(activity, np.where(activity == 0, 0, -1)),
                 "fixed_cached_local": combine_reviews(activity, screen, feature.observed),
                 "adaptive_cached_local": combine_reviews(activity, selected_reviews,
                                                          feature.observed)}
        for name, prediction in gates.items():
            methods.setdefault(name, {})[cid] = learned_stages(
                stage_codes, prediction.activity, feature.supported)
    for name in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage"):
        methods[name] = {cid: ActivityPrediction(
            np.array(numeric[name][cid]["activity"]), np.array(numeric[name][cid]["stage"]))
            for cid in cases}
    save_json(args.out / "selections.json", selections)
    save_json(args.out / "predictions.json", {
        name: {cid: value.to_dict() for cid, value in predictions.items()}
        for name, predictions in methods.items()})
    # Reference loaded only after predictions and selections have been saved.
    with (source / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reference_rows = [row for row in reader if row["case_id"] in cases]
        fieldnames = reader.fieldnames
    reference_path = args.out / "reference-subset.csv"
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reference_rows)
    labels = load_labels(reference_path, cases)
    scores = {}
    for name, predictions in methods.items():
        per_case = {cid: evaluate_case(labels[cid], value) for cid, value in predictions.items()}
        scores[name] = {"aggregate": aggregate_scores(list(per_case.values())),
                        "per_case": per_case}
    save_json(args.out / "results.json", scores)
    save_json(args.out / "manifest.json", {
        "status": "complete", "new_model_calls": 0,
        "scope": "Historical development replay; not contemporaneous or independent evaluation",
        "source_files": {str(path.relative_to(ROOT)): sha(path) for path in (
            source / "screen-results.json", source / "numeric-predictions.json",
            initial_run / "predictions.json", source / "reference-labels.csv")},
        "selected_reviews": sum(len(value["selected"]) for value in selections.values()),
        "all_fixed_reviews": sum(row["method"] == "window_single" for row in results)})
    for name, score in scores.items():
        row = score["aggregate"]
        print(name, row["daily"], "weak", row["weak_detected_days"],
              "stage", row["stage_macro_f1"], flush=True)


if __name__ == "__main__":
    main()
