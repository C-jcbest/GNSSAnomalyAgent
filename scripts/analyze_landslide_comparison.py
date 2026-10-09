"""Audit a completed development run without changing predictions or tuning parameters."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from run_landslide_comparison import (
    ROOT,
    ActivityPrediction,
    LandslideInput,
    aggregate_scores,
    evaluate_case,
    load_labels,
    save_json,
    sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    args = parser.parse_args()
    target = args.visual / "audit"
    target.mkdir(exist_ok=False)
    manifest = json.loads((args.numeric / "manifest.json").read_text(encoding="utf-8"))
    cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
             for path in (Path(manifest["data"]) / "cases").glob("*/input.json")}
    labels = load_labels(Path(manifest["labels"]), cases)
    distributions = {}
    for group, case_ids in manifest["split"].items():
        counts = Counter(row["feature"] for cid in case_ids for row in labels[cid])
        weak_cases = [cid for cid in case_ids if any(row["feature"] == "slow_displacement" for row in labels[cid])]
        distributions[group] = {"records": len(case_ids), "feature_days": dict(counts), "weak_cases": weak_cases}
    save_json(target / "label-distribution.json", distributions)

    predictions = json.loads((args.visual / "predictions.json").read_text(encoding="utf-8"))
    results = json.loads((args.visual / "results.json").read_text(encoding="utf-8"))
    case_ids = manifest["split"]["development_evaluation"]
    sensitivity = {}
    quiet = [cid for cid in case_ids if not any(int(row["activity_label"]) == 1 for row in labels[cid])]
    quiet_results = {}
    for name, records in predictions.items():
        scores = []
        for cid in case_ids:
            row = records[cid]
            prediction = ActivityPrediction(np.array(row["activity"]), np.array(row["stage"]),
                                             row["status"], row["gated"], row["stage_status"])
            scores.append(evaluate_case(labels[cid], prediction, iou_threshold=.5))
        sensitivity[name] = aggregate_scores(scores)
        quiet_results[name] = {cid: results[name]["per_case"][cid] for cid in quiet}
    save_json(target / "iou-05-sensitivity.json", sensitivity)
    save_json(target / "negative-control-records.json", quiet_results)

    # Resample whole records in pairs. Six records are far too few for confirmatory inference.
    rng = np.random.default_rng(20261007)
    resamples = rng.integers(0, len(case_ids), size=(2000, len(case_ids)))
    comparisons = [("visual_rule", "robust_rule"), ("visual_model", "numeric_model"),
                   ("visual_model", "visual_rule"), ("visual_model", "visual_model_raw"),
                   ("visual_model", "visual_model_ungated"), ("robust_rule", "ungated_rule")]
    bootstrap = []
    for left, right in comparisons:
        changes = {"daily_f1": [], "stage_macro_f1": []}
        for indices in resamples:
            scores = []
            for method in (left, right):
                scores.append(aggregate_scores([results[method]["per_case"][case_ids[index]] for index in indices]))
            changes["daily_f1"].append((scores[0]["daily"]["f1"] or 0) - (scores[1]["daily"]["f1"] or 0))
            changes["stage_macro_f1"].append((scores[0]["stage_macro_f1"] or 0) - (scores[1]["stage_macro_f1"] or 0))
        bootstrap.append({"left": left, "right": right,
                          "exploratory_percentile_95_interval": {name: np.quantile(values, [.025, .975]).tolist()
                                                                  for name, values in changes.items()}})
    save_json(target / "paired-record-bootstrap.json", {"seed": 20261007, "resamples": 2000,
                                                       "independent_units": len(case_ids),
                                                       "purpose": "Exploratory development diagnostic, not significance or external validity",
                                                       "comparisons": bootstrap})
    # Preserve the exact currently used implementation next to the audit, including prompts.
    source_paths = [ROOT / "scripts" / name for name in ("run_landslide_comparison.py", "run_landslide_visual_comparison.py",
                                                        "analyze_landslide_comparison.py")]
    source_paths += [ROOT / "src/gnss_sim" / name for name in ("landslide_baselines.py", "landslide_evaluation.py",
                                                             "landslide_visual.py", "landslide_diagnostics.py")]
    source_manifest = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        copy = target / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        source_manifest[str(relative)] = sha(path)
    save_json(target / "source-sha256.json", source_manifest)
    print(json.dumps({"label_distribution": distributions, "bootstrap": bootstrap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
