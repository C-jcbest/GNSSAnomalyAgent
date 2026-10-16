"""Report an explicitly post-hoc sensitivity to failed global answers, without replacing full scores."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import save_json, sha
from run_landslide_window_optimization import read_json
from summarize_landslide_missed_motion_repair import write_rows

from gnss_sim.landslide_evaluation import aggregate_scores


def diagnose(run, out):
    verify_manifest(run / "analysis-v1")
    verify_manifest(run / "supplement-v1")
    failures = read_json(run / "failures.json")
    ids = read_json(run / "protocol.json")["case_ids"]
    results = read_json(run / "analysis-v1/results.json")
    with (run / "supplement-v1/corrections.csv").open(encoding="utf-8", newline="") as handle:
        corrections = list(csv.DictReader(handle))
    rows, comparisons = [], []
    for repeat in (0, 1):
        retained = [cid for cid in ids if f"{cid}-direct-r{repeat}" not in failures]
        if len(retained) == len(ids):
            raise ValueError("This diagnostic expects a registered global-answer failure in each repeat")
        methods = [f"direct_visual_r{repeat}"]
        methods.extend(f"{stem}_r{repeat}__{variant}" for stem in
                       ("adaptive_no_feedback", "fixed_image_review", "same_count_fixed_early", "same_count_fixed_late")
                       for variant in ("whole_target", "conflict_scope"))
        numerical = [name for name in results if name.startswith(("xgboost", "robust_", "pelt_", "raw_"))]
        if len(numerical) != 9:
            raise ValueError("Sensitivity requires the same nine numerical controls")
        methods = list(dict.fromkeys(methods + numerical))
        aggregates = {}
        for method in methods:
            aggregate = aggregate_scores([results[method]["per_case"][cid] for cid in retained])
            aggregates[method] = aggregate
            rows.append({"repeat": repeat, "method": method, "retained_records": len(retained),
                         **aggregate["daily"], "weak_recall": aggregate["weak_recall"],
                         "event_f1_iou03": aggregate["events"]["f1"], "coverage": aggregate["coverage"]})
        for variant in ("whole_target", "conflict_scope"):
            main = f"adaptive_no_feedback_r{repeat}__{variant}"
            control = f"same_count_fixed_early_r{repeat}__{variant}"
            net = sum(int(row["net_correct"]) for row in corrections
                      if row["method"] == main and row["control"] == control and row["case_id"] in retained)
            comparisons.append({"repeat": repeat, "variant": variant,
                "excluded_records": [cid for cid in ids if cid not in retained],
                "f1_difference": aggregates[main]["daily"]["f1"] - aggregates[control]["daily"]["f1"],
                "weak_recall_difference": aggregates[main]["weak_recall"] - aggregates[control]["weak_recall"],
                "net_correct_difference": net})
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    write_rows(out / "summary.csv", rows)
    save_json(out / "diagnostic.json", {"post_hoc": True, "primary_population_unchanged": True,
        "purpose": "Check whether matched-position gains are solely explained by malformed initial outputs",
        "selection_confirmed_uses_full_population_only": True, "comparisons": comparisons,
        "new_api": 0, "no_significance_claim": True})
    save_json(out / "manifest.json", {"files": {path.name: sha(path) for path in out.iterdir() if path.is_file()}})
    print(comparisons, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    diagnose(arguments.run, arguments.out)
