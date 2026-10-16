"""Independently check exported counts, decisions, costs and manuscript lineage."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from close_landslide_budget_confirmation import read_rows
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json


def check_counts(row, counts):
    tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
    for name, value in (("tp", tp), ("fp", fp), ("fn", fn)):
        if int(row[name]) != value:
            raise ValueError(f"Exported {name} differs from daily decisions")
    expected = {"precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "coverage": 1 - counts["unknown"] / counts["known"]}
    for name, value in expected.items():
        if value is None:
            if row[name] != "":
                raise ValueError("Undefined metric acquired a numeric value")
        elif not math.isclose(float(row[name]), value, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Exported {name} differs from independent arithmetic")


def verify_delivery(run, manuscript):
    verify_freeze(run)
    data = ROOT / read_json(run / "protocol.json")["source_bundle"]
    directories = [data, run / "scope-v1", run / "analysis-v1", run / "supplement-v1",
                   run / "failure-sensitivity-v1", run / "figures-v1", manuscript,
                   run.parent / "budget-matched-selection-development-v2"]
    verified_files = sum(verify_manifest(path) for path in directories)
    for name, expected in read_json(data / "pre-generation-freeze.json").items():
        if sha(data / name) != expected:
            raise ValueError("Pre-generation registration changed")
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError("Registered executing method or scorer changed")
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"]
                for cid in read_json(run / "protocol.json")["case_ids"]}
    counts = defaultdict(Counter)
    family_counts = defaultdict(Counter)
    daily_rows = 0
    with (run / "analysis-v1/daily.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            actual, predicted = int(row["actual_activity"]), int(row["predicted_activity"])
            daily_rows += 1
            if actual < 0:
                continue
            keys = [counts[row["method"]], family_counts[(row["method"], families[row["case_id"]])]]
            for group in keys:
                group["known"] += 1
                group["unknown"] += predicted == -1
                group["tp"] += actual == 1 and predicted == 1
                group["fp"] += actual == 0 and predicted == 1
                group["fn"] += actual == 1 and predicted != 1
    rows = read_rows(run / "analysis-v1/summary.csv")
    indexed = {row["method"]: row for row in rows}
    for row in rows:
        check_counts(row, counts[row["method"]])
    family_rows = read_rows(run / "supplement-v1/family-results.csv")
    for row in family_rows:
        check_counts(row, family_counts[(row["method"], row["family"])])
    if len(rows) != 37 or len(family_rows) != 222 or daily_rows != 324120:
        raise ValueError("Unified comparison population is incomplete")
    decisions = read_json(run / "analysis-v1/decision.json")
    corrections = read_rows(run / "supplement-v1/corrections.csv")
    for comparison in decisions["comparisons"]:
        repeat, variant = comparison["repeat"], comparison["variant"]
        main = f"adaptive_no_feedback_r{repeat}__{variant}"
        control = f"same_count_fixed_early_r{repeat}__{variant}"
        net = sum(int(row["net_correct"]) for row in corrections
                  if row["method"] == main and row["control"] == control)
        expected = {"f1_greater_than_early": float(indexed[main]["f1"]) > float(indexed[control]["f1"]),
                    "weak_recall_nonloss_than_early": float(indexed[main]["weak_recall"]) >= float(indexed[control]["weak_recall"]),
                    "net_definite_correction_positive_than_early": net > 0}
        if comparison["gate"] != expected or comparison["net_vs_early"] != net:
            raise ValueError("Pre-registered position decision differs from exported evidence")
    if not decisions["selection_confirmed"] or decisions["scope_additional_benefit_retested"]:
        raise ValueError("Manuscript claims differ from trial decision")
    results = read_json(run / "analysis-v1/results.json")
    event_rows = read_rows(run / "supplement-v1/events.csv")
    for row in event_rows:
        key = "boundaries_iou03" if float(row["iou"]) == .3 else "boundaries_iou05"
        detail = results[row["method"]][key][row["case_id"]]
        for field, expected in (("matched", len(detail["matched_pairs"])),
                                ("unmatched_predictions", detail["unmatched_prediction_events"]),
                                ("missed", detail["missed_events"])):
            if int(row[field]) != expected:
                raise ValueError("Exported event row differs from matching evidence")
    ledger = read_json(manuscript / "run-ledger-end.json")
    costs = read_json(run / "supplement-v1/actual-costs.json")
    previous = read_json(run.parent / "manuscript-v21/run-ledger-end.json")
    for total, addition in (("cumulative_requests_started", "started"),
                            ("cumulative_local_receipts", "receipts"),
                            ("cumulative_reported_tokens_lower_bound", "reported_tokens")):
        if ledger[total] != previous[total] + costs[addition]:
            raise ValueError("Cumulative usage arithmetic differs")
    if ledger != read_json(run / "run-ledger-end.json") or len(costs["failures"]) != 2:
        raise ValueError("Failure or ledger disclosure differs")
    doc = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    report = ROOT / "docs/landslide-budget-selection-results.md"
    if sha(doc) != sha(manuscript / "manuscript-v22.md") or sha(report) != sha(manuscript / "budget-selection-report.md"):
        raise ValueError("Current paper/report no longer matches versioned copies")
    if sha(manuscript / "manuscript-v21-original.md") != sha(run.parent / "manuscript-v21/manuscript-v21.md"):
        raise ValueError("Original v21 bytes were not preserved")
    text = doc.read_text(encoding="utf-8")
    citations = [int(value) for value in re.findall(r"^\[(\d+)\] ", text, re.MULTILINE)]
    if citations != list(range(1, 13)):
        raise ValueError("Reference numbering changed unexpectedly")
    checks = {"verified_frozen_files": verified_files, "daily_rows": daily_rows,
              "daily_activity_groups_recomputed": len(rows), "family_groups_recomputed": len(family_rows),
              "event_rows_checked": len(event_rows), "registration_comparisons_checked": len(decisions["comparisons"]),
              "references": len(citations), "v21_exact_bytes": True, "cumulative_usage_checked": True,
              "all_failures_retained": True, "goal_status": "active", "new_api": 0}
    save_json(manuscript / "final-delivery-checks.json", checks)
    sources = ["scripts/prepare_landslide_budget_confirmation_v2.py", "scripts/probe_landslide_budget_matched_selection_v2.py",
               "scripts/analyze_landslide_budget_confirmation.py", "scripts/render_landslide_budget_confirmation.py",
               "scripts/summarize_landslide_budget_confirmation.py", "scripts/diagnose_landslide_budget_failures.py",
               "scripts/close_landslide_budget_confirmation.py", "scripts/verify_landslide_budget_delivery.py",
               "docs/landslide-budget-selection-results.md", "docs/landslide-adaptive-review-manuscript-draft.md",
               "docs/landslide-adaptive-review-method.md", "docs/project-status.md",
               "docs/landslide-research-continuation-prompt.md", "docs/landslide-submission-roadmap.md"]
    save_json(manuscript / "delivery-manifest.json", {
        "workspace_files": {name: sha(ROOT / name) for name in sources},
        "supplement_files": {"manifest.json": sha(manuscript / "manifest.json"),
                             "final-delivery-checks.json": sha(manuscript / "final-delivery-checks.json")},
        "trial_manifests": {str(path): sha(path / "manifest.json") for path in directories if path != manuscript}})
    print(checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    arguments = parser.parse_args()
    verify_delivery(arguments.run, arguments.manuscript)
