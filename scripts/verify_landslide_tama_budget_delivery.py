"""Check all 41 daily groups, nearest-design costs and exact manuscript lineage."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from close_landslide_budget_confirmation import read_rows
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_tama_adapter import verify as verify_tama
from run_landslide_window_optimization import read_json
from verify_landslide_budget_delivery import check_counts


def verify_delivery(run, manuscript):
    verify_tama(run)
    base = run.parent
    registration = base / "tama-budget-comparison-registration-v3"
    folders = [registration, run / "analysis-v2", run / "analysis-v3", run / "figures-v1", manuscript,
               base / "literature-v1/tama-budget-codecheck-v1"]
    verified_files = sum(verify_manifest(folder) for folder in folders)
    for path, expected in read_json(registration / "bindings.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Fixed registration no longer survives current-manuscript editing")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Frozen adapter prediction changed")
    analysis = run / "analysis-v3"
    families = {cid: read_json(base / "budget-selection-confirmation-data-v2/mechanism-reference" / cid / "truth.json")["family"]
                for cid in read_json(run / "protocol.json")["case_ids"]}
    references = {(row["case_id"], int(row["day_index"])): row for row in read_rows(run / "reference-subset.csv")}
    counts, family_counts = defaultdict(Counter), defaultdict(Counter)
    total_rows = 0
    with (analysis / "daily.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            reference = references[(row["case_id"], int(row["day_index"]))]
            actual, predicted = int(row["actual_activity"]), int(row["predicted_activity"])
            stage = int(row["predicted_stage"])
            if actual != int(reference["activity_label"]) or row["actual_stage"] != reference["feature"]:
                raise ValueError("Exported evaluation reference changed")
            if stage > 0 and predicted != 1:
                raise ValueError("Stage acquired a label outside activity")
            if actual < 0:
                if predicted != -1 or stage != -1:
                    raise ValueError("Missing day received a definite prediction")
                continue
            for group in (counts[row["method"]], family_counts[(row["method"], families[row["case_id"]])]):
                group["known"] += 1
                group["unknown"] += predicted == -1
                group["tp"] += actual == 1 and predicted == 1
                group["fp"] += actual == 0 and predicted == 1
                group["fn"] += actual == 1 and predicted != 1
                group["weak"] += int(reference["weak_signal"]) == 1
                group["weak_detected"] += int(reference["weak_signal"]) == 1 and predicted == 1
    rows = read_rows(analysis / "summary.csv")
    family_rows = read_rows(analysis / "family-results.csv")
    for row in rows + family_rows:
        group = family_counts[(row["method"], row["family"])] if "family" in row else counts[row["method"]]
        check_counts(row, group)
        if group["weak"]:
            if not math.isclose(float(row["weak_recall"]), group["weak_detected"] / group["weak"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Weak-motion recall differs from daily reference")
        elif row["weak_recall"] != "":
            raise ValueError("Undefined weak recall acquired a number")
    if len(rows) != 41 or len(family_rows) != 246 or total_rows != 359160:
        raise ValueError("Unified comparison population is incomplete")
    original = base / "budget-selection-confirmation-v1/analysis-v1"
    predictions = read_json(analysis / "predictions.json")
    results = read_json(analysis / "results.json")
    for method, expected in read_json(original / "predictions.json").items():
        if predictions[method] != expected or results[method] != read_json(original / "results.json")[method]:
            raise ValueError("Adding the nearest design changed an original arm")
    for name in ("predictions.json", "results.json", "summary.csv", "daily.csv", "family-results.csv", "corrections.csv"):
        if sha(analysis / name) != sha(run / "analysis-v2" / name):
            raise ValueError("Document-owner repair changed a score")
    costs = read_json(analysis / "actual-costs.json")
    ledger = read_json(manuscript / "run-ledger-end.json")
    previous = read_json(base / "manuscript-v22/run-ledger-end.json")
    for total, addition in (("cumulative_requests_started", "started"), ("cumulative_local_receipts", "receipts"),
                            ("cumulative_reported_tokens_lower_bound", "reported_tokens")):
        if ledger[total] != previous[total] + costs[addition]:
            raise ValueError("Cumulative usage arithmetic differs")
    if ledger != read_json(run / "run-ledger-end.json"):
        raise ValueError("Experiment and paper ledgers differ")
    doc = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    report = ROOT / "docs/landslide-tama-budget-comparison-results.md"
    if sha(doc) != sha(manuscript / "manuscript-v23.md") or sha(report) != sha(manuscript / "tama-budget-report.md"):
        raise ValueError("Current document differs from preserved version")
    if sha(manuscript / "manuscript-v22-original.md") != sha(base / "manuscript-v22/manuscript-v22.md"):
        raise ValueError("Complete original v22 bytes were not preserved")
    indexed = {row["method"]: row for row in rows}
    receipts = [read_json(path) for path in (run / "requests").glob("*.response.json")]
    for repeat in (0, 1):
        normal = [item for item in receipts if item["request_id"] == f"normal-pattern-r{repeat}"]
        detection = [item for item in receipts if item["request_id"].endswith(f"-detect-r{repeat}")]
        review = [item for item in receipts if item["request_id"].endswith(f"-review-r{repeat}")]
        for stem, used in (("no_review", normal + detection), ("positive_review", normal + detection + review)):
            row = indexed[f"tama_gnss_{stem}_r{repeat}"]
            if int(row["deployment_requests"]) != len(used) or int(row["reported_tokens"]) != sum(item["usage"]["total_tokens"] for item in used):
                raise ValueError("Normal-reference or deployment cost was omitted")
    checks = {"verified_frozen_files": verified_files, "daily_rows": total_rows, "activity_groups_recomputed": len(rows),
        "family_groups_recomputed": len(family_rows), "weak_recall_checked": True, "normal_reference_costs_checked": True,
        "all_original_37_predictions_scores_unchanged": True, "v2_v3_predictions_scores_identical": True,
        "registration_survives_current_draft_update": True, "v22_exact_bytes": True, "cost_arithmetic_checked": True,
        "new_records": 0, "new_api": 0, "goal_status": "active"}
    save_json(manuscript / "final-delivery-checks.json", checks)
    paths = ["scripts/prepare_landslide_tama_budget_comparison.py", "scripts/analyze_landslide_tama_budget_comparison_v2.py",
        "scripts/render_landslide_tama_budget_comparison.py", "scripts/close_landslide_tama_budget_comparison.py",
        "scripts/verify_landslide_tama_budget_delivery.py", "docs/landslide-tama-budget-comparison-results.md",
        "docs/landslide-adaptive-review-manuscript-draft.md", "docs/project-status.md", "docs/landslide-research-continuation-prompt.md",
        "docs/landslide-submission-roadmap.md", "docs/landslide-adaptive-review-method.md", "docs/landslide-nearest-method-code-review.md"]
    save_json(manuscript / "delivery-manifest.json", {"workspace_files": {name: sha(ROOT / name) for name in paths},
        "trial_manifests": {str(folder): sha(folder / "manifest.json") for folder in folders if folder != manuscript},
        "supplement_files": {name: sha(manuscript / name) for name in ("manifest.json", "final-delivery-checks.json")}})
    print(checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    arguments = parser.parse_args()
    verify_delivery(arguments.run, arguments.manuscript)
