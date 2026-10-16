"""Verify final claims and corrected lineage without repeating model requests."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_coordinate_review import verify
from run_landslide_window_optimization import read_json

from gnss_sim.landslide_evaluation import aggregate_scores


def verify_delivery(run, manuscript):
    verify(run)
    analysis = run / "analysis-v1"
    verified_files = verify_manifest(analysis) + verify_manifest(manuscript)
    rendering = read_json(run / "rendering.json")
    image_pairs = 0
    for name, raw in rendering.items():
        if raw["sign"] != 1:
            continue
        reflected = rendering[name.replace("-raw-", "-reflected-")]
        expected = -np.asarray(raw["axis_limits"])[:, ::-1]
        np.testing.assert_allclose(reflected["axis_limits"], expected, rtol=0, atol=1e-12)
        if (raw["start"], raw["stop"]) != (reflected["start"], reflected["stop"]):
            raise ValueError("Reflection changed the original observation extent")
        image_pairs += 1
    correction = read_json(manuscript / "protocol-correction.json")
    if correction["actual_population"] != {"quiet": 2, "weak_constant": 2, "reactivation": 2}:
        raise ValueError("Correction does not disclose the actual three-family population")
    if correction["invalid_gate"] not in read_json(analysis / "decision.json")["repeats"][0]["gate"]:
        raise ValueError("Design correction does not identify the original unreachable gate")
    results = read_json(analysis / "results.json")
    with (manuscript / "family-results.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    for group in groups:
        method, family = group["method"], group["family"]
        selected = [value for cid, value in results[method]["per_case"].items()
                    if correction["actual_families"][cid] == family]
        aggregate = aggregate_scores(selected)
        for field, measured in aggregate["daily"].items():
            actual = group[field]
            if measured is None:
                if actual != "":
                    raise ValueError("Undefined family metric was replaced by a number")
            elif not np.isclose(float(actual), measured, rtol=0, atol=1e-12):
                raise ValueError("Family table differs from the original daily scoring")
    costs = read_json(analysis / "actual-costs.json")
    previous = read_json(ROOT / "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v20/run-ledger-end.json")
    ledger = read_json(manuscript / "run-ledger-end.json")
    if ledger != read_json(run / "run-ledger-end.json"):
        raise ValueError("Manuscript and experiment ledgers differ")
    for target, before, added in (
        ("cumulative_requests_started", "cumulative_requests_started", "started"),
        ("cumulative_local_receipts", "cumulative_local_receipts", "receipts"),
        ("cumulative_reported_tokens_lower_bound", "cumulative_reported_tokens_lower_bound", "reported_tokens"),
    ):
        if ledger[target] != previous[before] + costs[added]:
            raise ValueError("Cumulative usage arithmetic differs")
    current = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    report = ROOT / "docs/landslide-coordinate-review-results.md"
    if sha(current) != sha(manuscript / "manuscript-v21.md") or sha(report) != sha(manuscript / "coordinate-report.md"):
        raise ValueError("Current manuscript/report differ from their versioned snapshots")
    old = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v20/manuscript-v20.md"
    if sha(old) != sha(manuscript / "manuscript-v20-original.md"):
        raise ValueError("The complete pre-round draft was not preserved exactly")
    text = current.read_text(encoding="utf-8")
    citations = [int(value) for value in re.findall(r"^\[(\d+)\] ", text, re.MULTILINE)]
    if citations != list(range(1, 13)) or "[12]" not in text.split("## 参考文献")[0]:
        raise ValueError("New citation was not connected to a unique reference")
    sources = {"src/gnss_sim/landslide_coordinate_review.py", "scripts/run_landslide_coordinate_review.py",
               "scripts/analyze_landslide_coordinate_review.py", "scripts/close_landslide_coordinate_review.py",
               "scripts/verify_landslide_coordinate_delivery.py", "tests/test_landslide_coordinate_review.py"}
    paths = sources | {"docs/landslide-coordinate-review-results.md", "docs/landslide-adaptive-review-manuscript-draft.md",
                       "docs/landslide-adaptive-review-method.md", "docs/landslide-research-plan.md",
                       "docs/project-status.md", "docs/landslide-research-continuation-prompt.md",
                       "docs/landslide-submission-roadmap.md"}
    checks = {"original_manifest_files": verified_files, "reflected_image_pairs": image_pairs,
              "family_rows_recomputed": len(groups), "references": len(citations),
              "v20_exact_bytes": True, "frozen_design_error_disclosed": True,
              "cumulative_cost_checked": True, "goal_status": "active"}
    save_json(manuscript / "final-delivery-checks.json", checks)
    save_json(manuscript / "delivery-manifest.json", {
        "workspace_files": {name: sha(ROOT / name) for name in sorted(paths)},
        "supplement_files": {name: sha(manuscript / name)
                             for name in ("manifest.json", "final-delivery-checks.json", "protocol-correction.json")},
    })
    print(checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manuscript", required=True, type=Path)
    args = parser.parse_args()
    verify_delivery(args.run, args.manuscript)
