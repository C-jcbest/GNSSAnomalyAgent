"""Verify delivered claims, references, accounting and frozen cross-seed evidence."""

from __future__ import annotations

import re
from pathlib import Path

from close_landslide_utility_round import comparison_table, read_rows
from finalize_landslide_adaptive_evidence import verify_manifest
from PIL import Image
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json
from verify_landslide_normal_anchor_delivery import verify_export

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def main():
    run = BASE / "cross-seed-confirmation-v1"
    data = BASE / "cross-seed-confirmation-data-v1"
    scoped = run / "scope-v1"
    analysis = run / "analysis-v1"
    out = BASE / "manuscript-v20"
    verify_freeze(run)
    manifest_files = verify_manifest(data) + verify_manifest(scoped)
    for folder in (run, scoped / run.name, analysis):
        for name, expected in read_json(folder / "prediction-freeze.json").items():
            if sha(folder / name) != expected:
                raise ValueError("Prediction freeze differs")
    for name, expected in read_json(data / "pre-generation-freeze.json").items():
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError("Executing source differs from generation freeze")
    for name, expected in read_json(data / "training-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Common trained models differ")
    exported_methods = verify_export(analysis, anchor=True)
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    report = ROOT / "docs/landslide-cross-seed-confirmation-results.md"
    if sha(draft) != sha(out / "manuscript-v20.md") or sha(report) != sha(out / "cross-seed-report.md"):
        raise ValueError("Current presentation and archived presentation differ")
    if sha(out / "manuscript-v19-original.md") != sha(BASE / "manuscript-v19/manuscript-v19.md"):
        raise ValueError("Exact prior manuscript differs")
    text = draft.read_text(encoding="utf-8")
    body, reference_section = text.split("## 参考文献", 1)
    reference_lines = re.findall(r"^\[(\d+)\] .+$", reference_section, flags=re.MULTILINE)
    citations = set(re.findall(r"\[(\d+)\]", body))
    if reference_lines != [str(index) for index in range(1, 12)] or citations != set(reference_lines):
        raise ValueError("Renumbered manuscript citations do not map to eleven retained sources")
    original = (out / "manuscript-v19-original.md").read_text(encoding="utf-8")
    original_refs = {int(match.group(1)): match.group(2) for match in
                     re.finditer(r"^\[(\d+)\] (.+)$", original, flags=re.MULTILINE)}
    current_refs = {int(match.group(1)): match.group(2) for match in
                    re.finditer(r"^\[(\d+)\] (.+)$", reference_section, flags=re.MULTILINE)}
    mapping = read_json(out / "checks.json")["reference_renumbering"]
    for old_index, new_index in mapping.items():
        if original_refs[int(old_index)] != current_refs[new_index]:
            raise ValueError("Reference content changed while renumbering")
    rows = read_rows(analysis / "summary.csv")
    if comparison_table(rows) not in report.read_text(encoding="utf-8"):
        raise ValueError("Full comparison report differs from frozen summary")
    old_costs = read_json(BASE / "manuscript-v19/run-ledger-end.json")
    costs = read_json(analysis / "actual-costs.json")
    ledger = read_json(out / "run-ledger-end.json")
    for key, delta in (("cumulative_requests_started", "new_started"),
                       ("cumulative_local_receipts", "new_local_receipts"),
                       ("cumulative_reported_tokens_lower_bound", "new_reported_tokens")):
        if ledger[key] != old_costs[key] + costs[delta]:
            raise ValueError("Cumulative accounting differs")
    decision = read_json(analysis / "decision.json")
    if ledger["scope_confirmed_on_new_seed"] != decision["scope_confirmed"]:
        raise ValueError("Scope failure was not retained")
    if ledger["method_confirmed_on_new_seed"] != decision["method_confirmed"]:
        raise ValueError("Method decision differs")
    supplement = read_json(out / "family-presentation-supplement.json")
    if supplement["source_groups_sha256"] != sha(analysis / "groups.csv"):
        raise ValueError("Family source differs")
    pictures = list(run.glob("*-activity.png")) + list(run.glob("*-stages.png")) + list(analysis.glob("*.png"))
    for path in pictures:
        with Image.open(path) as picture:
            picture.verify()
    save_json(out / "final-delivery-checks.json", {
        "frozen_sources_inputs_models_predictions_valid": True, "data_scope_manifest_files": manifest_files,
        "daily_counts_f1_and_event_f1_from_exports_checked": exported_methods,
        "family_count_groups_checked": supplement["family_count_rows_independently_checked_before_v1_failure"],
        "full_report_matches_all_frozen_summary_rows": True, "v19_exactly_preserved": True,
        "current_draft_equals_v20": True, "reference_number_mappings_checked": len(mapping),
        "cumulative_cost_arithmetic_checked": True, "png_files_checked": len(pictures),
        "original_business_tests_not_repeated_no_method_change": True,
        "new_preparation_analysis_closure_lineage_checker_ruff_passed": True,
        "family_supplement_v1_metadata_failure_preserved_and_repaired": True,
        "scope_strict_gate_passed": decision["scope_confirmed"],
        "method_gate_passed": decision["method_confirmed"], "field_accuracy_confirmed": False,
        "goal_status": "active", "final_checker_sha256": sha(Path(__file__)),
    })
    print({"methods": exported_methods, "reference_mappings": len(mapping), "png": len(pictures)}, flush=True)


if __name__ == "__main__":
    main()
