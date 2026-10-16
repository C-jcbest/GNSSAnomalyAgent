"""Finish provenance after the display-only supplement's final metadata write failed."""

from __future__ import annotations

from pathlib import Path

from close_landslide_utility_round import number, read_rows
from run_landslide_comparison import ROOT, save_json, sha

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def main():
    out = BASE / "manuscript-v20"
    analysis = BASE / "cross-seed-confirmation-v1/analysis-v1"
    metadata = out / "family-presentation-supplement.json"
    if metadata.exists():
        raise ValueError("Do not replace completed family provenance")
    rows = read_rows(analysis / "groups.csv")
    indexed = {(row["method"], row["family"]): row for row in rows}
    expected_lines = []
    for family, label in (("weak_after_strong", "强运动后弱尾"), ("weak_constant", "弱恒速"),
                          ("reactivation", "再活动"), ("asd", "A/S/D活动")):
        values = []
        for stem in ("adaptive_no_feedback_r{rep}__conflict_scope", "direct_visual_r{rep}",
                     "fixed_image_review_r{rep}__whole_target"):
            values.append("/".join(number(indexed[(stem.format(rep=rep), family)]["f1"]) for rep in (0, 1)))
        values.append(number(indexed[("xgboost_84", family)]["f1"]))
        expected_lines.append("| " + " | ".join([label, *values]) + " |")
    original_hashes, insertions = {}, []
    for original_name, current_paths in (
        ("manuscript-v20-generated-original.md", (ROOT / "docs/landslide-adaptive-review-manuscript-draft.md",
                                                 out / "manuscript-v20.md")),
        ("cross-seed-report-generated-original.md", (ROOT / "docs/landslide-cross-seed-confirmation-results.md",
                                                   out / "cross-seed-report.md")),
    ):
        original = out / original_name
        prefix, suffix = original.read_text(encoding="utf-8").split("![新种子同方法比较]", 1)
        for path in current_paths:
            current = path.read_text(encoding="utf-8")
            ending = "![新种子同方法比较]" + suffix
            if not current.startswith(prefix) or not current.endswith(ending):
                raise ValueError("Supplement changed content outside its family insertion")
            insertion = current[len(prefix):-len(ending)]
            if not all(line in insertion for line in expected_lines):
                raise ValueError("Displayed family results differ from frozen scores")
            original_hashes[str(path)] = sha(original)
            insertions.append(insertion)
    if len(set(insertions)) != 1:
        raise ValueError("Family insertions differ between manuscript and report")
    failed_source = ROOT / "scripts/supplement_landslide_cross_seed_family_results.py"
    save_json(metadata, {
        "source_groups_sha256": sha(analysis / "groups.csv"), "source_daily_sha256": sha(analysis / "daily.csv"),
        "family_count_rows_independently_checked_before_v1_failure": len(rows),
        "original_presentation_hashes": original_hashes, "family_table_checked_in_all_four_files": True,
        "only_family_insertion_changed": True, "predictions_scores_and_gates_unchanged": True,
        "failed_source": str(failed_source), "failed_source_sha256": sha(failed_source),
        "failure": "After four presentation files saved, sha(__file__) raised AttributeError: "
                   "'str' object has no attribute 'read_bytes'; no metadata written",
        "repair_source": str(Path(__file__)), "repair_sha256": sha(Path(__file__)),
        "repair": "Validate existing insertion and save provenance only; no display regeneration or inference",
    })
    print({"family_table_files_checked": 4, "metadata_repaired": True}, flush=True)


if __name__ == "__main__":
    main()
