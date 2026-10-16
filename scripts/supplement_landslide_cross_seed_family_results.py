"""Make the supported scene explicit without changing frozen predictions or scores."""

from __future__ import annotations

from collections import defaultdict

from close_landslide_utility_round import number, read_rows
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def main():
    analysis = BASE / "cross-seed-confirmation-v1/analysis-v1"
    data = BASE / "cross-seed-confirmation-data-v1"
    out = BASE / "manuscript-v20"
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"]
                for cid in read_json(data / "protocol.json")["case_ids"]}
    counts = defaultdict(lambda: [0, 0, 0])
    for row in read_rows(analysis / "daily.csv"):
        actual, predicted = int(row["actual_activity"]), int(row["predicted_activity"])
        key = (row["method"], families[row["case_id"]])
        if actual == 1:
            counts[key][0 if predicted == 1 else 2] += 1
        elif actual == 0 and predicted == 1:
            counts[key][1] += 1
    groups = read_rows(analysis / "groups.csv")
    for row in groups:
        if [int(row[key]) for key in ("tp", "fp", "fn")] != counts[(row["method"], row["family"])]:
            raise ValueError("Family counts differ from the complete daily export")
    indexed = {(row["method"], row["family"]): row for row in groups}
    names = {"weak_after_strong": "强运动后弱尾", "weak_constant": "弱恒速",
             "reactivation": "再活动", "asd": "A/S/D活动"}
    lines = ["| 家族（各2记录） | 主F1两遍 | 直接F1两遍 | 固定F1两遍 | XGB84 F1 |",
             "|---|---:|---:|---:|---:|"]
    for family, label in names.items():
        values = []
        for stem in ("adaptive_no_feedback_r{rep}__conflict_scope", "direct_visual_r{rep}",
                     "fixed_image_review_r{rep}__whole_target"):
            values.append("/".join(number(indexed[(stem.format(rep=rep), family)]["f1"]) for rep in (0, 1)))
        values.append(number(indexed[("xgboost_84", family)]["f1"]))
        lines.append("| " + " | ".join([label, *values]) + " |")
    section = "\n\n" + "\n".join(lines) + """

家族分层使适用场景更清楚：新批收益主要在强运动后弱尾及再活动，弱恒速主F1两遍均低于直接，A/S/D活动家族与直接相同。强后弱仍低于XGB84，不能把汇总弱召回提高写成所有弱运动都改善。quiet对照主FP146/164，高于直接126/121；各家族仅两记录，以上为预定家族的描述性分析，不作新的选择器或显著性证据。

"""
    paths = [ROOT / "docs/landslide-adaptive-review-manuscript-draft.md", out / "manuscript-v20.md",
             ROOT / "docs/landslide-cross-seed-confirmation-results.md", out / "cross-seed-report.md"]
    if paths[0].read_bytes() != paths[1].read_bytes() or paths[2].read_bytes() != paths[3].read_bytes():
        raise ValueError("Delivered manuscript and report snapshots differ")
    original_hashes = {}
    for path in paths:
        original_hashes[str(path)] = sha(path)
        retained = out / ("manuscript-v20-generated-original.md" if path == paths[0]
                          else "cross-seed-report-generated-original.md")
        if path in (paths[0], paths[2]):
            if retained.exists():
                raise ValueError("Family supplement must not overwrite an existing revision")
            retained.write_bytes(path.read_bytes())
        text = path.read_text(encoding="utf-8")
        marker = "![新种子同方法比较]"
        if text.count(marker) != 1:
            raise ValueError("Expected one comparison-figure insertion location")
        path.write_text(text.replace(marker, section + marker, 1), encoding="utf-8")
    save_json(out / "family-presentation-supplement.json", {
        "source_groups_sha256": sha(analysis / "groups.csv"),
        "source_daily_sha256": sha(analysis / "daily.csv"),
        "family_count_rows_independently_checked": len(groups),
        "original_presentation_hashes": original_hashes,
        "purpose": "Disclose scene-specific gains and losses; all predictions, scores and gates unchanged",
        "source": str(__file__), "source_sha256": sha(__file__),
    })
    print({"family_groups_verified": len(groups), "predictions_changed": False}, flush=True)


if __name__ == "__main__":
    main()
