"""Cross-check the stopped probes from their exports and immutable bindings."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from close_landslide_utility_round import read_rows
from PIL import Image
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_normal_contrast import BASE, verify
from run_landslide_window_optimization import read_json


def verify_export(folder, anchor=False):
    counts = defaultdict(lambda: [0, 0, 0])
    for row in read_rows(folder / "daily.csv"):
        actual, predicted = int(row["actual_activity"]), int(row["predicted_activity"])
        if actual == 1:
            counts[row["method"]][0 if predicted == 1 else 2] += 1
        elif actual == 0 and predicted == 1:
            counts[row["method"]][1] += 1
    events = defaultdict(lambda: [0, 0, 0])
    event_keys = ("matched", "unmatched_predictions", "missed") if anchor else (
        "matched_events", "unmatched_prediction_events", "missed_events")
    for row in read_rows(folder / "events.csv"):
        if float(row["iou"]) == .5:
            for index, key in enumerate(event_keys):
                events[row["method"]][index] += int(row[key])
    summaries = read_rows(folder / "summary.csv")
    for row in summaries:
        tp, fp, fn = counts[row["method"]]
        if [int(row[key]) for key in ("tp", "fp", "fn")] != [tp, fp, fn]:
            raise ValueError("Export daily counts differ")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        if abs(f1 - float(row["f1"])) > 1e-12:
            raise ValueError("Export daily F1 differs")
        tp, fp, fn = events[row["method"]]
        event_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        if abs(event_f1 - float(row["event_f1_iou05"])) > 1e-12:
            raise ValueError("Export event F1 differs")
    return len(summaries)


def main():
    normal = BASE / "normal-contrast-development-v1"
    prepared = BASE / "numerical-anchor-development-v1"
    anchor = prepared / "cache-screen-v1"
    out = BASE / "manuscript-v19"
    verify(normal)
    for folder in (normal, anchor):
        for name, expected in read_json(folder / "prediction-freeze.json").items():
            if sha(folder / name) != expected:
                raise ValueError("Frozen prediction or screening source differs")
    for name, expected in read_json(prepared / "bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Numerical preflight source differs")
    daily_normal = verify_export(normal / "analysis-v1")
    daily_anchor = verify_export(anchor, anchor=True)
    old = read_json(BASE / "manuscript-v18/run-ledger-end.json")
    costs = read_json(normal / "analysis-v1/actual-costs.json")
    ledger = read_json(out / "run-ledger-end.json")
    for current, previous, delta in (
        ("cumulative_requests_started", "cumulative_requests_started", "new_started"),
        ("cumulative_local_receipts", "cumulative_local_receipts", "new_local_receipts"),
        ("cumulative_reported_tokens_lower_bound", "cumulative_reported_tokens_lower_bound", "new_reported_tokens"),
    ):
        if ledger[current] != old[previous] + costs[delta]:
            raise ValueError("Cumulative costs differ")
    if sha(out / "manuscript-v18-original.md") != sha(BASE / "manuscript-v18/manuscript-v18.md"):
        raise ValueError("Exact prior draft was not retained")
    if sha(ROOT / "docs/landslide-adaptive-review-manuscript-draft.md") != sha(out / "manuscript-v19.md"):
        raise ValueError("Current draft differs from archive")
    source = (out / "source.py").read_text(encoding="utf-8")
    corrected = ROOT / "scripts/close_landslide_normal_anchor_v2.py"
    unused_line = '    changes = read_json(normal / "analysis-v1/changes.json")\n'
    if source.replace(unused_line, "") != corrected.read_text(encoding="utf-8"):
        raise ValueError("Closure lint repair has unrelated differences")
    images = list((normal / "analysis-v1").glob("*.png")) + list(out.glob("*.png"))
    for path in images:
        with Image.open(path) as picture:
            picture.verify()
    save_json(out / "closure-lint-lineage.json", {
        "executed_source_retained": sha(out / "source.py"),
        "executed_v1_ruff": "F841 unused local; closure ran successfully before lint repair",
        "corrected_source": str(corrected), "corrected_sha256": sha(corrected),
        "repair": "Remove one unused read only; no prediction/report regeneration or frozen-source edit",
        "v2_ruff": "passed", "v2_executed": False,
    })
    save_json(out / "final-delivery-checks.json", {
        "source_and_prediction_bindings_valid": True,
        "normal_daily_and_event_exports_checked": daily_normal,
        "anchor_daily_and_event_exports_checked": daily_anchor,
        "v18_exactly_preserved": True, "current_draft_equals_v19": True,
        "cumulative_cost_arithmetic_checked": True, "png_files_checked": len(images),
        "normal_business_tests_previously_passed": 17, "tests_not_repeated": True,
        "executed_closure_v1_ruff_failed": True, "corrected_closure_v2_ruff_passed": True,
        "anchor_repeat1_unexecuted_unscored": True, "goal_status": "active",
    })
    print({"normal_methods": daily_normal, "anchor_methods": daily_anchor, "png": len(images)}, flush=True)


if __name__ == "__main__":
    main()
