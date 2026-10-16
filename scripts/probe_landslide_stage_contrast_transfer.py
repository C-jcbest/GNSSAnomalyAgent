"""Check a development-selected 42-day stage contrast on a second seen cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from probe_landslide_stage_raw_consistency_v2 import execute
from run_landslide_comparison import save_json, sha
from run_landslide_window_optimization import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "protocol.json", {
        "scope": "Second previously seen controlled development cohort; not independent confirmation",
        "selection_source": "stage-raw-consistency-development-v2",
        "main": "raw42 only, selected after dual-scale main produced zero interventions",
        "policy": {"separation": 42, "relative_change": 0.25, "noise_multiplier": 2.0,
                   "minimum_conflict_days": 14},
        "mechanistic_reason": "Short-scale contrast has less accumulated rate change relative to noise",
        "do_not_claim": "Post-hoc raw42 choice passed the original dual-scale hypothesis",
        "gate": {"each_repeat_removed_error_fraction_minimum": 0.75,
                 "each_repeat_correct_stage_loss_fraction_maximum": 0.05,
                 "each_repeat_error_records_minimum": 2,
                 "each_repeat_internal_macro_f1_nonnegative_change": True},
        "api_requests": 0, "further_tuning": False,
        "source_sha256": sha(Path(__file__)),
    })
    replay = args.out / "replay-v1"
    execute(args.native, replay)
    removals = read_json(replay / "removals.json")
    results = read_json(replay / "results.json")
    decisions = []
    for repeat in range(2):
        before = f"direct_visual_native_stage_r{repeat}"
        after = before + "__raw42"
        row = removals[after]["total"]
        delta = (results[after]["true_activity_internal"]["macro_f1"]
                 - results[before]["true_activity_internal"]["macro_f1"])
        passed = (row["removed_error_fraction"] is not None
                  and row["removed_error_fraction"] >= 0.75
                  and row["correct_stage_loss_fraction"] <= 0.05
                  and row["error_records"] >= 2 and delta >= 0)
        decisions.append({"method": after, "passed": passed, "internal_macro_delta": delta,
                          **row})
    decision = {"allows_visual_review": all(row["passed"] for row in decisions),
                "repeats": decisions, "interpretation": "Development transfer only"}
    save_json(args.out / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
