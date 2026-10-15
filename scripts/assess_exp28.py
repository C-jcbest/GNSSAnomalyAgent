"""Describe metric tradeoffs and separate observed decisions from provider failures."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from report_exp25 import aggregate_rows, bootstrap

from gnss_research.files import read_json, write_json


def endpoints(reports):
    aggregated = aggregate_rows(reports)
    transient = aggregated["range"]["transient"]
    return {"stations": len(reports), "positive_transient_stations": transient["affiliation"]["positive_stations"],
            "aff_f1": transient["affiliation"]["station_macro_f1"],
            "daily_f1": transient["daily_micro"]["f1"], "daily": transient["daily_micro"],
            "four_type_daily_f1": aggregated["original"]["types_daily_macro_f1"]}


def no_dependency_failure(row):
    return row["status"] == "success" and all(value != "failed" for value in row["dependency_status"].values())


def assess(report: Path, output: Path):
    with gzip.open(report / "per-station.json.gz", "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    methods = ["ce_flash_image", "ce_vl_image", "edge_flash_image", "trend_flash_image", "cf_flash_image"]
    controls = ["cf_flash_image", "ce_flash_cfonly", "ce_flash_raw", "ce_flash_text", "ce_flash_noimage",
                "ce_vl_text", "ce_vl_noimage", "vector_nearest", "sdr_flash",
                "cf_flash_text", "cf_flash_raw", "cf_flash_noimage", "segment_numeric"]
    comparisons = {}
    for method in methods:
        comparisons[method] = {}
        for control in controls:
            if control not in rows or control == method:
                continue
            first, second = rows[method], rows[control]
            shared = [(a, b) for a, b in zip(first, second) if no_dependency_failure(a) and no_dependency_failure(b)]
            changed = []
            for a, b in zip(first, second):
                if a["event_signature"] == b["event_signature"]:
                    continue
                fa, fb = a["range"]["transient"], b["range"]["transient"]
                af, bf = fa["affiliation"]["f1"], fb["affiliation"]["f1"]
                changed.append({"station": a["id"], "cohort_seed": a["cohort_seed"], "scenario": a["scenario"],
                    "both_decisions_succeeded": no_dependency_failure(a) and no_dependency_failure(b),
                    "aff_delta": af - bf if af is not None and bf is not None else None,
                    "daily_f1_delta": fa["daily"]["f1"] - fb["daily"]["f1"],
                    **{key + "_delta": fa["daily"][key] - fb["daily"][key] for key in ("tp", "fp", "fn")},
                    "main_events": a["event_signature"], "control_events": b["event_signature"]})
            comparisons[method][control] = {
                "all_stations": {"main": endpoints(first), "control": endpoints(second)},
                "both_decisions_successful_only": {"main": endpoints([a for a, _ in shared]),
                    "control": endpoints([b for _, b in shared]),
                    "interpretation": "Conditional diagnostic subset; never replaces the all-station result"} if shared else None,
                "changed_stations": len(changed), "changes": changed,
                "aff_wins": sum(c["aff_delta"] is not None and c["aff_delta"] > 0 for c in changed),
                "aff_losses": sum(c["aff_delta"] is not None and c["aff_delta"] < 0 for c in changed)}
    summary = read_json(report / "summary.json")
    main, base, ablated = (summary["methods"][m]["range"]["transient"] for m in
                          ("ce_flash_image", "cf_flash_image", "ce_flash_cfonly"))
    aff_delta = main["affiliation"]["station_macro_f1"] - base["affiliation"]["station_macro_f1"]
    day_delta = main["daily_micro"]["f1"] - base["daily_micro"]["f1"]
    image_delta = main["affiliation"]["station_macro_f1"] - ablated["affiliation"]["station_macro_f1"]
    image_day_delta = main["daily_micro"]["f1"] - ablated["daily_micro"]["f1"]
    supported_gain = any(c["both_decisions_succeeded"] and
                         ((c["aff_delta"] is not None and c["aff_delta"] > 0)
                          or c["tp_delta"] > 0 or (c["fp_delta"] < 0 and c["fn_delta"] <= 0))
                         for c in comparisons["ce_flash_image"]["ce_flash_cfonly"]["changes"])
    decision = {"split": summary["split"], "stations": summary["stations"],
        "main_method": "ce_flash_image", "aff_delta_vs_cfv": aff_delta, "daily_delta_vs_cfv": day_delta,
        "aff_delta_vs_same_prompt_cfonly": image_delta, "daily_delta_vs_same_prompt_cfonly": image_day_delta,
        "has_successful_range_metric_gain_case_vs_cfonly": supported_gain,
        "passes_declared_metric_conditions": max(aff_delta, day_delta) >= .005
            and max(image_delta, image_day_delta) > 0 and supported_gain,
        "scope": "Adoption only considered on independent test; all tradeoffs and textual controls still required"}
    # The original CFV is a fixed comparator, not a replacement EXP28 primary method.
    cfv_uncertainty = {
        control: bootstrap(rows["cf_flash_image"], rows[control])
        for control in ("cf_flash_text", "cf_flash_raw", "vector_nearest", "sdr_flash")
        if control in rows
    }
    secondary_uncertainty = {}
    for method, matched_controls in (
        ("ce_vl_image", ("ce_vl_text", "ce_vl_noimage", "vector_nearest", "sdr_flash")),
        ("edge_flash_image", ("cf_flash_image", "cf_flash_text", "vector_nearest")),
    ):
        secondary_uncertainty[method] = {
            control: bootstrap(rows[method], rows[control])
            for control in matched_controls
        }
    write_json(output, {"decision": decision, "comparisons": comparisons,
                        "cfv_descriptive_bootstrap": cfv_uncertainty,
                        "secondary_descriptive_bootstrap": secondary_uncertainty,
                        "secondary_scope": "Post-scoring descriptive analysis; no method changes or primary replacement"})
    print(decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assess(args.report, args.out)
