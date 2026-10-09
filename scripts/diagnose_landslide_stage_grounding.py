"""Read-only post-freeze diagnostics for the paired evidence-grounding experiment."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from landslide_stage_grounding import ROOT, RUN, replay

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_frozen_stage_heads import stage_pair_counts
from gnss_sim.landslide_stage_agreement import (
    disagreement_spans,
    stage_agreement,
    support_filter_effect,
)
from gnss_sim.landslide_stage_numeric_experiment import ARMS


def selected_agreement(reference, prediction, selected):
    """Select aligned dates for reporting only; never refit or alter predictions."""
    actual, guessed = [], []
    for ref, pred in zip(reference, prediction, strict=True):
        if (ref["case_id"], ref["day_index"]) in selected:
            actual.append(ref)
            guessed.append(pred)
    return stage_agreement(actual, guessed)


def agreement_state(reference, prediction):
    if prediction["feature"] == "unknown":
        return "abstained"
    if reference["feature"] == prediction["feature"]:
        return "same"
    return "different"


def main():
    destination = RUN / "diagnosis-v1.json"
    if destination.exists():
        raise FileExistsError("Preserve this diagnosis; use a new version for changes")
    public, cases, final = replay()
    reference_path = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1/compiled/daily-reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    analysis = json.loads((public / "analysis.json").read_text(encoding="utf-8"))
    if sha(reference_path) != analysis["reference_sha256"]:
        raise ValueError("Post-freeze reference differs from primary analysis")
    reference_by_case = defaultdict(list)
    for row in reference:
        reference_by_case[row["case_id"]].append(row)
    results = json.loads((public / "results.json").read_text(encoding="utf-8"))
    statuses = {item["request_id"]: item["status"] for item in results["results"]}
    native = {arm: [row for case in cases for row in case["native"][arm]] for arm in ARMS}
    per_case, interiors, accepted_answers = {}, [], {}
    common_ids = []
    activity_entries = {entry["case_id"]: entry for entry in json.loads(
        (public / "source/activity-manifest.json").read_text(encoding="utf-8"))["cases"]}
    locator_totals = {arm: Counter() for arm in ARMS}
    costs = {arm: {"calls": 0, "total_tokens": 0, "completion_tokens": 0,
                   "request_seconds": 0.0, "finish_reasons": Counter(), "statuses": Counter()}
             for arm in ARMS}
    for item in cases:
        case_id = item["case_id"]
        refs = reference_by_case[case_id]
        case_statuses = {arm: statuses[f"{case_id}-{arm}"] for arm in ARMS}
        if all(status == "success" for status in case_statuses.values()):
            common_ids.append(case_id)
        per_case[case_id] = {
            "statuses": case_statuses,
            "agreement": {arm: stage_agreement(refs, item["final"][arm]) for arm in ARMS},
            "repeat_pairs": item["pairs"],
        }
        review = json.loads((public / "source/activity-source" / activity_entries[case_id]["review"]).read_text(encoding="utf-8"))
        for activity in review["spans"]:
            if activity["state"] != "activity":
                continue
            start, stop = activity["start"], activity["stop"]
            selected = {(case_id, day) for day in range(start, stop)}
            interiors.append({"case_id": case_id, "interior": [start, stop],
                              "agreement": {arm: selected_agreement(refs, item["final"][arm], selected)
                                            for arm in ARMS}})
        for arm, audit in item["citation_audits"].items():
            if "proposed_stages" not in audit:
                locator_totals[arm]["unauditable_or_skipped_records"] += 1
                continue
            for key in ("proposed_stages", "locator_and_coverage_pass", "unknown_image_id_stages",
                        "missing_raw_citation_stages", "no_covering_raw_citation_stages",
                        "missing_motion_citation_stages"):
                locator_totals[arm][key] += audit[key]
            if case_statuses[arm] == "success":
                locator_totals[arm]["accepted_stages"] += audit["proposed_stages"]
                locator_totals[arm]["accepted_locator_and_coverage_pass"] += audit["locator_and_coverage_pass"]
        for arm, request in item["requests"].items():
            costs[arm]["statuses"][case_statuses[arm]] += 1
            if request is None:
                continue
            receipt = request["response"]
            costs[arm]["calls"] += 1
            for key in ("total_tokens", "completion_tokens"):
                costs[arm][key] += receipt.get("usage", {}).get(key, 0)
            costs[arm]["request_seconds"] += receipt["seconds"]
            costs[arm]["finish_reasons"][receipt.get("finish_reason", "unavailable")] += 1
            # These are unmodified raw answers, including the rejected one.
            if receipt["status"] == "returned_json":
                accepted_answers[f"{case_id}-{arm}"] = json.loads(receipt["output"])
    common_dates = {(row["case_id"], row["day_index"]) for row in reference
                    if row["case_id"] in common_ids}
    common_pairs = {}
    for condition in ("image", "numeric"):
        first = [row for row in final[f"{condition}-r1"] if row["case_id"] in common_ids]
        second = [row for row in final[f"{condition}-r2"] if row["case_id"] in common_ids]
        common_pairs[condition] = stage_pair_counts(first, second)
    changes = {}
    for repetition in (1, 2):
        counts = Counter()
        for ref, control, treatment in zip(reference, final[f"image-r{repetition}"],
                                          final[f"numeric-r{repetition}"], strict=True):
            if ref["stage_evaluable"]:
                counts[f"{agreement_state(ref, control)}->{agreement_state(ref, treatment)}"] += 1
        changes[f"r{repetition}"] = dict(counts)
    diagnosis = {
        "reference_use": "development_only", "formal_reference_verified": False,
        "performance_scores": None, "analysis_timing": "post_inference_freeze",
        "reference_sha256": sha(reference_path),
        "primary_analysis_sha256": sha(public / "analysis.json"),
        "script_sha256": sha(ROOT / "scripts/diagnose_landslide_stage_grounding.py"),
        "costs": costs, "total_tokens": results["total_tokens"],
        "elapsed_seconds": results["elapsed_seconds"], "per_case": per_case,
        "interiors": interiors, "raw_answers": accepted_answers,
        "native_agreement": {arm: stage_agreement(reference, native[arm]) for arm in ARMS},
        "support_filter_effect": {arm: support_filter_effect(reference, native[arm], final[arm]) for arm in ARMS},
        "evaluable_control_to_treatment_states": changes,
        "locator_totals": locator_totals,
        "locator_meaning": "mechanical tags and temporal coverage only; control was not required to cite tags",
        "citation_audits": {item["case_id"]: item["citation_audits"] for item in cases},
        "common_success_sensitivity": {
            "meaning": "post-hoc outcome-selected sensitivity; never replaces complete primary denominator",
            "case_ids": common_ids,
            "agreement": {arm: selected_agreement(reference, final[arm], common_dates) for arm in ARMS},
            "repeat_pairs": common_pairs,
        },
        "final_disagreement_spans": {arm: disagreement_spans(reference, final[arm]) for arm in ARMS},
    }
    write_json(destination, diagnosis)
    print(json.dumps({"saved": str(destination), "costs": costs, "changes": changes, "locator_totals": locator_totals,
                      "common_success_records": common_ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
