"""Independent subgroup arithmetic and locator coverage checks after freezing."""

from __future__ import annotations

import json
import re

from landslide_stage_grounding import ROOT, RUN
from verify_landslide_stage_turning_diagnosis import check_metrics

from gnss_sim.artifacts import sha, write_json

ARMS = ("image-r1", "numeric-r1", "image-r2", "numeric-r2")


def main():
    public = RUN / "public"
    diagnosis = json.loads((RUN / "diagnosis-v1.json").read_text(encoding="utf-8"))
    analysis = json.loads((public / "analysis.json").read_text(encoding="utf-8"))
    reference_path = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1/compiled/daily-reference.json"
    if sha(reference_path) != diagnosis["reference_sha256"]:
        raise ValueError("Reference changed")
    if sha(public / "analysis.json") != diagnosis["primary_analysis_sha256"]:
        raise ValueError("Primary analysis changed")
    if sha(ROOT / "scripts/diagnose_landslide_stage_grounding.py") != diagnosis["script_sha256"]:
        raise ValueError("Diagnostic executable changed")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    case_ids = list(diagnosis["per_case"])
    groups = 0
    for arm in ARMS:
        final = [row for case_id in case_ids for row in json.loads(
            (public / f"daily/{case_id}-{arm}.json").read_text(encoding="utf-8"))]
        native = [row for case_id in case_ids for row in json.loads(
            (public / f"native/{case_id}-{arm}.json").read_text(encoding="utf-8"))]
        for rows in (final, native):
            if [(r["case_id"], r["day_index"], r["date"]) for r in rows] != [
                    (r["case_id"], r["day_index"], r["date"]) for r in reference]:
                raise ValueError("Full calendar alignment differs")
        check_metrics(reference, final, analysis["development_agreement"][arm], lambda ref: True)
        check_metrics(reference, native, diagnosis["native_agreement"][arm], lambda ref: True)
        groups += 2
        for case_id, item in diagnosis["per_case"].items():
            check_metrics(reference, final, item["agreement"][arm], lambda ref: ref["case_id"] == case_id)
            groups += 1
        common_ids = diagnosis["common_success_sensitivity"]["case_ids"]
        check_metrics(reference, final, diagnosis["common_success_sensitivity"]["agreement"][arm],
                      lambda ref: ref["case_id"] in common_ids)
        groups += 1
        for item in diagnosis["interiors"]:
            start, stop = item["interior"]
            selected_case = item["case_id"]
            check_metrics(reference, final, item["agreement"][arm],
                          lambda ref: ref["case_id"] == selected_case and start <= ref["day_index"] < stop)
            groups += 1
    inventories = json.loads((public / "image-inventories.json").read_text(encoding="utf-8"))
    checked_stages = 0
    for case_id, audits in diagnosis["citation_audits"].items():
        inventory = {item["id"]: item for item in inventories[case_id]["images"]}
        for arm, audit in audits.items():
            if "proposed_stages" not in audit:
                continue
            answer = diagnosis["raw_answers"][f"{case_id}-{arm}"]
            for proposal, saved in zip(answer["stages"], audit["stages"], strict=True):
                tags = set(re.findall(r"\[(R\d+|M\d+)\]", proposal["evidence"]))
                unknown = tags - set(inventory)
                covering = [tag for tag in tags if tag in inventory and tag.startswith("R")
                            and inventory[tag]["axis_days_inclusive"][0] <= proposal["start"]
                            and proposal["stop"] <= inventory[tag]["axis_days_inclusive"][1] + 1]
                auxiliary = any(tag in inventory and tag.startswith("M") for tag in tags)
                if saved["locator_and_coverage_pass"] != bool(covering and auxiliary and not unknown):
                    raise ValueError("Independent citation coverage differs")
                checked_stages += 1
    freeze = json.loads((RUN / "presentation-freeze.json").read_text(encoding="utf-8"))
    for name, digest in freeze["public_files_sha256"].items():
        if sha(public / name) != digest:
            raise ValueError("Frozen public file changed")
    result = {"status": "passed", "sklearn_metric_groups": groups,
              "proposed_stage_locator_checks": checked_stages, "public_freeze_verified": True,
              "new_model_requests": 0, "reference_use": "development_only", "browser_rendering": "not_verified"}
    write_json(RUN / "diagnosis-verification-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
