"""Independently check post-freeze subgroup arithmetic and supplemental delivery."""

from __future__ import annotations

import hashlib
import json
from urllib.request import urlopen

from landslide_stage_turning import ROOT, RUN
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json

STAGES = ["acceleration", "steady_motion", "deceleration"]
ARMS = ["image-r1", "numeric-r1", "image-r2", "numeric-r2"]


def check_metrics(reference, prediction, reported, select):
    pairs = [(ref, pred) for ref, pred in zip(reference, prediction, strict=True)
             if select(ref) and ref["stage_evaluable"]]
    actual = [ref["feature"] for ref, _ in pairs]
    guessed = [pred["feature"] for _, pred in pairs]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, guessed, labels=STAGES, zero_division=0,
    )
    same = sum(a == b for a, b in zip(actual, guessed, strict=True))
    unknown = guessed.count("unknown")
    expected = {"reference_evaluable_days": len(pairs), "exact_agreement_days": same,
                "different_class_days": len(pairs) - same - unknown,
                "abstained_evaluable_days": unknown}
    for name, value in expected.items():
        if reported[name] != value:
            raise ValueError(f"Subgroup count differs: {name}")
    for index, stage in enumerate(STAGES):
        if reported["per_class"][stage]["reference_support"] != int(support[index]):
            raise ValueError("Reference class support differs")
        for name, value in (("precision", precision[index]), ("recall", recall[index]), ("f1", f1[index])):
            reported_value = reported["per_class"][stage][name]
            if reported_value is not None and abs(reported_value - value) > 1e-12:
                raise ValueError("Subgroup arithmetic differs from sklearn")
    macro = reported["development_agreement_macro_f1"]
    if all(support > 0):
        if abs(macro - float(f1.mean())) > 1e-12:
            raise ValueError("Macro-F1 differs")
    elif macro is not None:
        raise ValueError("Unsupported class must not have a macro-F1")


def main():
    public = RUN / "public"
    diagnosis = json.loads((RUN / "diagnosis-v1.json").read_text(encoding="utf-8"))
    analysis = json.loads((public / "analysis.json").read_text(encoding="utf-8"))
    reference_path = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1/compiled/daily-reference.json"
    if sha(reference_path) != diagnosis["reference_sha256"] or sha(public / "analysis.json") != diagnosis["primary_analysis_sha256"]:
        raise ValueError("Diagnosis source changed")
    if sha(ROOT / "scripts/diagnose_landslide_stage_turning.py") != diagnosis["script_sha256"]:
        raise ValueError("Diagnosis executable changed")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    case_ids = list(diagnosis["per_case"])
    groups = 0
    for arm in ARMS:
        final = [row for case_id in case_ids for row in json.loads(
            (public / f"daily/{case_id}-{arm}.json").read_text(encoding="utf-8"))]
        native = [row for case_id in case_ids for row in json.loads(
            (public / f"native/{case_id}-{arm}.json").read_text(encoding="utf-8"))]
        if [(r["case_id"], r["day_index"], r["date"]) for r in reference] != [
                (r["case_id"], r["day_index"], r["date"]) for r in final]:
            raise ValueError("Reference / prediction calendar differs")
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
        for name, has_candidates in (("with_candidates", True), ("without_candidates", False)):
            dates = {(item["case_id"], day) for item in diagnosis["interiors"]
                     if bool(item["candidate_count"]) == has_candidates
                     for day in range(*item["interior"])}
            check_metrics(reference, final, diagnosis["candidate_availability_groups"][name][arm],
                          lambda ref: (ref["case_id"], ref["day_index"]) in dates)
            groups += 1
    freeze = json.loads((RUN / "presentation-v1/freeze.json").read_text(encoding="utf-8"))
    if sha(public / "index.html") != freeze["frozen_primary_page_sha256"]:
        raise ValueError("Primary page changed")
    if sha(RUN / "diagnosis-v1.json") != freeze["diagnosis_sha256"]:
        raise ValueError("Presentation diagnosis changed")
    if sha(ROOT / "scripts/build_landslide_stage_turning_presentation.py") != freeze["builder_sha256"]:
        raise ValueError("Presentation builder changed")
    for name, digest in freeze["files_sha256"].items():
        if sha(RUN / "presentation-v1" / name) != digest:
            raise ValueError("Supplemental page changed")
    names = ["presentation-v1/index.html", "presentation-v1/report.md", "diagnosis-v1.json",
             "public/index.html", "public/images/case_0005-raw-0.png",
             "public/requests/case_0011-numeric-r2.response.json"]
    for name in names:
        with urlopen(f"http://127.0.0.1:18791/{name}", timeout=10) as response:
            if response.status != 200 or hashlib.sha256(response.read()).hexdigest() != sha(RUN / name):
                raise ValueError("Supplemental HTTP bytes differ")
    result = {"status": "passed", "sklearn_metric_groups": groups,
              "supplemental_http_files": len(names), "frozen_primary_page_unchanged": True,
              "supplemental_freeze_verified": True, "new_model_requests": 0,
              "reference_use": "development_only", "browser_rendering": "not_verified"}
    write_json(RUN / "diagnosis-verification-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
