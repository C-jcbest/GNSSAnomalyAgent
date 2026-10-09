"""Read frozen predictions and report development agreement without new inference."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations

from landslide_stage_reference import DEFAULT_KIT, ROOT, read_json, verify_bundle

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_stage_agreement import (
    disagreement_spans,
    stage_agreement,
    support_filter_effect,
)
from gnss_sim.landslide_stage_reference import compile_reference

CONFIG = ROOT / "configs/landslide-stage-agreement-v1.json"
SOURCE_FILES = (
    "configs/landslide-stage-agreement-v1.json",
    "docs/landslide-stage-agreement-protocol.md",
    "src/gnss_sim/landslide_stage_agreement.py",
    "scripts/analyze_landslide_stage_agreement.py",
    "scripts/landslide_stage_agreement_page.html",
    "tests/test_landslide_stage_agreement.py",
)


def check_digest(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def load_inputs(config):
    prediction_run = ROOT / config["prediction_run"]
    reference_run = ROOT / config["reference_run"]
    inputs = {}

    def check(path, expected):
        check_digest(path, expected)
        inputs[path.relative_to(ROOT).as_posix()] = expected

    manifest = read_json(prediction_run / "manifest.json")
    check(prediction_run / "inference-freeze.json", manifest["inference_freeze_sha256"])
    for name, digest in read_json(prediction_run / "inference-freeze.json")["evidence_sha256"].items():
        check(prediction_run / name, digest)
    for name, digest in manifest["source_sha256"].items():
        check(ROOT / name, digest)
        check(prediction_run / "source" / name, digest)
    check(prediction_run / "models/stage-xgboost.json", manifest["model_sha256"])
    for item in read_json(prediction_run / "delivery-audit.json")["files"]:
        check(prediction_run / "public" / item["path"], item["sha256"])

    freeze = read_json(reference_run / "review-freeze.json")
    for name, digest in freeze["notes_sha256"].items():
        check(reference_run / name, digest)
    check(reference_run / "decision-corrections.json", freeze["corrections_sha256"])
    check(ROOT / "scripts/build_landslide_ai_stage_review.py", freeze["implementation_sha256"])
    check(reference_run / "submitted-review.json", freeze["submitted_review_sha256"])
    check(reference_run / "compiled/reference-freeze.json", freeze["compiled_freeze_sha256"])
    check(reference_run / "delivery-audit.json", freeze["delivery_audit_sha256"])
    for name, digest in read_json(reference_run / "compiled/reference-freeze.json")["files_sha256"].items():
        check(reference_run / "compiled" / name, digest)
    for name, digest in read_json(reference_run / "delivery-audit.json")["public_files_sha256"].items():
        check(reference_run / "public" / name, digest)
    contexts, source_digest = verify_bundle(DEFAULT_KIT)
    packet = read_json(reference_run / "submitted-review.json")
    compiled = compile_reference(packet, contexts, source_digest)
    for name, result in zip(
        ("review.json", "daily-reference.json", "summary.json", "support-inventory.json"), compiled, strict=True,
    ):
        if result != read_json(reference_run / "compiled" / name):
            raise ValueError(f"Reference recompilation differs: {name}")
    if compiled[2]["reference_use"] != "development_only":
        raise ValueError("This protocol is specifically a development review diagnostic")
    for path in (prediction_run / "manifest.json", prediction_run / "delivery-audit.json",
                 reference_run / "review-freeze.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)

    html = (reference_run / "public/index.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="page-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    if match is None:
        raise ValueError("Frozen review page lacks display payload")
    display = json.loads(match.group(1))
    return packet, compiled[1], display, inputs


def paired_agreement(reference, first, second):
    counts = Counter()
    for actual, a, b in zip(reference, first, second, strict=True):
        if not actual["stage_evaluable"]:
            continue
        a_agrees = a["feature"] == actual["feature"]
        b_agrees = b["feature"] == actual["feature"]
        if a_agrees and b_agrees:
            counts["both_agree"] += 1
        elif a_agrees:
            counts["first_only_agrees"] += 1
        elif b_agrees:
            counts["second_only_agrees"] += 1
        else:
            counts["neither_agrees"] += 1
    return {name: counts[name] for name in (
        "both_agree", "first_only_agrees", "second_only_agrees", "neither_agrees",
    )}


def main():
    config = read_json(CONFIG)
    output = ROOT / config["output_run"]
    if output.exists():
        raise FileExistsError("Use a new version; agreement output must not overwrite frozen results")
    packet, reference, display, inputs = load_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    write_json(output / "run-ledger-start.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "input_files_sha256": inputs, "source_files_sha256": source_hashes,
        "reference_use": "development_only", "new_inference_requests": 0,
    })
    for name in SOURCE_FILES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    by_case = {}
    for row in reference:
        by_case.setdefault(row["case_id"], []).append(row)
    predictions = {mode: {method: [] for method in config["methods"]} for mode in ("native", "final")}
    page_cases = []
    daily_csv = []
    all_spans = []
    display_cases = {case["case_id"]: case for case in display["cases"]}
    public = output / "public"
    (public / "images").mkdir(parents=True)
    for case in packet["cases"]:
        case_id = case["case_id"]
        actual = by_case[case_id]
        case_predictions = {}
        case_scores = {}
        for mode, directory in (("native", "native"), ("final", "daily")):
            case_predictions[mode] = {}
            case_scores[mode] = {}
            for method in config["methods"]:
                relative = f"public/{directory}/{case_id}-{method}.json"
                rows = read_json(ROOT / config["prediction_run"] / relative)
                case_scores[mode][method] = stage_agreement(actual, rows)
                case_predictions[mode][method] = rows
                predictions[mode][method].extend(rows)
                for reference_row, predicted in zip(actual, rows, strict=True):
                    daily_csv.append({"mode": mode, "method": method, **predicted,
                                      "reference_feature": reference_row["feature"],
                                      "reference_evaluable": reference_row["stage_evaluable"],
                                      "reference_reason": reference_row["reference_reason"]})
                if mode == "final":
                    all_spans.extend({"method": method, **span} for span in disagreement_spans(actual, rows))
        effects = {method: support_filter_effect(actual, case_predictions["native"][method],
                                                case_predictions["final"][method]) for method in config["methods"]}
        interiors = []
        for interior in case["interiors"]:
            start, stop = interior["start"], interior["stop"]
            interior_scores = {method: stage_agreement(actual[start:stop], case_predictions["final"][method][start:stop])
                               for method in config["methods"]}
            segment_scores = []
            for segment in interior["segments"]:
                a, b = segment["start"], segment["stop"]
                segment_scores.append({**segment, "methods": {
                    method: stage_agreement(actual[a:b], case_predictions["final"][method][a:b])
                    for method in config["methods"]}})
            interiors.append({"start": start, "stop": stop, "methods": interior_scores,
                              "raw_activity_evidence": interior["raw_activity_evidence"],
                              "segments": segment_scores})
        visible = display_cases[case_id]
        for image_name in (*visible["raw_images"], *visible["auxiliary_images"]):
            shutil.copyfile(ROOT / config["reference_run"] / "public" / image_name, public / image_name)
        page_cases.append({
            "case_id": case_id, "scores": case_scores, "support_filter": effects,
            "interiors": interiors, "raw_images": visible["raw_images"],
            "auxiliary_images": visible["auxiliary_images"], "observed": visible["observed"],
            "reference_features": [row["feature"] for row in actual],
            "reference_evaluable": [row["stage_evaluable"] for row in actual],
            "prediction_features": {
                mode: {method: [row["feature"] for row in rows] for method, rows in heads.items()}
                for mode, heads in case_predictions.items()},
        })

    scores = {mode: {method: stage_agreement(reference, rows) for method, rows in heads.items()}
              for mode, heads in predictions.items()}
    effects = {method: support_filter_effect(reference, predictions["native"][method],
                                            predictions["final"][method]) for method in config["methods"]}
    paired = {f"{a}_vs_{b}": paired_agreement(reference, predictions["final"][a], predictions["final"][b])
              for a, b in combinations(config["methods"], 2)}
    summary = {
        "protocol_id": config["protocol_id"], "reference_use": "development_only",
        "formal_reference_verified": False, "performance_scores": None,
        "reference_reviewer": packet["reviewer"], "cases": len(packet["cases"]),
        "interiors": sum(len(case["interiors"]) for case in packet["cases"]),
        "reference_features": dict(Counter(row["feature"] for row in reference if row["activity_label"] == 1)),
        "methods": scores, "support_filter": effects, "paired_agreement": paired,
        "source_files_verified": len(inputs), "new_inference_requests": 0,
        "prediction_rows": len(daily_csv), "disagreement_spans": len(all_spans),
        "s_supporting_cases": [case["case_id"] for case in page_cases
                               if case["scores"]["final"]["rule"]["per_class"]["steady_motion"]["reference_support"]],
    }
    write_json(public / "summary.json", summary)
    write_json(public / "disagreement-spans.json", all_spans)
    write_json(public / "case-analysis.json", page_cases)
    write_json(public / "reference-review.json", packet)
    with (public / "daily-comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily_csv[0]))
        writer.writeheader()
        writer.writerows(daily_csv)
    data = {"summary": summary, "cases": page_cases, "disagreements": all_spans}
    html = (ROOT / "scripts/landslide_stage_agreement_page.html").read_text(encoding="utf-8")
    if html.count("__PAGE_DATA__") != 1:
        raise ValueError("Page payload placeholder must be unique")
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    (public / "index.html").write_text(html.replace("__PAGE_DATA__", encoded), encoding="utf-8")
    for name, digest in inputs.items():
        check_digest(ROOT / name, digest)
    write_json(output / "analysis-freeze.json", {
        "input_files_sha256": inputs, "source_files_sha256": source_hashes,
        "public_files_sha256": {path.relative_to(public).as_posix(): sha(path)
                                for path in public.rglob("*") if path.is_file()},
        "reference_use": "development_only", "performance_scores": None,
    })
    print(json.dumps({"reference_features": summary["reference_features"], "final": scores["final"],
                      "native": scores["native"], "support_filter": effects,
                      "paired": paired, "disagreement_spans": len(all_spans)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
