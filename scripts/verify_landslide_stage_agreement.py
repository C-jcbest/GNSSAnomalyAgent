"""Independently cross-check development agreement arithmetic and delivery."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/landslide-stage-agreement-2026-10-08/analysis-v1"
STAGES = ("acceleration", "steady_motion", "deceleration")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_digest(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def check_class_metrics(rows, reported):
    evaluable = [row for row in rows if row["reference_evaluable"] == "True"]
    if len(evaluable) != reported["reference_evaluable_days"]:
        raise ValueError("Different evaluable denominator")
    if not evaluable:
        if reported["agreement_rate"] is not None or reported["development_agreement_macro_f1"] is not None:
            raise ValueError("Empty reference acquired a metric")
        return
    actual = [row["reference_feature"] for row in evaluable]
    predicted = [row["feature"] for row in evaluable]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        actual, predicted, labels=list(STAGES), average=None, zero_division=0,
    )
    for index, stage in enumerate(STAGES):
        result = reported["per_class"][stage]
        if int(supports[index]) != result["reference_support"]:
            raise ValueError("Reference class support differs")
        for key, value in (("precision", precisions[index]), ("recall", recalls[index]), ("f1", f1s[index])):
            if result[key] is not None and abs(result[key] - float(value)) > 1e-12:
                raise ValueError(f"Class metric differs from sklearn: {stage} {key}")
    if reported["development_agreement_macro_f1"] is not None:
        if abs(float(f1s.mean()) - reported["development_agreement_macro_f1"]) > 1e-12:
            raise ValueError("Macro arithmetic differs from sklearn")
    same = sum(a == b for a, b in zip(actual, predicted, strict=True))
    abstained = predicted.count("unknown")
    if (same != reported["exact_agreement_days"] or abstained != reported["abstained_evaluable_days"]
            or len(evaluable) - same - abstained != reported["different_class_days"]):
        raise ValueError("Agreement / abstention accounting differs")


def verify_http(item):
    name, digest = item
    with urlopen(f"http://127.0.0.1:18789/{name}", timeout=10) as response:
        data = response.read()
        if response.status != 200 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"HTTP delivery differs: {name}")
    return len(data)


def main():
    freeze = read_json(RUN / "analysis-freeze.json")
    for name, digest in freeze["input_files_sha256"].items():
        check_digest(ROOT / name, digest)
    for name, digest in freeze["source_files_sha256"].items():
        check_digest(ROOT / name, digest)
        check_digest(RUN / "source" / name, digest)
    for name, digest in freeze["public_files_sha256"].items():
        check_digest(RUN / "public" / name, digest)
    summary = read_json(RUN / "public/summary.json")
    cases = read_json(RUN / "public/case-analysis.json")
    case_map = {case["case_id"]: case for case in cases}
    with (RUN / "public/daily-comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    totals = defaultdict(list)
    by_case = defaultdict(list)
    seen = set()
    for row in rows:
        mode, method, case_id = row["mode"], row["method"], row["case_id"]
        day = int(row["day_index"])
        identity = (mode, method, case_id, day)
        if identity in seen:
            raise ValueError("Duplicate prediction identity")
        seen.add(identity)
        totals[(mode, method)].append(row)
        by_case[(mode, method, case_id)].append(row)
        case = case_map[case_id]
        if (row["feature"] != case["prediction_features"][mode][method][day]
                or row["reference_feature"] != case["reference_features"][day]
                or row["reference_evaluable"] != str(case["reference_evaluable"][day])):
            raise ValueError("CSV differs from page label data")
    if len(rows) != 78840 or len(totals) != 6 or len(by_case) != 72:
        raise ValueError("Incomplete prediction calendar")
    for (mode, method), values in totals.items():
        check_class_metrics(values, summary["methods"][mode][method])
    for (mode, method, case_id), values in by_case.items():
        check_class_metrics(values, case_map[case_id]["scores"][mode][method])
    node = subprocess.run(
        ["node", str(ROOT / "scripts/check_landslide_stage_agreement_ui.cjs"),
         str(RUN / "public/index.html")],
        check=True, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        transferred = sum(executor.map(verify_http, freeze["public_files_sha256"].items()))
    result = {
        "status": "passed", "input_hashes": len(freeze["input_files_sha256"]),
        "source_files": len(freeze["source_files_sha256"]),
        "public_files_http_verified": len(freeze["public_files_sha256"]),
        "http_bytes": transferred, "csv_rows": len(rows),
        "independent_sklearn_checks": len(totals) + len(by_case), "metric_tolerance": 1e-12,
        "csv_page_labels_match": True, "ui_script_check": json.loads(node.stdout),
        "reference_use": "development_only", "performance_scores": None,
        "browser_rendering": "not_verified; previously observed CUA kernel assets path error persists as an unresolved tool limitation",
    }
    write_json(RUN / "verification-audit.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
