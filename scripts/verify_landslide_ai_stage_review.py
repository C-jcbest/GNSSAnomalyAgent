"""Verify frozen AI review, deterministic compilation, UI scripts and HTTP bytes."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from landslide_stage_reference import DEFAULT_KIT, ROOT, read_json, verify_bundle

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_stage_reference import compile_reference

REVIEW = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1"


def require_digest(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def page_packet(page: Path) -> dict:
    html = page.read_text(encoding="utf-8")
    match = re.search(r'<script id="page-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    if match is None:
        raise ValueError(f"Page payload missing: {page}")
    return json.loads(match.group(1))


def verify_http(item: tuple[str, Path]) -> int:
    url, path = item
    with urlopen(url, timeout=10) as response:
        data = response.read()
        if response.status != 200:
            raise ValueError(f"HTTP status failed: {url}")
    if hashlib.sha256(data).hexdigest() != sha(path):
        raise ValueError(f"HTTP bytes differ from disk: {url}")
    return len(data)


def verify() -> dict:
    contexts, source_digest = verify_bundle(DEFAULT_KIT)
    freeze = read_json(REVIEW / "review-freeze.json")
    if freeze["source_manifest_sha256"] != source_digest:
        raise ValueError("Source review kit identity changed")
    for name, digest in freeze["notes_sha256"].items():
        require_digest(REVIEW / name, digest)
    frozen_paths = {
        "corrections_sha256": REVIEW / "decision-corrections.json",
        "implementation_sha256": ROOT / "scripts/build_landslide_ai_stage_review.py",
        "submitted_review_sha256": REVIEW / "submitted-review.json",
        "compiled_freeze_sha256": REVIEW / "compiled/reference-freeze.json",
        "delivery_audit_sha256": REVIEW / "delivery-audit.json",
    }
    for key, path in frozen_paths.items():
        require_digest(path, freeze[key])
    for name, digest in read_json(REVIEW / "compiled/reference-freeze.json")["files_sha256"].items():
        require_digest(REVIEW / "compiled" / name, digest)
    delivery = read_json(REVIEW / "delivery-audit.json")
    for name, digest in delivery["public_files_sha256"].items():
        require_digest(REVIEW / "public" / name, digest)
    ledger = read_json(REVIEW / "inspection-ledger.json")
    for name, digest in ledger["images_actually_inspected_sha256"].items():
        require_digest(REVIEW / "public" / name, digest)

    submission = read_json(REVIEW / "submitted-review.json")
    compiled = compile_reference(submission, contexts, source_digest)
    for name, result in zip(
        ("review.json", "daily-reference.json", "summary.json", "support-inventory.json"), compiled, strict=True,
    ):
        if result != read_json(REVIEW / "compiled" / name):
            raise ValueError(f"Deterministic compilation differs: {name}")
    _, daily, summary, _ = compiled
    if summary["reference_use"] != "development_only" or summary["performance_scores"] is not None:
        raise ValueError("AI review claimed formal performance")
    original_activity = {(context["case"].case_id, day): row["activity_label"]
                         for context in contexts for day, row in enumerate(context["rows"])}
    for row in daily:
        if row["activity_label"] != original_activity[(row["case_id"], row["day_index"])]:
            raise ValueError("AI stages changed original activity")
    with (REVIEW / "compiled/daily-reference.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if csv_rows != [{key: str(value) for key, value in row.items()} for row in daily]:
        raise ValueError("CSV differs from JSON reference")

    public_page = REVIEW / "public/index.html"
    presentation_page = REVIEW / "presentation-v1/index.html"
    presentation = read_json(REVIEW / "presentation-v1/presentation-freeze.json")
    require_digest(public_page, presentation["source_page_sha256"])
    require_digest(presentation_page, presentation["page_sha256"])
    require_digest(REVIEW / "review-freeze.json", presentation["review_freeze_sha256"])
    require_digest(ROOT / "scripts/build_landslide_ai_review_presentation.py", presentation["implementation_sha256"])
    if page_packet(public_page) != page_packet(presentation_page):
        raise ValueError("Presentation changed the frozen payload")
    if page_packet(public_page)["template"] != submission:
        raise ValueError("Page initial answers differ from submitted review")

    ui_results = []
    for page in (public_page, presentation_page):
        result = subprocess.run(
            ["node", str(ROOT / "scripts/check_landslide_ai_stage_review_ui.cjs"),
             str(page), str(REVIEW / "compiled/daily-reference.json")],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        ui_results.append(json.loads(result.stdout))
    http_items = [(f"http://127.0.0.1:18787/{name}", REVIEW / "public" / name)
                  for name in delivery["public_files_sha256"]]
    http_items.extend([
        ("http://127.0.0.1:18788/presentation-v1/index.html", presentation_page),
        ("http://127.0.0.1:18788/presentation-v1/presentation-freeze.json",
         REVIEW / "presentation-v1/presentation-freeze.json"),
    ])
    # The added page resolves its base URL on 18788, so verify those assets there too.
    http_items.extend((f"http://127.0.0.1:18788/public/{name}", REVIEW / "public" / name)
                      for name in delivery["public_files_sha256"])
    with ThreadPoolExecutor(max_workers=4) as executor:
        transferred_bytes = sum(executor.map(verify_http, http_items))

    per_case = []
    for case in submission["cases"]:
        active = [row for row in daily if row["case_id"] == case["case_id"] and row["activity_label"] == 1]
        features = Counter(row["feature"] for row in active)
        per_case.append({"case_id": case["case_id"], "interiors": len(case["interiors"]),
                         "observed_activity_days": len(active), "features": dict(features)})
    result = {
        "status": "passed", "rows": len(daily), "source_kit_unchanged": True,
        "frozen_files_unchanged": True, "deterministic_compilation": "exact_match",
        "csv_json_match": True, "presentation_payload_unchanged": True,
        "inspected_images": len(ledger["images_actually_inspected_sha256"]),
        "public_files": len(delivery["public_files_sha256"]),
        "http_resources": len(http_items), "http_bytes": transferred_bytes,
        "ui_script_checks": ui_results, "per_case": per_case,
        "coverage": summary["stage_evaluable_days"] / sum(row["activity_label"] == 1 for row in daily),
        "reference_use": "development_only", "performance_scores": None,
        "browser_rendering": "not_verified; CUA failed to write kernel assets (os error 3)",
    }
    write_json(REVIEW / "validation-audit.json", result)
    return result


if __name__ == "__main__":
    audit = verify()
    print(json.dumps({key: value for key, value in audit.items() if key != "ui_script_checks"}, ensure_ascii=False))
