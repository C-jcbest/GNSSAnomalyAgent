"""Validate an observed development reference; pending templates cannot be exported."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_observation_review import (  # noqa: E402
    ObservationReview,
    activity_review_sha256,
    compile_observation_review,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path, help="Public development packet")
    parser.add_argument("--review", required=True, type=Path, help="Completed observation review JSON")
    parser.add_argument("--activity-hash-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New exclusive reference directory")
    args = parser.parse_args()
    review = ObservationReview.model_validate_json(args.review.read_text(encoding="utf-8"))
    manifest = json.loads((args.packet / "manifest.json").read_text(encoding="utf-8"))
    record = next(row for row in manifest["cases"] if row["case_id"] == review.case_id)
    case = LandslideInput.model_validate_json((args.packet / record["input"]).read_bytes())
    diagnostics = LandslideDiagnostics.model_validate_json(
        (args.packet / record["diagnostics"]["61"]["data"]).read_bytes(),
    )
    image_hash = sha(args.packet / record["views"][0]["image"])
    if image_hash != record["views"][0]["image_sha256"]:
        raise ValueError("Review packet raw image differs from its manifest")
    if args.activity_hash_only:
        if review.stages:
            raise ValueError("Freeze an activity-only review before adding stages")
        compile_observation_review(case, review, diagnostics, image_hash)
        print(activity_review_sha256(review))
        return
    rows = compile_observation_review(case, review, diagnostics, image_hash)
    if args.output is None:
        raise ValueError("An exclusive output directory is required")
    args.output.mkdir(parents=True, exist_ok=False)
    labels = args.output / "daily-labels.csv"
    with labels.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "review.json", review.model_dump(mode="json"))
    write_json(args.output / "freeze.json", {
        "status": "compiled_observation_reference", "case_id": case.case_id,
        "input_sha256": review.input_sha256, "raw_image_sha256": image_hash,
        "activity_sha256": activity_review_sha256(review), "labels_sha256": sha(labels),
        "reviewer": review.reviewer, "provenance": review.provenance,
        "independent_reference_verified": False,
        "counts": {str(label): sum(row["activity_label"] == label for row in rows)
                   for label in (-1, 0, 1)},
    })
    print(f"Validated observed labels saved to {labels}")


if __name__ == "__main__":
    main()
