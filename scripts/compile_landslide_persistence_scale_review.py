"""Compile explicitly authored observational labels for the scale extension."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compile_landslide_paired_confirmation_review import numerical_inventory, render_review
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_reference_interface import parse_partition_draft, partition_sha256
from gnss_sim.landslide_stage_reference import compile_reference, reference_template


def compile_review(directory: Path) -> None:
    out = directory / "compiled"
    if out.exists():
        raise FileExistsError("Preserve an existing reference; use a new version")
    for name, digest in read_json(directory / "preparation-manifest.json")["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Prepared observation changed: " + name)
    activity = read_json(directory / "activity-decisions.json")
    activity_freeze = read_json(directory / "activity-freeze.json")
    if sha(directory / "activity-decisions.json") != activity_freeze["activity_decisions_sha256"]:
        raise ValueError("Activity decisions changed after stage review started")
    stages = read_json(directory / "stage-decisions.json")
    expected_ids = read_json(directory.parent / "protocol.json")["secondary_review_ids"]
    if list(activity["cases"]) != expected_ids or list(stages["cases"]) != expected_ids:
        raise ValueError("Retain every pre-specified review record in order")
    contexts = []
    for cid, spans in activity["cases"].items():
        folder = directory / cid
        case = LandslideInput.model_validate_json((folder / "input.json").read_bytes())
        input_digest = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        record = {"input_sha256": input_digest, "views": [
            {"image_sha256": sha(folder / name)} for name in ("raw-global.png", "raw-local.png")
        ]}
        payload = {
            "spans": [dict(zip(("start", "stop", "state", "raw_evidence"), span, strict=True))
                      for span in spans],
            "activity_notes": activity["activity_notes"],
        }
        review, rows = parse_partition_draft(payload, case, record, "Codex direct review")
        motions = {window: LandslideDiagnostics.model_validate_json(
            (folder / f"motion-{window}.json").read_bytes()) for window in (31, 61, 91)}
        if any(motion.input_sha256 != input_digest for motion in motions.values()):
            raise ValueError("Auxiliary evidence does not identify the reviewed raw input")
        contexts.append({"case": case, "review": review, "rows": rows,
                         "activity_sha256": partition_sha256(review), "motions": motions})
    source_sha = sha(directory / "preparation-manifest.json")
    packet = reference_template(contexts, source_sha)
    packet["reviewer"] = {
        "name": "Codex / actual visual inspection and numerical assistance",
        "background": "AI开发者，知道历史方法与标签；本批具体预测及过程参考未读取。非独立专家。",
        "reviewer_type": "ai", "prior_predictions_seen": True,
        "prior_generation_labels_seen": True, "participated_in_activity_review": True,
    }
    stamp = datetime.now(timezone.utc).isoformat()
    rate_checks = {}
    for entry, context in zip(packet["cases"], contexts, strict=True):
        segments = [dict(zip(stages["columns"], row, strict=True))
                    for row in stages["cases"][entry["case_id"]]]
        used = []
        for interior in entry["interiors"]:
            selected = [segment for segment in segments
                        if interior["start"] <= segment["start"] < interior["stop"]]
            used.extend(selected)
            span = next(span for span in context["review"]["spans"]
                        if span["start"] == interior["start"])
            interior.update(review_status="reviewed", raw_activity_status="confirmed",
                            raw_activity_evidence=span["raw_evidence"], raw_views_seen=True,
                            auxiliary_views_seen=True, reviewed_at=stamp, segments=selected)
        if used != segments:
            raise ValueError("Stage segment outside frozen activity")
        rate_checks[entry["case_id"]] = numerical_inventory(context, segments)
    compiled, daily, summary, support = compile_reference(packet, contexts, source_sha)
    summary.update(activity_counts=dict(Counter(row["activity_label"] for row in daily)),
                   calendar_rows=len(daily), missing_days=sum(
                       value is None for context in contexts for value in context["case"].displacement_mm))
    out.mkdir()
    for name, value in (
        ("activity-reviews.json", [context["review"] for context in contexts]),
        ("review.json", compiled), ("daily-reference.json", daily), ("summary.json", summary),
        ("support-inventory.json", support), ("rate-checks.json", rate_checks),
    ):
        export(out / name, value)
    write_csv(out / "daily-reference.csv", daily)
    plots = {}
    for context in contexts:
        cid = context["case"].case_id
        rows = [row for row in daily if row["case_id"] == cid]
        plots[cid] = render_review(context["case"], rows, out / f"{cid}-review.png")
    export(out / "plot-data.json", plots)
    export(out / "inspection-ledger.json", {
        "raw_images": 2 * len(contexts), "auxiliary_images": len(contexts),
        "images_actually_inspected_sha256": {
            f"{cid}/{name}": sha(directory / cid / name) for cid in expected_ids
            for name in ("raw-global.png", "raw-local.png", "auxiliary.png")},
        "process": "Raw global/local and 30-day blocks -> activity freeze -> auxiliary and multiscale rates -> stages",
        "specific_predictions_seen": False, "specific_process_labels_seen": False,
        "independent_expert": False, "new_remote_requests": 0,
    })
    export(out / "reference-freeze.json", {
        "reviewed_at": stamp, "activity_decisions_sha256": sha(directory / "activity-decisions.json"),
        "stage_decisions_sha256": sha(directory / "stage-decisions.json"),
        "formal_reference_verified": False,
        "files": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    })
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_directory", type=Path)
    compile_review(parser.parse_args().review_directory.resolve())
