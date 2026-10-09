"""Execute and freeze raw-first development drafts; never score against these drafts."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_observation_review import activity_review_sha256  # noqa: E402
from gnss_sim.landslide_reference_draft import (  # noqa: E402
    PROVENANCE,
    activity_prompt,
    parse_activity_draft,
    parse_stage_draft,
    render_draft_overlay,
    stage_prompt,
    unknown_rows,
)
from gnss_sim.landslide_visual import VisualRequests  # noqa: E402


def copy_frozen(source: Path, destination: Path, expected: str):
    if sha(source) != expected:
        raise ValueError("Review input differs from the frozen source packet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(destination) != expected:
        raise ValueError("Review input copy changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-reference-draft-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/landslide-reference-draft-2026-10-08/run-v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config["maximum_requests"], config["repetitions"], config["temperature"], config["top_p"],
        config["max_tokens"], config["retries"], config["scoring_this_step"]) != (24, 1, 0.1, 0.3, 2048, 0, False):
        raise ValueError("Configuration differs from the fixed transport and draft-only policy")
    source = ROOT / config["source_packet"]
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if len(source_manifest["cases"]) != 12 or source_manifest["sealed_test_count"] != 12:
        raise ValueError("Expected the complete twelve-record development packet")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    public.mkdir()
    (public / "drafts").mkdir()
    source_files = {path.relative_to(source).as_posix(): sha(path)
                    for path in source.rglob("*") if path.is_file()}
    for name, expected in source_files.items():
        destination_name = "source-" + name if name in {"manifest.json", "protocol.json", "index.html"} else name
        copy_frozen(source / name, public / destination_name, expected)
    source_names = ["src/gnss_sim/landslide_reference_draft.py",
                    "src/gnss_sim/landslide_observation_review.py", "src/gnss_sim/landslide_visual.py",
                    "src/gnss_sim/visual.py", "scripts/run_landslide_reference_draft.py",
                    "configs/landslide-reference-draft-v1.json", "docs/landslide-reference-draft-protocol.md"]
    source_hashes = {}
    for name in source_names:
        source_hashes[name] = sha(ROOT / name)
        copy_frozen(ROOT / name, out / "source" / name, source_hashes[name])
    copy_frozen(args.config, public / "protocol.json", sha(args.config))
    write_json(out / "run-ledger-start.json", {"status": "running", "command": sys.argv,
                                               "maximum_requests": 24, "provenance": PROVENANCE})
    cases, diagnostics, records, outputs = {}, {}, {}, {}
    for record in source_manifest["cases"]:
        case_id = record["case_id"]
        cases[case_id] = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        diagnostics[case_id] = LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"]["61"]["data"]).read_bytes(),
        )
        records[case_id] = record
        outputs[case_id] = {"case_id": case_id, "activity_status": "pending", "stage_status": "pending",
                            "review": None, "rows": unknown_rows(cases[case_id]), "calls": [], "errors": []}
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 24)
    packets = []
    start_time = time.monotonic()
    # Complete the whole raw-only layer before ANY second-layer request.
    for case_id in sorted(cases):
        if time.monotonic() - start_time > config["hard_timeout_seconds"]:
            raise TimeoutError("Draft run reached its fixed timeout before the next request")
        record, case, result = records[case_id], cases[case_id], outputs[case_id]
        image_names = [view["image"] for view in record["views"]]
        prompt = activity_prompt(case_id, len(case.dates) - 1, image_names)
        request_id = case_id + "-activity"
        packet = {"case_id": case_id, "layer": "activity", "request_id": request_id,
                  "images": image_names, "prompt": prompt}
        packets.append(packet)
        result["calls"].append(request_id)
        try:
            payload = transport.ask(request_id, prompt, [public / name for name in image_names])
            review, rows = parse_activity_draft(payload, case, record, diagnostics[case_id], config["model"])
            result.update(activity_status="valid_ai_draft", review=review, rows=rows)
            write_json(public / f"drafts/{case_id}.activity-freeze.json", {
                "activity_sha256": activity_review_sha256(review), "stage_review_not_started": True,
                "provenance": PROVENANCE, "review": review.model_dump(mode="json"),
            })
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            result["activity_status"] = "failed_unknown"
            result["errors"].append({"layer": "activity", "type": type(error).__name__, "message": str(error)[:400]})
        write_json(out / f"activity-status/{case_id}.json", {key: result[key]
                   for key in ("case_id", "activity_status", "errors")})
        print(f"ACTIVITY {case_id}: {result['activity_status']}", flush=True)
    write_json(out / "activity-pass-freeze.json", {
        "status": "raw-only pass complete before any stage calls",
        "reviews_sha256": {path.name: sha(path) for path in (public / "drafts").glob("*.activity-freeze.json")},
    })
    for case_id in sorted(cases):
        record, case, result = records[case_id], cases[case_id], outputs[case_id]
        review = result["review"]
        if review is None:
            result["stage_status"] = "skipped_failed_activity"
            continue
        if not review.episodes:
            result["stage_status"] = "not_applicable_no_confirmed_activity"
            continue
        if time.monotonic() - start_time > config["hard_timeout_seconds"]:
            raise TimeoutError("Draft run reached its fixed timeout before the next request")
        image_names = [view["image"] for view in record["views"]]
        image_names.extend(record["diagnostics"][str(window)]["image"] for window in (31, 61, 91))
        prompt = stage_prompt(review, len(case.dates) - 1, image_names)
        request_id = case_id + "-stages"
        packet = {"case_id": case_id, "layer": "stages", "request_id": request_id,
                  "images": image_names, "prompt": prompt, "activity_sha256": activity_review_sha256(review)}
        packets.append(packet)
        result["calls"].append(request_id)
        try:
            payload = transport.ask(request_id, prompt, [public / name for name in image_names])
            staged, rows = parse_stage_draft(payload, review, case, diagnostics[case_id])
            if activity_review_sha256(staged) != activity_review_sha256(review):
                raise ValueError("Stage review changed its frozen activity")
            result.update(stage_status="valid_ai_draft", review=staged, rows=rows,
                          stage_notes=payload["stage_notes"])
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            result["stage_status"] = "failed_unknown_stages"
            result["errors"].append({"layer": "stages", "type": type(error).__name__, "message": str(error)[:400]})
        print(f"STAGES {case_id}: {result['stage_status']}", flush=True)
    summary = {"protocol_id": config["protocol_id"], "provenance": PROVENANCE,
               "independent_expert_reference": False, "performance_scores": None,
               "cases": [], "calls": len(packets), "planned_maximum_calls": 24}
    all_rows = []
    plot = json.loads((public / "source-protocol.json").read_text(encoding="utf-8"))["plot"]
    for case_id in sorted(cases):
        result, record = outputs[case_id], records[case_id]
        rows = result.pop("rows")
        all_rows.extend(rows)
        review = result.pop("review")
        if review is not None:
            review_name = f"drafts/{case_id}.review.json"
            write_json(public / review_name, review.model_dump(mode="json"))
            result["review_file"] = review_name
        image_name = f"images/{case_id}-draft-overlay.png"
        spans = render_draft_overlay(cases[case_id], record, rows, public / image_name, plot)
        result.update(overlay=image_name, overlay_sha256=sha(public / image_name), mask_spans=spans,
                      activity_counts=dict(Counter(str(row["activity_label"]) for row in rows)),
                      feature_counts=dict(Counter(row["feature"] for row in rows)),
                      input_sha256=record["input_sha256"])
        write_json(public / f"drafts/{case_id}.daily.json", rows)
        summary["cases"].append(result)
    with (public / "daily-draft-labels.csv").open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    receipts = [json.loads((public / "requests" / (packet["request_id"] + ".response.json")).read_text(encoding="utf-8"))
                for packet in packets]
    summary["total_tokens"] = sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts)
    summary["service_seconds"] = sum(receipt["seconds"] for receipt in receipts)
    summary["elapsed_seconds"] = time.monotonic() - start_time
    summary["activity_counts"] = dict(Counter(str(row["activity_label"]) for row in all_rows))
    summary["feature_counts"] = dict(Counter(row["feature"] for row in all_rows))
    write_json(public / "packets.json", packets)
    write_json(public / "drafts.json", summary)
    for name, expected in source_files.items():
        if sha(source / name) != expected:
            raise ValueError("Original development packet changed during draft review")
    write_json(out / "manifest.json", {
        "status": "complete", "source_packet": config["source_packet"], "source_packet_sha256": source_files,
        "source_sha256": source_hashes, "calls": len(packets), "provenance": PROVENANCE,
        "source_packet_unchanged": True, "sealed_test_read": False, "retries": 0,
        "evidence_sha256": {path.relative_to(out).as_posix(): sha(path)
                            for path in out.rglob("*") if path.is_file()},
    })
    print(f"COMPLETE {len(packets)} calls; AI development drafts only; no performance scores", flush=True)


if __name__ == "__main__":
    main()
