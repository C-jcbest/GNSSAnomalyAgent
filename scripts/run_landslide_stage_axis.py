"""Run paired repeated stage inference differing only in auxiliary horizontal presentation."""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_landslide_frozen_stage_heads import read_json, verify_hashes  # noqa: E402
from build_landslide_raw_adjudication import verify as verify_activity_run  # noqa: E402
from run_landslide_frozen_stage_heads import copy_frozen  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    compile_stage_answer,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_stage_axis_experiment import render_auxiliary_axis_pair  # noqa: E402
from gnss_sim.landslide_visual import VisualRequests  # noqa: E402


def main():
    config_path = ROOT / "configs/landslide-stage-axis-v1.json"
    config = read_json(config_path)
    required = {"conditions": ["dates", "day_index"], "repetitions": 2, "maximum_requests": 36,
                "model": "qwen3.8-flash", "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048,
                "retries": 0, "scoring": False, "sealed_test_access": "none"}
    if any(config[k] != v for k, v in required.items()):
        raise ValueError("Configuration differs from the preregistered presentation protocol")
    previous = ROOT / config["source_run"]
    old_public = previous / "public"
    old_manifest = read_json(previous / "manifest.json")
    verify_hashes(previous, old_manifest)
    author = ROOT / config["activity_run"]
    verify_activity_run(author)
    out = ROOT / "artifacts/landslide-stage-axis-2026-10-08/run-v1"
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    public.mkdir()
    sources = ["configs/landslide-stage-axis-v1.json", "docs/landslide-stage-axis-protocol.md",
               "src/gnss_sim/landslide_stage_axis_experiment.py", "tests/test_landslide_stage_axis_experiment.py",
               "scripts/run_landslide_stage_axis.py", "scripts/analyze_landslide_stage_axis.py",
               "scripts/landslide_stage_axis_page.html", "src/gnss_sim/landslide_frozen_stage_heads.py",
               "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_visual.py",
               "src/gnss_sim/visual.py", "src/gnss_sim/landslide_trace.py",
               "scripts/analyze_landslide_frozen_stage_heads.py", "scripts/run_landslide_frozen_stage_heads.py"]
    source_hashes = {name: sha(ROOT / name) for name in sources}
    for name, digest in source_hashes.items():
        copy_frozen(ROOT / name, out / "source-code" / name, digest)
    ledger = read_json(previous / "run-ledger-start.json")
    for name, digest in ledger["prepared_inputs_sha256"].items():
        copy_frozen(old_public / name, public / "source" / name, digest)
    # The numeric heads are unchanged historical context, not newly executed competitors.
    old_freeze = read_json(previous / "inference-freeze.json")
    for name, digest in old_freeze["evidence_sha256"].items():
        relative = Path(name).relative_to("public")
        if relative.parts[0] in {"native", "daily"} and not relative.stem.endswith("visual"):
            copy_frozen(previous / name, public / "source" / relative, digest)
    write_json(public / "protocol.json", config)
    records = read_json(public / "source/source-manifest.json")["cases"]
    if [r["case_id"] for r in records] != [f"case_{i:04d}" for i in range(1, 13)]:
        raise ValueError("Complete twelve-case development packet required")
    old_packets = {p["case_id"]: p for p in read_json(public / "source/packets.json")}
    old_entries = {r["case_id"]: r for r in read_json(author / "public/manifest.json")["cases"]}
    cases, motions, reviews, activity_rows, audits = {}, {}, {}, {}, {}
    for record in records:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((public / "source" / record["input"]).read_bytes())
        review = read_json(public / "source/activity-source" / old_entries[case_id]["review"])
        rows = read_json(public / "source/activity-source" / old_entries[case_id]["daily"])
        digest = read_json(public / "source/activity-freeze.json")["reviews"][case_id]["activity_sha256"]
        validate_activity(case, review, rows, digest)
        cases[case_id], reviews[case_id], activity_rows[case_id] = case, review, rows
        motions[case_id] = {}
        for view in record["views"]:
            copy_frozen(public / "source" / view["image"], public / "day-index" / view["image"], view["image_sha256"])
        for window in (31, 61, 91):
            source = record["diagnostics"][str(window)]
            motion = LandslideDiagnostics.model_validate_json((public / "source" / source["data"]).read_bytes())
            if motion.case_id != case_id or motion.input_sha256 != record["input_sha256"] or motion.window_days != window:
                raise ValueError("Auxiliary provenance differs from frozen observations")
            motions[case_id][window] = motion
            control, day_image, audit = render_auxiliary_axis_pair(case, motion, record["views"][0]["audit"]["tick_audit"]["ticks"])
            if hashlib.sha256(control).hexdigest() != source["image_sha256"]:
                raise ValueError(f"Legacy auxiliary PNG did not replay exactly: {case_id}/{window}")
            target = public / "day-index" / source["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(day_image)
            audits[f"{case_id}-{window}"] = {**audit, "dates_image": "source/" + source["image"],
                                            "day_index_image": "day-index/" + source["image"],
                                            "dates_sha256": source["image_sha256"], "day_index_sha256": sha(target)}
        print(f"prepared {case_id}: all three legacy PNGs match; curve geometry unchanged", flush=True)
    write_json(public / "image-audits.json", audits)
    packets = []
    active_number = 0
    for record in records:
        case_id = record["case_id"]
        if case_id not in old_packets:
            continue
        active_number += 1
        order = [("dates", 1), ("day_index", 1), ("day_index", 2), ("dates", 2)]
        if active_number % 2 == 0:
            order = [("day_index", 1), ("dates", 1), ("dates", 2), ("day_index", 2)]
        old_packet = old_packets[case_id]
        for condition, repetition in order:
            image_root = "source" if condition == "dates" else "day-index"
            packets.append({"case_id": case_id, "condition": condition, "repetition": repetition,
                            "request_id": f"{case_id}-{condition}-r{repetition}",
                            "activity_sha256": old_packet["activity_sha256"],
                            "prompt": old_packet["prompt"], "logical_images": old_packet["images"],
                            "images": [image_root + "/" + name for name in old_packet["images"]]})
    if len(packets) != 36:
        raise ValueError("Expected exactly thirty-six preregistered requests")
    write_json(public / "packets.json", packets)
    prepared = {p.relative_to(public).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()}
    write_json(out / "run-ledger-start.json", {"command": sys.argv, "started_utc": datetime.now(timezone.utc).isoformat(),
                                              "prepared_inputs_sha256": prepared,
                                              "source_inference_freeze_sha256": sha(previous / "inference-freeze.json"),
                                              "source_manifest_sha256": sha(previous / "manifest.json"),
                                              "source_sha256": source_hashes, "maximum_requests": 36})
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 36)
    results, csv_rows = [], []
    started = time.monotonic()
    for packet in packets:
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise TimeoutError("Presentation experiment reached the hard timeout")
        case_id, request_id = packet["case_id"], packet["request_id"]
        rows = activity_rows[case_id]
        status, error = "success", None
        try:
            answer = transport.ask(request_id, packet["prompt"], [public / name for name in packet["images"]])
            review, native, final, suppressed = compile_stage_answer(
                answer, cases[case_id], reviews[case_id], rows, packet["activity_sha256"], motions[case_id][61])
            write_json(public / f"stage-reviews/{request_id}.json", review)
        except (RuntimeError, ValueError, TypeError, KeyError) as failure:
            status = "failed_unknown_activity_stages"
            error = {"type": type(failure).__name__, "message": str(failure)[:400]}
            native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
        write_json(public / f"native/{request_id}.json", native)
        write_json(public / f"daily/{request_id}.json", final)
        result = {"case_id": case_id, "condition": packet["condition"], "repetition": packet["repetition"],
                  "request_id": request_id, "status": status, "error": error,
                  "native_activity_features": dict(Counter(r["feature"] for r in native if r["activity_label"] == 1)),
                  "final_activity_features": dict(Counter(r["feature"] for r in final if r["activity_label"] == 1)),
                  "suppressed_support": suppressed}
        results.append(result)
        csv_rows.extend({"condition": packet["condition"], "repetition": packet["repetition"], **r} for r in final)
        write_json(out / f"case-status/{request_id}.json", result)
        print(f"{len(results)}/36 {request_id}: {status}", flush=True)
    for record in records:
        case_id = record["case_id"]
        if case_id in old_packets:
            continue
        for condition in config["conditions"]:
            for repetition in (1, 2):
                request_id = f"{case_id}-{condition}-r{repetition}"
                rows = unclassified_rows(activity_rows[case_id])
                write_json(public / f"native/{request_id}.json", rows)
                write_json(public / f"daily/{request_id}.json", rows)
                results.append({"case_id": case_id, "condition": condition, "repetition": repetition,
                                "request_id": request_id, "status": "skipped_no_confirmed_activity", "error": None,
                                "native_activity_features": {}, "final_activity_features": {}, "suppressed_support": {}})
                csv_rows.extend({"condition": condition, "repetition": repetition, **r} for r in rows)
    receipts = [read_json(p) for p in (public / "requests").glob("*.response.json")]
    write_json(public / "results.json", {"protocol_id": config["protocol_id"], "results": results,
                                         "calls": len(receipts), "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in receipts),
                                         "service_seconds": sum(r["seconds"] for r in receipts),
                                         "elapsed_seconds": time.monotonic() - started,
                                         "performance_scores": None, "independent_stage_reference": False,
                                         "sealed_test_read": False, "retries": 0})
    with (public / "daily-stage-axis.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    verify_hashes(previous, old_manifest)
    verify_activity_run(author)
    if any(sha(public / name) != digest for name, digest in prepared.items()):
        raise ValueError("Prepared input changed during inference")
    write_json(out / "inference-freeze.json", {"evidence_sha256": {
        p.relative_to(out).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()}})
    write_json(out / "manifest.json", {"status": "complete", "config": config, "source_sha256": source_hashes,
                                       "inference_freeze_sha256": sha(out / "inference-freeze.json"),
                                       "calls": len(receipts), "source_run_unchanged": True, "performance_scores": None})
    print(f"COMPLETE: {len(receipts)} calls; frozen activity and previous inference unchanged", flush=True)


if __name__ == "__main__":
    main()
