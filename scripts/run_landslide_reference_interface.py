"""Freeze and run all-record paired raw-only activity interface comparisons."""
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
    activity_prompt,
    parse_activity_draft,
    unknown_rows,
)
from gnss_sim.landslide_reference_interface import (  # noqa: E402
    parse_partition_draft,
    partition_prompt,
    partition_sha256,
)
from gnss_sim.landslide_visual import VisualRequests  # noqa: E402
from gnss_sim.visual import _unique_object  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def verify_previous_run(directory):
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["evidence_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Historical evidence changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Historical source changed: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-reference-interface-v2.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/landslide-reference-interface-2026-10-08/run-v2")
    args = parser.parse_args()
    config = read_json(args.config)
    required = {
        "model": "qwen3.8-flash", "arms": ["episode_v1", "partition_v2"],
        "maximum_requests": 24, "repetitions": 1, "temperature": 0.1, "top_p": 0.3,
        "max_tokens": 2048, "retries": 0, "stage_calls": 0, "performance_scoring": False,
        "hard_timeout_seconds": 1800, "sealed_test_access": "none",
        "order": "case ascending; episode first for odd case numbers, partition first for even",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Configuration differs from the frozen paired experiment")
    historical = ROOT / config["historical_run"]
    historical_manifest = verify_previous_run(historical)
    source = ROOT / config["source_packet"]
    source_manifest = read_json(source / "manifest.json")
    records = sorted(source_manifest["cases"], key=lambda record: record["case_id"])
    if [row["case_id"] for row in records] != [f"case_{index:04d}" for index in range(1, 13)]:
        raise ValueError("The entire twelve-record development packet is required")
    if source_manifest["sealed_test_count"] != 12:
        raise ValueError("Unexpected sealed test count")
    source_files = historical_manifest["source_packet_sha256"]
    for name, expected in source_files.items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen development packet changed: {name}")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    (public / "drafts").mkdir(parents=True)
    for name, expected in source_files.items():
        destination_name = "source-" + name if name in {"manifest.json", "protocol.json", "index.html"} else name
        destination = public / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        if sha(destination) != expected:
            raise ValueError(f"Frozen packet copy changed: {name}")
    source_names = [
        "src/gnss_sim/landslide_reference_interface.py", "scripts/run_landslide_reference_interface.py",
        "configs/landslide-reference-interface-v2.json", "docs/landslide-reference-interface-protocol.md",
        "src/gnss_sim/landslide_reference_draft.py", "src/gnss_sim/landslide_observation_review.py",
        "src/gnss_sim/landslide_visual.py", "src/gnss_sim/visual.py", "src/gnss_sim/landslide.py",
        "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_evaluation.py",
        "src/gnss_sim/schemas.py", "src/gnss_sim/artifacts.py",
    ]
    source_hashes = {}
    for name in source_names:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hashes[name] = sha(ROOT / name)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(args.config, public / "protocol.json")
    cases = {}
    diagnostics = {}
    packets = []
    for record in records:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        cases[case_id] = case
        diagnostics[case_id] = LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"]["61"]["data"]).read_bytes(),
        )
        images = [view["image"] for view in record["views"]]
        if len(images) != 4 or not all("-raw-" in name for name in images):
            raise ValueError("Exactly four frozen raw images are required")
        order = list(config["arms"])
        if int(case_id.rsplit("_", 1)[1]) % 2 == 0:
            order.reverse()
        for arm in order:
            prompt_function = activity_prompt if arm == "episode_v1" else partition_prompt
            packets.append({"case_id": case_id, "arm": arm, "request_id": case_id + "-" + arm,
                            "images": images, "prompt": prompt_function(case_id, len(case.dates) - 1, images)})
    if len(packets) != config["maximum_requests"]:
        raise ValueError("Request plan does not match the fixed budget")
    write_json(public / "packets.json", packets)
    write_json(out / "run-ledger-start.json", {
        "status": "running", "command": sys.argv, "requests": 24, "stages": 0,
        "request_plan_sha256": sha(public / "packets.json"), "source_sha256": source_hashes,
        "historical_manifest_sha256": sha(historical / "manifest.json"),
    })
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 24)
    record_by_id = {record["case_id"]: record for record in records}
    results = []
    all_rows = []
    started = time.monotonic()
    for packet in packets:
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise TimeoutError("Paired run reached its fixed timeout before the next request")
        case_id, arm = packet["case_id"], packet["arm"]
        case, record = cases[case_id], record_by_id[case_id]
        result = {"case_id": case_id, "arm": arm, "request_id": packet["request_id"],
                  "status": "failed_unknown", "error": None, "review_file": None}
        rows = unknown_rows(case)
        try:
            payload = transport.ask(packet["request_id"], packet["prompt"],
                                    [public / name for name in packet["images"]])
            if arm == "episode_v1":
                review, rows = parse_activity_draft(payload, case, record, diagnostics[case_id], config["model"])
                review_hash = activity_review_sha256(review)
                review_data = review.model_dump(mode="json")
            else:
                review_data, rows = parse_partition_draft(payload, case, record, config["model"])
                review_hash = partition_sha256(review_data)
            review_name = f"drafts/{packet['request_id']}.review.json"
            write_json(public / review_name, review_data)
            result.update(status="valid_ai_draft", review_file=review_name, activity_sha256=review_hash)
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
            rows = unknown_rows(case)
        result["activity_counts"] = dict(Counter(str(row["activity_label"]) for row in rows))
        result["observed_days"] = record["observed_days"]
        result["observed_unknown_days"] = result["activity_counts"].get("-1", 0) - record["missing_days"]
        write_json(public / f"drafts/{packet['request_id']}.daily.json", rows)
        write_json(out / f"call-status/{packet['request_id']}.json", result)
        results.append(result)
        all_rows.extend({"arm": arm, **row} for row in rows)
        print(f"{len(results)}/24 {packet['request_id']}: {result['status']}", flush=True)
    with (public / "daily-activity-drafts.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    receipts = [read_json(public / "requests" / (packet["request_id"] + ".response.json")) for packet in packets]
    summary = {"protocol_id": config["protocol_id"], "results": results, "calls": len(results),
               "total_tokens": sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts),
               "service_seconds": sum(receipt["seconds"] for receipt in receipts),
               "elapsed_seconds": time.monotonic() - started,
               "independent_expert_reference": False, "performance_scores": None,
               "stage_calls": 0, "retries": 0}
    write_json(public / "drafts.json", summary)
    verify_previous_run(historical)
    for name, expected in source_files.items():
        if sha(source / name) != expected:
            raise ValueError(f"Development packet changed during the experiment: {name}")
    write_json(out / "activity-pass-freeze.json", {
        "status": "both complete raw-only arms frozen; no stage inference",
        "drafts_sha256": {path.name: sha(path) for path in (public / "drafts").iterdir()},
    })
    write_json(out / "manifest.json", {
        "status": "complete", "source_packet": config["source_packet"],
        "source_packet_sha256": source_files, "source_sha256": source_hashes,
        "historical_run": config["historical_run"], "historical_unchanged": True,
        "calls": 24, "stage_calls": 0, "retries": 0, "sealed_test_read": False,
        "evidence_sha256": {path.relative_to(out).as_posix(): sha(path)
                            for path in out.rglob("*") if path.is_file()},
    })
    print("COMPLETE 24 raw-only calls; both arms frozen; no accuracy scores", flush=True)


if __name__ == "__main__":
    main()
