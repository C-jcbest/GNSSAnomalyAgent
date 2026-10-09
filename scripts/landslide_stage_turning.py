"""Prepare, run, replay and present a new paired turning-proposal experiment."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter

from analyze_landslide_stage_axis import replay_request
from analyze_landslide_stage_axis import verify_run as verify_axis_run
from landslide_stage_evidence import feature_counts, load_case
from landslide_stage_reference import ROOT, read_json
from run_landslide_frozen_stage_heads import copy_frozen

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_frozen_stage_heads import (
    compile_stage_answer,
    stage_pair_counts,
    unclassified_rows,
)
from gnss_sim.landslide_stage_agreement import stage_agreement
from gnss_sim.landslide_stage_numeric_experiment import (
    ARMS,
    render_numeric_comparison,
    request_order,
)
from gnss_sim.landslide_stage_turning_experiment import turning_prompt, turning_summary
from gnss_sim.landslide_visual import VisualRequests

CONFIG_PATH = ROOT / "configs/landslide-stage-turning-v1.json"
RUN = ROOT / "artifacts/landslide-stage-turning-2026-10-08/run-v2"
SOURCES = (
    "configs/landslide-stage-turning-v1.json", "docs/landslide-stage-turning-protocol.md",
    "scripts/landslide_stage_turning.py", "scripts/landslide_stage_turning_page.html",
    "src/gnss_sim/landslide_stage_turning_experiment.py", "tests/test_landslide_stage_turning_experiment.py",
    "src/gnss_sim/landslide_stage_numeric_experiment.py", "src/gnss_sim/landslide_stage_agreement.py",
    "scripts/landslide_stage_evidence.py", "scripts/analyze_landslide_stage_axis.py",
    "src/gnss_sim/landslide_frozen_stage_heads.py", "src/gnss_sim/landslide_visual.py",
    "src/gnss_sim/visual.py",
)


def configuration():
    config = read_json(CONFIG_PATH)
    required = {"conditions": ["image", "numeric"], "repetitions": 2, "maximum_requests": 36,
                "model": "qwen3.8-flash", "temperature": 0.1, "top_p": 0.3,
                "max_tokens": 2048, "retries": 0, "formal_performance_scores": False,
                "sealed_test_access": "none"}
    if any(config[key] != value for key, value in required.items()) or ROOT / config["output_run"] != RUN:
        raise ValueError("Configuration differs from this version's fixed protocol")
    return config


def prepare():
    config = configuration()
    axis = ROOT / config["source_axis_run"]
    axis_manifest, _ = verify_axis_run(axis)
    source = axis / "public/source"
    records = read_json(source / "source-manifest.json")["cases"]
    author = ROOT / axis_manifest["config"]["activity_run"]
    entries = {entry["case_id"]: entry for entry in read_json(author / "public/manifest.json")["cases"]}
    previous = {packet["case_id"]: packet for packet in read_json(axis / "public/packets.json")}
    RUN.mkdir(parents=True, exist_ok=False)
    public = RUN / "public"
    public.mkdir()
    for name in ("source-manifest.json", "activity-freeze.json"):
        copy_frozen(source / name, public / "source" / name, sha(source / name))
    write_json(public / "source/activity-manifest.json", {"cases": list(entries.values())})
    write_json(public / "protocol.json", config)
    plot_path = ROOT / "artifacts/landslide-background-2026-10-08/batch-v1/public/protocol.json"
    write_json(public / "plot.json", read_json(plot_path)["plot"])
    for record in records:
        case_id = record["case_id"]
        names = [record["input"], *[record["diagnostics"][str(w)]["data"] for w in (31, 61, 91)],
                 *["activity-source/" + entries[case_id][key] for key in ("review", "daily")]]
        for name in names:
            copy_frozen(source / name, public / "source" / name, sha(source / name))
        images = [view["image"] for view in record["views"]]
        images.extend(record["diagnostics"][str(w)]["image"] for w in (31, 61, 91))
        for name in images:
            origin = axis / "public/day-index" / name
            copy_frozen(origin, public / name, sha(origin))
    summaries, packets = {}, []
    active_number = 0
    for record in records:
        case, review, _, digest, motions = load_case(public, record, entries)
        summary = turning_summary(case, review, motions, config["proposal_parameters"])
        summaries[case.case_id] = summary
        if not summary["activities"]:
            continue
        active_number += 1
        original = previous[case.case_id]
        for condition, repetition in request_order(active_number):
            packets.append({"case_id": case.case_id, "condition": condition, "repetition": repetition,
                            "request_id": f"{case.case_id}-{condition}-r{repetition}",
                            "activity_sha256": digest, "images": original["logical_images"],
                            "prompt": original["prompt"] if condition == "image"
                            else turning_prompt(original["prompt"], summary)})
    if len(records) != 12 or len(packets) != 36:
        raise ValueError("Expected complete twelve-record / thirty-six-request plan")
    write_json(public / "turning-summaries.json", summaries)
    write_json(public / "packets.json", packets)
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name, digest in source_hashes.items():
        copy_frozen(ROOT / name, RUN / "source-code" / name, digest)
    write_json(RUN / "run-ledger-start.json", {
        "prepared_inputs_sha256": {p.relative_to(public).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()},
        "source_sha256": source_hashes, "source_axis_manifest_sha256": sha(axis / "manifest.json"),
        "source_axis_freeze_sha256": sha(axis / "inference-freeze.json"),
        "phase": "prepared_without_stage_reference_access", "maximum_requests": 36,
    })
    print({"prepared_requests": len(packets), "candidate_counts": {
        case_id: sum(len(it["candidates"]) for it in summary["activities"])
        for case_id, summary in summaries.items()}})


def verify_prepared():
    config = configuration()
    axis = ROOT / config["source_axis_run"]
    verify_axis_run(axis)
    ledger = read_json(RUN / "run-ledger-start.json")
    for name, digest in ledger["prepared_inputs_sha256"].items():
        if sha(RUN / "public" / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    for name, digest in ledger["source_sha256"].items():
        if sha(ROOT / name) != digest or sha(RUN / "source-code" / name) != digest:
            raise ValueError(f"Execution source changed: {name}")
    if (sha(axis / "manifest.json") != ledger["source_axis_manifest_sha256"]
            or sha(axis / "inference-freeze.json") != ledger["source_axis_freeze_sha256"]):
        raise ValueError("Historical source changed")
    public = RUN / "public"
    records = read_json(public / "source/source-manifest.json")["cases"]
    entries = {entry["case_id"]: entry for entry in read_json(public / "source/activity-manifest.json")["cases"]}
    return config, public, records, entries


def run():
    config, public, records, entries = verify_prepared()
    cases = {record["case_id"]: load_case(public, record, entries) for record in records}
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 36)
    results, csv_rows = [], []
    started = time.monotonic()
    for packet in read_json(public / "packets.json"):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise TimeoutError("Turning experiment reached its hard timeout")
        case_id, request_id = packet["case_id"], packet["request_id"]
        case, review, rows, digest, motions = cases[case_id]
        status, error = "success", None
        try:
            answer = transport.ask(request_id, packet["prompt"], [public / name for name in packet["images"]])
            stage_review, native, final, suppressed = compile_stage_answer(answer, case, review, rows, digest, motions[61])
            write_json(public / f"stage-reviews/{request_id}.json", stage_review)
        except (RuntimeError, ValueError, TypeError, KeyError) as failure:
            status = "failed_unknown_activity_stages"
            error = {"type": type(failure).__name__, "message": str(failure)[:400]}
            native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
        for mode, output in (("native", native), ("daily", final)):
            write_json(public / mode / f"{request_id}.json", output)
        result = {"case_id": case_id, "condition": packet["condition"], "repetition": packet["repetition"],
                  "request_id": request_id, "status": status, "error": error,
                  **feature_counts(native, final, suppressed)}
        results.append(result)
        csv_rows.extend({"condition": packet["condition"], "repetition": packet["repetition"], **row} for row in final)
        write_json(RUN / f"case-status/{request_id}.json", result)
        print(f"{len(results)}/36 {request_id}: {status}", flush=True)
    active_ids = {result["case_id"] for result in results}
    for case_id, (_, _, rows, _, _) in cases.items():
        if case_id in active_ids:
            continue
        output = unclassified_rows(rows)
        for condition in config["conditions"]:
            for repetition in (1, 2):
                request_id = f"{case_id}-{condition}-r{repetition}"
                for mode in ("native", "daily"):
                    write_json(public / mode / f"{request_id}.json", output)
                results.append({"case_id": case_id, "condition": condition, "repetition": repetition,
                                "request_id": request_id, "status": "skipped_no_confirmed_activity", "error": None,
                                **feature_counts(output, output, {})})
                csv_rows.extend({"condition": condition, "repetition": repetition, **row} for row in output)
    receipts = [read_json(path) for path in (public / "requests").glob("*.response.json")]
    write_json(public / "results.json", {
        "results": results, "calls": len(receipts), "elapsed_seconds": time.monotonic() - started,
        "total_tokens": sum(receipt.get("usage", {}).get("total_tokens", 0) for receipt in receipts),
        "receipt_statuses": dict(Counter(receipt["status"] for receipt in receipts)),
        "performance_scores": None, "retries": 0,
    })
    with (public / "daily-stage-turning.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    verify_prepared()
    write_json(RUN / "inference-freeze.json", {"evidence_sha256": {
        p.relative_to(RUN).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()}})
    write_json(RUN / "manifest.json", {"config": config, "status": "complete",
                                        "inference_freeze_sha256": sha(RUN / "inference-freeze.json"),
                                        "performance_scores": None})
    print("COMPLETE: new paired inference frozen; no historical requests retried", flush=True)


def replay():
    config, public, records, entries = verify_prepared()
    manifest = read_json(RUN / "manifest.json")
    if sha(RUN / "inference-freeze.json") != manifest["inference_freeze_sha256"]:
        raise ValueError("Inference freeze changed")
    freeze = read_json(RUN / "inference-freeze.json")
    for name, digest in freeze["evidence_sha256"].items():
        if sha(RUN / name) != digest:
            raise ValueError(f"Inference evidence changed: {name}")
    packets = {p["request_id"]: p for p in read_json(public / "packets.json")}
    results = {r["request_id"]: r for r in read_json(public / "results.json")["results"]}
    summaries = read_json(public / "turning-summaries.json")
    cases, all_rows = [], {}
    for record in records:
        case, review, rows, _, motions = load_case(public, record, entries)
        if turning_summary(case, review, motions, config["proposal_parameters"]) != summaries[case.case_id]:
            raise ValueError("Turning proposals did not replay")
        native, final, requests = {}, {}, {}
        for arm in ARMS:
            request_id = f"{case.case_id}-{arm}"
            saved = results[request_id]
            packet = packets.get(request_id)
            if packet:
                a, b, suppressed, status, error, request = replay_request(public, packet, case, review, rows, motions[61])
                if saved["status"] != status or saved["error"] != error:
                    raise ValueError("Request status replay differs")
                if any(saved[key] != value for key, value in feature_counts(a, b, suppressed).items()):
                    raise ValueError("Stage counts replay differs")
            else:
                a, b, request = unclassified_rows(rows), unclassified_rows(rows), None
            if a != read_json(public / f"native/{request_id}.json") or b != read_json(public / f"daily/{request_id}.json"):
                raise ValueError("Stage replay differs")
            native[arm], final[arm], requests[arm] = a, b, request
            all_rows.setdefault(arm, []).extend(b)
        cases.append({"case_id": case.case_id, "case": case, "record": record, "native": native, "final": final,
                      "requests": requests, "summary": summaries[case.case_id],
                      "pairs": {f"{a}_vs_{b}": stage_pair_counts(final[a], final[b])
                                for a, b in (("image-r1", "image-r2"), ("numeric-r1", "numeric-r2"))}})
    return public, cases, all_rows


def analyze():
    public, cases, all_rows = replay()
    # Stage reference access starts only after inference and immutable replay.
    reference_run = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1"
    reference_freeze = read_json(reference_run / "compiled/reference-freeze.json")
    reference_path = reference_run / "compiled/daily-reference.json"
    if sha(reference_path) != reference_freeze["files_sha256"]["daily-reference.json"]:
        raise ValueError("Development reference changed")
    reference = read_json(reference_path)
    agreement = {arm: stage_agreement(reference, rows) for arm, rows in all_rows.items()}
    plot = read_json(public / "plot.json")
    (public / "presentation").mkdir(exist_ok=False)
    audits = {}
    for item in cases:
        item["plots"] = []
        for index, view in enumerate(item["record"]["views"]):
            name = f"presentation/{item['case_id']}-{index}.png"
            audit = render_numeric_comparison(item["case"], view, item["final"], public / name, plot)
            if audit["xlim"] != view["audit"]["panels"][0]["xlim"] or audit["ylim_NEU"] != [p["ylim"] for p in view["audit"]["panels"]]:
                raise ValueError("Frozen axes changed")
            audits[name] = {**audit, "sha256": sha(public / name)}
            item["plots"].append(name)
        for key in ("case", "record", "native", "final"):
            del item[key]
    summary = read_json(public / "results.json")
    analysis = {"development_agreement": agreement, "reference_use": "development_only", "performance_scores": None,
                "reference_sha256": sha(reference_path), "plots": len(audits), "requests_replayed": 36,
                "pairs": {arm: sum(sum(n for key, n in item["pairs"][arm]["transitions"].items()
                                            if key.split("->")[0] != key.split("->")[1]) for item in cases)
                          for arm in ("image-r1_vs_image-r2", "numeric-r1_vs_numeric-r2")}}
    write_json(public / "analysis.json", analysis)
    write_json(public / "presentation/plot-audits.json", audits)
    encoded = json.dumps({"summary": summary, "analysis": analysis, "cases": cases}, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "scripts/landslide_stage_turning_page.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(template.replace("__PAGE_DATA__", encoded), encoding="utf-8")
    replay()
    write_json(RUN / "presentation-freeze.json", {"public_files_sha256": {
        p.relative_to(public).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()},
        "reference_sha256": sha(reference_path), "inference_unchanged": True})
    print(json.dumps(analysis, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze", "verify"))
    mode = parser.parse_args().mode
    if mode == "prepare":
        prepare()
    elif mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        replay()
        print("Verified inference and candidate summaries")


if __name__ == "__main__":
    main()
