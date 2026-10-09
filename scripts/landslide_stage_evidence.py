"""Run, replay and present the frozen per-activity evidence-instruction experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/landslide-stage-evidence-2026-10-08/run-v1"
sys.path.insert(0, str(ROOT / "src"))

from analyze_landslide_frozen_stage_heads import read_json  # noqa: E402
from analyze_landslide_stage_axis import (  # noqa: E402
    replay_request,
    save_display_json,
    sign_counts,
)
from analyze_landslide_stage_axis import (
    verify_run as verify_axis_run,
)
from run_landslide_frozen_stage_heads import copy_frozen  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    compile_stage_answer,
    stage_pair_counts,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_stage_evidence_experiment import (  # noqa: E402
    ARMS,
    PAIRS,
    evidence_prompt,
    render_evidence_comparison,
    request_order,
)
from gnss_sim.landslide_visual import VisualRequests  # noqa: E402

SOURCE_FILES = (
    "configs/landslide-stage-evidence-v1.json", "docs/landslide-stage-evidence-protocol.md",
    "docs/landslide-independent-stage-review-guide.md",
    "src/gnss_sim/landslide_stage_evidence_experiment.py", "tests/test_landslide_stage_evidence_experiment.py",
    "scripts/landslide_stage_evidence.py", "scripts/landslide_stage_evidence_page.html",
    "scripts/analyze_landslide_stage_axis.py", "scripts/analyze_landslide_frozen_stage_heads.py",
    "scripts/run_landslide_frozen_stage_heads.py", "scripts/build_landslide_raw_adjudication.py",
    "src/gnss_sim/landslide_stage_axis_experiment.py", "src/gnss_sim/landslide_frozen_stage_heads.py",
    "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_visual.py",
    "src/gnss_sim/visual.py", "src/gnss_sim/landslide_trace.py",
)


def load_case(public, record, entries):
    case_id = record["case_id"]
    source = public / "source"
    case = LandslideInput.model_validate_json((source / record["input"]).read_bytes())
    review = read_json(source / "activity-source" / entries[case_id]["review"])
    rows = read_json(source / "activity-source" / entries[case_id]["daily"])
    digest = read_json(source / "activity-freeze.json")["reviews"][case_id]["activity_sha256"]
    validate_activity(case, review, rows, digest)
    motions = {}
    for window in (31, 61, 91):
        motion = LandslideDiagnostics.model_validate_json((source / record["diagnostics"][str(window)]["data"]).read_bytes())
        if motion.case_id != case_id or motion.input_sha256 != record["input_sha256"] or motion.window_days != window:
            raise ValueError("Frozen diagnostic provenance mismatch")
        motions[window] = motion
    return case, review, rows, digest, motions


def feature_counts(native, final, suppressed):
    return {
        "native_activity_features": dict(Counter(r["feature"] for r in native if r["activity_label"] == 1)),
        "final_activity_features": dict(Counter(r["feature"] for r in final if r["activity_label"] == 1)),
        "suppressed_support": suppressed,
    }


def prepare(config):
    axis = ROOT / config["source_axis_run"]
    axis_manifest, _ = verify_axis_run(axis)
    old_public = axis / "public"
    source = old_public / "source"
    records = read_json(source / "source-manifest.json")["cases"]
    if [r["case_id"] for r in records] != [f"case_{i:04d}" for i in range(1, 13)]:
        raise ValueError("Expected all twelve development records")
    author = ROOT / axis_manifest["config"]["activity_run"]
    entries = {r["case_id"]: r for r in read_json(author / "public/manifest.json")["cases"]}
    old_packets = {p["case_id"]: p for p in read_json(old_public / "packets.json")}
    RUN.mkdir(parents=True, exist_ok=False)
    public = RUN / "public"
    public.mkdir()
    for name in ("source-manifest.json", "activity-freeze.json"):
        copy_frozen(source / name, public / "source" / name, sha(source / name))
    write_json(public / "source/activity-manifest.json", {"cases": list(entries.values())})
    write_json(public / "protocol.json", config)
    guide = ROOT / "docs/landslide-independent-stage-review-guide.md"
    copy_frozen(guide, public / "independent-stage-review-guide.md", sha(guide))
    image_audits = {}
    for record in records:
        case_id = record["case_id"]
        metadata = [record["input"], *[record["diagnostics"][str(w)]["data"] for w in (31, 61, 91)],
                    *["activity-source/" + entries[case_id][key] for key in ("review", "daily")]]
        for name in metadata:
            copy_frozen(source / name, public / "source" / name, sha(source / name))
        images = [v["image"] for v in record["views"]]
        images.extend(record["diagnostics"][str(w)]["image"] for w in (31, 61, 91))
        for name in images:
            previous = old_public / "day-index" / name
            copy_frozen(previous, public / name, sha(previous))
            image_audits[name] = {"sha256": sha(previous), "previous_path": previous.relative_to(ROOT).as_posix()}
    write_json(public / "input-image-audits.json", image_audits)
    packets, review_template = [], []
    active_number = 0
    for record in records:
        case, review, rows, digest, _ = load_case(public, record, entries)
        interiors = [s for s in review["spans"] if s["state"] == "activity"]
        review_template.append({
            "case_id": case.case_id, "activity_sha256": digest,
            "raw_images": [v["image"] for v in record["views"]],
            "auxiliary_images": [record["diagnostics"][str(w)]["image"] for w in (31, 61, 91)],
            "interiors": [{"start": s["start"], "stop": s["stop"], "review_status": "unreviewed",
                           "observed_direction": None, "rate_evolution": None, "turn_uncertainty": [],
                           "steady_nonzero_evidence": None, "stages": [], "notes": None} for s in interiors],
        })
        if not interiors:
            if case.case_id in old_packets:
                raise ValueError("Historical active packet differs from frozen activity")
            continue
        active_number += 1
        previous = old_packets[case.case_id]
        for condition, repetition in request_order(active_number):
            packets.append({"case_id": case.case_id, "condition": condition, "repetition": repetition,
                            "request_id": f"{case.case_id}-{condition}-r{repetition}",
                            "activity_sha256": digest, "logical_images": previous["logical_images"],
                            "images": previous["logical_images"],
                            "prompt": evidence_prompt(previous["prompt"], condition)})
    if len(packets) != 36 or sum(len(c["interiors"]) for c in review_template) != 21:
        raise ValueError("Planned packet or independent review template incomplete")
    write_json(public / "packets.json", packets)
    write_json(public / "independent-stage-review-template.json", {
        "status": "UNREVIEWED preparation only; NOT independent labels", "reviewer_identity": None,
        "reviewer_blinding": None, "reference_source": "author review-v3 activity, conditional stage review",
        "cases": review_template,
    })
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    for name, digest in source_hashes.items():
        copy_frozen(ROOT / name, RUN / "source-code" / name, digest)
    write_json(RUN / "run-ledger-start.json", {
        "command": sys.argv, "source_axis_manifest_sha256": sha(axis / "manifest.json"),
        "source_axis_freeze_sha256": sha(axis / "inference-freeze.json"),
        "source_sha256": source_hashes,
        "prepared_inputs_sha256": {p.relative_to(public).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()},
        "maximum_requests": 36,
    })
    return public, records, entries, packets, source_hashes


def run_inference():
    config = read_json(ROOT / "configs/landslide-stage-evidence-v1.json")
    expected = {"conditions": ["control", "structured"], "repetitions": 2, "maximum_requests": 36,
                "model": "qwen3.8-flash", "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048,
                "retries": 0, "scoring": False, "sealed_test_access": "none", "hard_timeout_seconds": 7200}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Configuration differs from frozen protocol")
    public, records, entries, packets, source_hashes = prepare(config)
    print("PREPARED: all inputs frozen; both conditions use identical seven day-index images", flush=True)
    cases = {r["case_id"]: load_case(public, r, entries) for r in records}
    transport = VisualRequests(public / "requests", ROOT / ".env", config["model"], 36)
    results, csv_rows = [], []
    started = time.monotonic()
    for packet in packets:
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise TimeoutError("Stage evidence experiment reached hard timeout")
        case_id, request_id = packet["case_id"], packet["request_id"]
        case, activity_review, rows, digest, motions = cases[case_id]
        status, error = "success", None
        try:
            answer = transport.ask(request_id, packet["prompt"], [public / name for name in packet["images"]])
            draft, native, final, suppressed = compile_stage_answer(answer, case, activity_review, rows, digest, motions[61])
            write_json(public / f"stage-reviews/{request_id}.json", draft)
        except (RuntimeError, ValueError, TypeError, KeyError) as failure:
            status = "failed_unknown_activity_stages"
            error = {"type": type(failure).__name__, "message": str(failure)[:400]}
            native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
        write_json(public / f"native/{request_id}.json", native)
        write_json(public / f"daily/{request_id}.json", final)
        result = {key: packet[key] for key in ("case_id", "request_id", "condition", "repetition")}
        result.update(status=status, error=error, **feature_counts(native, final, suppressed))
        results.append(result)
        csv_rows.extend({"condition": packet["condition"], "repetition": packet["repetition"], **r} for r in final)
        write_json(RUN / f"case-status/{request_id}.json", result)
        print(f"{len(results)}/36 {request_id}: {status}", flush=True)
    for record in records:
        case_id = record["case_id"]
        if any(p["case_id"] == case_id for p in packets):
            continue
        rows = unclassified_rows(cases[case_id][2])
        for condition in config["conditions"]:
            for repetition in (1, 2):
                request_id = f"{case_id}-{condition}-r{repetition}"
                write_json(public / f"native/{request_id}.json", rows)
                write_json(public / f"daily/{request_id}.json", rows)
                results.append({"case_id": case_id, "request_id": request_id, "condition": condition,
                                "repetition": repetition, "status": "skipped_no_confirmed_activity", "error": None,
                                **feature_counts(rows, rows, {})})
                csv_rows.extend({"condition": condition, "repetition": repetition, **r} for r in rows)
    receipts = [read_json(p) for p in (public / "requests").glob("*.response.json")]
    write_json(public / "results.json", {
        "protocol_id": config["protocol_id"], "results": results, "calls": len(receipts),
        "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in receipts),
        "service_seconds": sum(r["seconds"] for r in receipts), "elapsed_seconds": time.monotonic() - started,
        "performance_scores": None, "independent_stage_reference": False, "sealed_test_read": False, "retries": 0,
    })
    with (public / "daily-stage-evidence.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    verify_prepared(config, source_hashes)
    write_json(RUN / "inference-freeze.json", {"evidence_sha256": {
        p.relative_to(RUN).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()}})
    write_json(RUN / "manifest.json", {"status": "complete", "config": config, "source_sha256": source_hashes,
                                        "inference_freeze_sha256": sha(RUN / "inference-freeze.json"),
                                        "calls": len(receipts), "source_run_unchanged": True, "performance_scores": None})
    print("COMPLETE: 36 calls; inference frozen; old activity and predictions unchanged", flush=True)


def verify_prepared(config, source_hashes):
    axis = ROOT / config["source_axis_run"]
    verify_axis_run(axis)
    ledger = read_json(RUN / "run-ledger-start.json")
    if sha(axis / "manifest.json") != ledger["source_axis_manifest_sha256"] or sha(axis / "inference-freeze.json") != ledger["source_axis_freeze_sha256"]:
        raise ValueError("Source axis inference changed")
    for name, digest in ledger["prepared_inputs_sha256"].items():
        if sha(RUN / "public" / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest or sha(RUN / "source-code" / name) != digest:
            raise ValueError(f"Execution source changed: {name}")


def verify_inference():
    manifest = read_json(RUN / "manifest.json")
    if manifest["status"] != "complete" or sha(RUN / "inference-freeze.json") != manifest["inference_freeze_sha256"]:
        raise ValueError("Inference incomplete or freeze changed")
    freeze = read_json(RUN / "inference-freeze.json")
    for name, digest in freeze["evidence_sha256"].items():
        if sha(RUN / name) != digest:
            raise ValueError(f"Inference evidence changed: {name}")
    verify_prepared(manifest["config"], manifest["source_sha256"])
    return manifest, freeze


def pair_analysis(first, second):
    result = stage_pair_counts(first, second)
    result["all_feature_disagreement_days"] = sum(value for key, value in result["transitions"].items()
                                                   if key.split("->")[0] != key.split("->")[1])
    return result


def replay():
    manifest, freeze = verify_inference()
    public = RUN / "public"
    records = read_json(public / "source/source-manifest.json")["cases"]
    entries = {r["case_id"]: r for r in read_json(public / "source/activity-manifest.json")["cases"]}
    summary = read_json(public / "results.json")
    results = {r["request_id"]: r for r in summary["results"]}
    packets = read_json(public / "packets.json")
    if len(packets) != 36 or len(results) != 48:
        raise ValueError("Complete paired packet required")
    old_public = ROOT / manifest["config"]["source_axis_run"] / "public"
    old_packets = {p["case_id"]: p for p in read_json(old_public / "packets.json")}
    for name, audit in read_json(public / "input-image-audits.json").items():
        if sha(public / name) != audit["sha256"] or sha(ROOT / audit["previous_path"]) != audit["sha256"]:
            raise ValueError("Seven-image input differs from old day-index figures")
    planned = []
    active_number = 0
    for record in records:
        if record["case_id"] not in old_packets:
            continue
        active_number += 1
        previous = old_packets[record["case_id"]]
        for condition, repetition in request_order(active_number):
            planned.append({"case_id": record["case_id"], "condition": condition, "repetition": repetition,
                            "request_id": f"{record['case_id']}-{condition}-r{repetition}",
                            "activity_sha256": previous["activity_sha256"], "images": previous["logical_images"],
                            "logical_images": previous["logical_images"],
                            "prompt": evidence_prompt(previous["prompt"], condition)})
    if planned != packets:
        raise ValueError("Packet order, prompt intervention or image association changed")
    packets_by_id = {p["request_id"]: p for p in packets}
    arms = {a: {"native_activity_features": Counter(), "final_activity_features": Counter(),
                "suppressed_support": Counter(), "statuses": Counter()} for a in ARMS}
    signs = {a: {str(w): Counter() for w in (31, 61, 91)} for a in ARMS}
    cases, csv_by_request, costs = [], {}, {}
    for record in records:
        case, activity_review, rows, digest, motions = load_case(public, record, entries)
        native, final, case_results, requests = {}, {}, {}, {}
        for arm in ARMS:
            request_id = case.case_id + "-" + arm
            saved = results[request_id]
            packet = packets_by_id.get(request_id)
            if packet:
                native[arm], final[arm], suppressed, status, error, requests[arm] = replay_request(
                    public, packet, case, activity_review, rows, motions[61])
                receipt = requests[arm]["response"]
                costs.setdefault(arm, Counter()).update(calls=1, total_tokens=receipt.get("usage", {}).get("total_tokens", 0),
                                                      service_seconds=receipt["seconds"], **{str(receipt.get("finish_reason")): 1})
            else:
                if any(r["activity_label"] == 1 for r in rows):
                    raise ValueError("Active case missing planned request")
                native[arm], final[arm] = unclassified_rows(rows), unclassified_rows(rows)
                suppressed, status, error, requests[arm] = {}, "skipped_no_confirmed_activity", None, None
            counts = feature_counts(native[arm], final[arm], suppressed)
            if saved["status"] != status or saved["error"] != error or any(saved[k] != counts[k] for k in counts):
                raise ValueError("Compilation status/counts did not replay")
            for mode, output in (("native", native[arm]), ("daily", final[arm])):
                if output != read_json(public / mode / (request_id + ".json")):
                    raise ValueError("Stage output did not replay")
            for key in counts:
                arms[arm][key].update(counts[key])
            arms[arm]["statuses"].update([status])
            for window, motion in motions.items():
                signs[arm][str(window)].update(sign_counts(final[arm], motion))
            csv_by_request[request_id] = [{"condition": saved["condition"], "repetition": saved["repetition"], **r} for r in final[arm]]
            case_results[arm] = saved
        cases.append({"case_id": case.case_id, "active_days": sum(r["activity_label"] == 1 for r in rows),
                      "results": case_results, "requests": requests,
                      "pairs": {f"{a}_vs_{b}": pair_analysis(final[a], final[b]) for a, b in PAIRS},
                      "plots": {m: [f"presentation/{case.case_id}-{m}-{i}.png" for i in range(4)] for m in ("native", "final")},
                      "case": case, "record": record, "native": native, "final": final})
    expected_csv = [r for result in summary["results"] for r in csv_by_request[result["request_id"]]]
    with (public / "daily-stage-evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_csv = list(csv.DictReader(stream))
    if len(saved_csv) != 52560 or saved_csv != [{k: str(v) for k, v in r.items()} for r in expected_csv]:
        raise ValueError("CSV did not replay")
    if len(list((public / "requests").glob("*.started.json"))) != 36 or len(list((public / "requests").glob("*.response.json"))) != 36:
        raise ValueError("Actual request budget differs")
    if summary["calls"] != 36 or summary["total_tokens"] != sum(c["total_tokens"] for c in costs.values()):
        raise ValueError("Aggregate request counts/costs changed")
    pairs = {}
    common_success = [c for c in cases if c["active_days"] and all(c["results"][a]["status"] == "success" for a in ARMS)]
    sensitivity = {"meaning": "Response-selected descriptive subset, NOT accuracy or primary sample",
                   "common_success_cases": [c["case_id"] for c in common_success], "pairs": {}}
    for a, b in PAIRS:
        name = f"{a}_vs_{b}"
        transitions = Counter()
        subset_transitions = Counter()
        for case in cases:
            transitions.update(case["pairs"][name]["transitions"])
        for case in common_success:
            subset_transitions.update(case["pairs"][name]["transitions"])
        pairs[name] = {
            "active_days": sum(transitions.values()),
            "all_feature_disagreement_days": sum(v for k, v in transitions.items() if k.split("->")[0] != k.split("->")[1]),
            "common_classified_days": sum(c["pairs"][name]["common_classified_days"] for c in cases),
            "common_class_disagreement_days": sum(c["pairs"][name]["common_class_disagreement_days"] for c in cases),
            "transitions": dict(transitions),
        }
        sensitivity["pairs"][name] = {"active_days": sum(subset_transitions.values()),
                                      "all_feature_disagreement_days": sum(v for k, v in subset_transitions.items() if k.split("->")[0] != k.split("->")[1])}
    analysis = {"arms": arms, "pairs": pairs, "sign_counts": signs, "costs": costs, "sensitivity": sensitivity,
                "csv_rows_replayed": 52560, "request_bodies_replayed": 36, "image_associations_replayed": 252,
                "inference_files_verified": len(freeze["evidence_sha256"]), "performance_scores": None,
                "independent_stage_reference": False, "meaning": "Agreement, abstention, support and diagnostics, NOT accuracy"}
    return summary, analysis, cases


def analyze(findings_path, verify_only):
    summary, analysis, cases = replay()
    public = RUN / "public"
    if verify_only:
        delivery = read_json(RUN / "presentation-audit.json")
        for name, digest in delivery["display_sha256"].items():
            if sha(RUN / name) != digest:
                raise ValueError("Presentation changed")
        for name, audit in read_json(public / "presentation/plot-audits.json").items():
            if sha(public / name) != audit["sha256"]:
                raise ValueError("Presentation figure changed")
        if analysis != read_json(public / "analysis.json"):
            raise ValueError("Presentation analysis did not replay")
        print(json.dumps({k: v for k, v in analysis.items() if k not in {"arms", "pairs", "sign_counts", "costs", "sensitivity"}}))
        return
    plot = read_json(ROOT / "artifacts/landslide-background-2026-10-08/batch-v1/public/protocol.json")["plot"]
    (public / "presentation").mkdir(exist_ok=True)
    audits = {}
    for item in cases:
        for mode in ("native", "final"):
            for i, view in enumerate(item["record"]["views"]):
                name = item["plots"][mode][i]
                audit = render_evidence_comparison(item["case"], view, item[mode], public / name, plot)
                if audit["xlim"] != view["audit"]["panels"][0]["xlim"] or audit["ylim_NEU"] != [p["ylim"] for p in view["audit"]["panels"]]:
                    raise ValueError("Frozen raw presentation axes changed")
                audits[name] = {**audit, "sha256": sha(public / name)}
        for key in ("case", "record", "native", "final"):
            del item[key]
    findings = read_json(findings_path) if findings_path else []
    if any(f["case_id"] not in {c["case_id"] for c in cases} or not f["limit"] for f in findings):
        raise ValueError("Author findings require a known case and explicit limit")
    save_display_json(public / "analysis.json", analysis)
    save_display_json(public / "presentation/plot-audits.json", audits)
    encoded = json.dumps({"summary": summary, "analysis": analysis, "cases": cases, "findings": findings},
                         ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "scripts/landslide_stage_evidence_page.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(template.replace("__PAGE_DATA__", encoded), encoding="utf-8")
    verify_inference()
    display = [public / "index.html", public / "analysis.json", public / "presentation/plot-audits.json"]
    if findings_path:
        display.append(findings_path)
    save_display_json(RUN / "presentation-audit.json", {"inference_unchanged": True,
                                                       "display_sha256": {p.relative_to(RUN).as_posix(): sha(p) for p in display}})
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "analyze", "verify"))
    parser.add_argument("--findings", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        if args.findings:
            parser.error("Findings are post-inference only")
        run_inference()
    else:
        analyze(args.findings.resolve() if args.findings else None, args.mode == "verify")


if __name__ == "__main__":
    main()
