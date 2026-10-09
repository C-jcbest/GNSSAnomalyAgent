"""Replay horizontal-presentation inference and paired variability without accuracy claims."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_landslide_frozen_stage_heads import read_json, verify_hashes  # noqa: E402
from build_landslide_raw_adjudication import verify as verify_activity_run  # noqa: E402

from gnss_sim.artifacts import sha  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    compile_stage_answer,
    stage_pair_counts,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_stage_axis_experiment import (  # noqa: E402
    render_auxiliary_axis_pair,
    render_stage_axis_comparison,
)
from gnss_sim.visual import _unique_object  # noqa: E402

ARMS = ("dates-r1", "day_index-r1", "dates-r2", "day_index-r2")
PAIRS = (("dates-r1", "day_index-r1"), ("dates-r2", "day_index-r2"),
         ("dates-r1", "dates-r2"), ("day_index-r1", "day_index-r2"))


def verify_run(run):
    manifest = read_json(run / "manifest.json")
    freeze = read_json(run / "inference-freeze.json")
    if sha(run / "inference-freeze.json") != manifest["inference_freeze_sha256"]:
        raise ValueError("Inference freeze changed")
    for name, digest in freeze["evidence_sha256"].items():
        if sha(run / name) != digest:
            raise ValueError(f"Inference evidence changed: {name}")
    for name, digest in manifest["source_sha256"].items():
        if sha(ROOT / name) != digest or sha(run / "source-code" / name) != digest:
            raise ValueError(f"Execution source changed: {name}")
    ledger = read_json(run / "run-ledger-start.json")
    old = ROOT / manifest["config"]["source_run"]
    if sha(old / "manifest.json") != ledger["source_manifest_sha256"]:
        raise ValueError("Source run manifest changed")
    if sha(old / "inference-freeze.json") != ledger["source_inference_freeze_sha256"]:
        raise ValueError("Source run freeze changed")
    verify_hashes(old, read_json(old / "manifest.json"))
    verify_activity_run(ROOT / manifest["config"]["activity_run"])
    for name, digest in ledger["prepared_inputs_sha256"].items():
        if sha(run / "public" / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    return manifest, freeze


def replay_request(public, packet, case, review, rows, motion):
    request_id = packet["request_id"]
    started = read_json(public / f"requests/{request_id}.started.json")
    response = read_json(public / f"requests/{request_id}.response.json")
    if started["prompt"] != packet["prompt"]:
        raise ValueError("Prompt changed")
    expected = {Path(name).name: sha(public / name) for name in packet["images"]}
    if list(expected.items()) != list(started["image_sha256"].items()):
        raise ValueError("Actual image content/order changed")
    content = [{"type": "text", "text": started["prompt"]}]
    for name in packet["images"]:
        encoded = base64.b64encode((public / name).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
    body = {k: started[k] for k in ("model", "temperature", "top_p", "max_tokens")}
    body.update(messages=[{"role": "user", "content": content}], enable_thinking=False,
                response_format={"type": "json_object"})
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != started["request_sha256"]:
        raise ValueError("Actual request body did not replay")
    if response["request_sha256"] != started["request_sha256"]:
        raise ValueError("Response not associated with this request")
    error = None
    status = "success"
    if response["status"] == "returned_json":
        try:
            payload = json.loads(response["output"], object_pairs_hook=_unique_object)
            stage_review, native, final, suppressed = compile_stage_answer(
                payload, case, review, rows, packet["activity_sha256"], motion)
        except (ValueError, TypeError, KeyError) as failure:
            error = {"type": type(failure).__name__, "message": str(failure)[:400]}
            status = "failed_unknown_activity_stages"
            native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
        else:
            if stage_review != read_json(public / f"stage-reviews/{request_id}.json"):
                raise ValueError("Stage draft changed")
    else:
        error = {"type": "RuntimeError", "message": f"Visual request failed: {request_id}; see receipt"}
        status = "failed_unknown_activity_stages"
        native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
    request = {"started": started, "response": response, "images": packet["images"],
               "receipt_summary": {k: response.get(k) for k in (
                   "request_id", "model", "returned_model", "temperature", "top_p", "max_tokens", "no_retries",
                   "status", "finish_reason", "usage", "seconds", "request_sha256")}}
    return native, final, suppressed, status, error, request


def sign_counts(rows, motion):
    counts = Counter()
    for row, value in zip(rows, motion.tangential_acceleration_mm_day2, strict=True):
        if row["feature"] not in {"acceleration", "deceleration"}:
            continue
        if value is None or not np.isfinite(value):
            counts["no_finite_tangent"] += 1
            continue
        counts["finite_A_D_days"] += 1
        if row["feature"] == "acceleration" and value < 0:
            counts["A_negative_tangent"] += 1
        elif row["feature"] == "deceleration" and value > 0:
            counts["D_positive_tangent"] += 1
        elif value == 0:
            counts["zero_tangent"] += 1
    return counts


def replay(run):
    manifest, freeze = verify_run(run)
    public = run / "public"
    source = public / "source"
    records = read_json(source / "source-manifest.json")["cases"]
    author = ROOT / manifest["config"]["activity_run"]
    entries = {r["case_id"]: r for r in read_json(author / "public/manifest.json")["cases"]}
    activity_freeze = read_json(source / "activity-freeze.json")
    packets = {p["request_id"]: p for p in read_json(public / "packets.json")}
    old_packets = {p["case_id"]: p for p in read_json(source / "packets.json")}
    summary = read_json(public / "results.json")
    results = {r["request_id"]: r for r in summary["results"]}
    image_audits = read_json(public / "image-audits.json")
    if len(packets) != 36 or len(results) != 48:
        raise ValueError("Complete paired two-repeat packet required")
    arms = {arm: {"native_activity_features": Counter(), "final_activity_features": Counter(),
                  "suppressed_support": Counter(), "statuses": Counter()} for arm in ARMS}
    signs = {arm: {str(w): Counter() for w in (31, 61, 91)} for arm in ARMS}
    pair_totals = {f"{a}_vs_{b}": {"active_days": 0, "common_classified_days": 0,
                                    "common_same_class_days": 0, "common_class_disagreement_days": 0,
                                    "all_feature_disagreement_days": 0, "transitions": Counter()} for a, b in PAIRS}
    cases = []
    csv_by_request = {}
    for record in records:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((source / record["input"]).read_bytes())
        entry = entries[case_id]
        review = read_json(source / "activity-source" / entry["review"])
        rows = read_json(source / "activity-source" / entry["daily"])
        activity = validate_activity(case, review, rows, activity_freeze["reviews"][case_id]["activity_sha256"])
        motions = {w: LandslideDiagnostics.model_validate_json(
            (source / record["diagnostics"][str(w)]["data"]).read_bytes()) for w in (31, 61, 91)}
        for window, motion in motions.items():
            original, changed, audit = render_auxiliary_axis_pair(case, motion, record["views"][0]["audit"]["tick_audit"]["ticks"])
            saved = image_audits[f"{case_id}-{window}"]
            if hashlib.sha256(original).hexdigest() != saved["dates_sha256"]:
                raise ValueError("Original PNG replay changed")
            if hashlib.sha256(changed).hexdigest() != saved["day_index_sha256"]:
                raise ValueError("Day-index PNG replay changed")
            if any(audit[k] != saved[k] for k in audit):
                raise ValueError("Image geometry audit changed")
        native, final, case_results, requests = {}, {}, {}, {}
        for arm in ARMS:
            request_id = case_id + "-" + arm
            result = results[request_id]
            packet = packets.get(request_id)
            if packet:
                if packet["prompt"] != old_packets[case_id]["prompt"] or packet["logical_images"] != old_packets[case_id]["images"]:
                    raise ValueError("Prompt or logical seven-image order differs from historical input")
                native[arm], final[arm], suppressed, status, error, requests[arm] = replay_request(
                    public, packet, case, review, rows, motions[61])
            else:
                if case_id in old_packets:
                    raise ValueError("Missing planned active-case request")
                native[arm], final[arm] = unclassified_rows(rows), unclassified_rows(rows)
                suppressed, status, error, requests[arm] = {}, "skipped_no_confirmed_activity", None, None
            if result["status"] != status or result["error"] != error:
                raise ValueError("Output failure status did not replay")
            for mode, labels in (("native", native[arm]), ("daily", final[arm])):
                if labels != read_json(public / f"{mode}/{request_id}.json"):
                    raise ValueError("Stage output did not replay")
            counts = {"native_activity_features": dict(Counter(r["feature"] for r in native[arm] if r["activity_label"] == 1)),
                      "final_activity_features": dict(Counter(r["feature"] for r in final[arm] if r["activity_label"] == 1)),
                      "suppressed_support": suppressed}
            if any(counts[k] != result[k] for k in counts):
                raise ValueError("Counts changed")
            for k in counts:
                arms[arm][k].update(counts[k])
            arms[arm]["statuses"].update([status])
            for window, motion in motions.items():
                signs[arm][str(window)].update(sign_counts(final[arm], motion))
            csv_by_request[request_id] = [{"condition": result["condition"], "repetition": result["repetition"], **r}
                                          for r in final[arm]]
            case_results[arm] = result
        pairs = {}
        for a, b in PAIRS:
            pair = stage_pair_counts(final[a], final[b])
            pair["all_feature_disagreement_days"] = sum(n for k, n in pair["transitions"].items()
                                                         if k.split("->")[0] != k.split("->")[1])
            name = f"{a}_vs_{b}"
            pairs[name] = pair
            for k in pair_totals[name]:
                if k == "transitions":
                    pair_totals[name][k].update(pair[k])
                else:
                    pair_totals[name][k] += pair[k]
        plots = {mode: [f"presentation/{case_id}-{mode}-{i}.png" for i in range(4)] for mode in ("native", "final")}
        cases.append({"case_id": case_id, "active_days": int(np.count_nonzero(activity == 1)),
                      "results": case_results, "pairs": pairs, "requests": requests, "plots": plots,
                      "case": case, "record": record, "native": native, "final": final})
    expected_csv = [r for result in summary["results"] for r in csv_by_request[result["request_id"]]]
    with (public / "daily-stage-axis.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_csv = list(csv.DictReader(stream))
    if len(saved_csv) != 52560 or saved_csv != [{k: str(v) for k, v in r.items()} for r in expected_csv]:
        raise ValueError("Combined CSV did not replay")
    receipts = list((public / "requests").glob("*.response.json"))
    if len(receipts) != 36 or len(list((public / "requests").glob("*.started.json"))) != 36:
        raise ValueError("Actual request count differs from fixed budget")
    analysis = {"arms": arms, "pairs": pair_totals, "sign_counts": signs, "csv_rows_replayed": 52560,
                "request_bodies_replayed": 36, "image_associations_replayed": 252,
                "auxiliary_png_pairs_replayed": 36, "inference_files_verified": len(freeze["evidence_sha256"]),
                "performance_scores": None, "independent_stage_reference": False,
                "meaning": "Prediction agreement, abstention and sign diagnostics; NOT accuracy"}
    return manifest, summary, analysis, cases


def save_display_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read-only replay, with no plots/page rewritten")
    parser.add_argument("--findings", type=Path, help="Separate post-inference author observations")
    args = parser.parse_args()
    run = ROOT / "artifacts/landslide-stage-axis-2026-10-08/run-v1"
    public = run / "public"
    manifest, summary, analysis, cases = replay(run)
    if args.verify:
        for name, audit in read_json(public / "presentation/plot-audits.json").items():
            if sha(public / name) != audit["sha256"]:
                raise ValueError("Presentation plot changed")
        delivery = read_json(run / "presentation-audit.json")
        for name, digest in delivery["display_sha256"].items():
            if sha(run / name) != digest:
                raise ValueError("Presentation file changed")
        if analysis != read_json(public / "analysis.json"):
            raise ValueError("Presentation analysis did not replay")
        print(json.dumps({k: v for k, v in analysis.items() if k not in {"arms", "pairs", "sign_counts"}}, ensure_ascii=False))
        return
    plot = read_json(ROOT / "artifacts/landslide-background-2026-10-08/batch-v1/public/protocol.json")["plot"]
    presentation = public / "presentation"
    presentation.mkdir(exist_ok=True)
    plot_audits = {}
    for item in cases:
        for mode in ("native", "final"):
            for index, view in enumerate(item["record"]["views"]):
                name = item["plots"][mode][index]
                target = public / name
                audit = render_stage_axis_comparison(item["case"], view, item[mode], target, plot)
                if audit["xlim"] != view["audit"]["panels"][0]["xlim"]:
                    raise ValueError("Presentation x-axis changed")
                if audit["ylim_NEU"] != [p["ylim"] for p in view["audit"]["panels"]]:
                    raise ValueError("Presentation y-axes changed")
                plot_audits[name] = {**audit, "sha256": sha(target)}
        for key in ("case", "record", "native", "final"):
            del item[key]
    findings = read_json(args.findings) if args.findings else []
    for finding in findings:
        if finding["case_id"] not in {c["case_id"] for c in cases} or not finding["limit"]:
            raise ValueError("Author finding requires a known case and explicit limit")
    save_display_json(public / "analysis.json", analysis)
    save_display_json(presentation / "plot-audits.json", plot_audits)
    data = {"summary": summary, "analysis": analysis, "cases": cases, "findings": findings}
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "scripts/landslide_stage_axis_page.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(template.replace("__PAGE_DATA__", encoded), encoding="utf-8")
    verify_run(run)
    display = [public / "index.html", public / "analysis.json", presentation / "plot-audits.json"]
    if args.findings:
        display.append(args.findings)
    save_display_json(run / "presentation-audit.json", {"inference_unchanged": True,
                                                        "display_sha256": {p.relative_to(run).as_posix(): sha(p) for p in display}})
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
