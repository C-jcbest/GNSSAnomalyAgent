"""Read-only evidence assembly for the saved landslide visual experiments."""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction, runs
from gnss_sim.landslide_visual import interval_mask, parse_activity, parse_stages
from gnss_sim.visual import _unique_object

ACTIVITY_REQUEST_KINDS = {"locate", "locate-global", "locate-local"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inclusive_spans(mask):
    return [[start, stop - 1] for start, stop in runs(np.asarray(mask, dtype=bool))]


def checked_asset(path, out, expected=None):
    """Use content-addressed copies so equal filenames cannot mix experiment rounds."""
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Image SHA256 mismatch: {path}")
    relative = Path("assets") / (digest + path.suffix)
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256(destination) != digest:
        shutil.copyfile(path, destination)
    return {"name": path.name, "url": relative.as_posix(), "sha256": digest,
            "request_hash_verified": expected is not None}


def request_evidence(started, image_root, out):
    images = []
    content = [{"type": "text", "text": started["prompt"]}]
    for name, expected in started["image_sha256"].items():
        if Path(name).name != name:
            raise ValueError("An image receipt must contain a filename, not a path")
        path = image_root / name
        images.append(checked_asset(path, out, expected))
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
    # These two flags come from the inspected transport, not explicit receipt fields.
    # Matching the entire request digest verifies this reconstruction of the body.
    payload = {key: started[key] for key in ("model", "temperature", "top_p", "max_tokens")}
    payload.update(messages=[{"role": "user", "content": content}], enable_thinking=False,
                   response_format={"type": "json_object"})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != started["request_sha256"]:
        raise ValueError(f"Request reconstruction SHA256 mismatch: {started['request_id']}")
    return images


def contract_details(payload, gate, observed, dates, gated):
    """Explain interval conflicts without repairing or accepting the model output."""
    findings = []
    previous = None
    for item in payload.get("stages", []):
        if not isinstance(item, list) or len(item) != 3:
            continue
        try:
            mask = interval_mask([item[:2]], len(observed))
        except ValueError:
            continue
        if previous is not None and item[0] <= previous[1]:
            findings.append({"kind": "overlap", "interval": item, "previous": previous,
                             "message": f"闭区间 {previous} 与 {item} 重叠或次序错误；不能共享端点日。"})
        outside = mask & observed & (gate != 1)
        for start, end in inclusive_spans(outside):
            findings.append({"kind": "outside_gate" if gated else "ungated_outside",
                             "interval": [start, end], "dates": [dates[start], dates[end]],
                             "message": f"第 {start}–{end} 日（{dates[start]} 至 {dates[end]}）有观测但不在确认活动范围。"
                             + ("按契约拒绝本次阶段输出。" if gated else "本项解除门控，允许返回；不代表确认了活动。")})
        previous = item
    return findings


def activity_conflict_details(payload, dates):
    findings = []
    if "activity" not in payload or "uncertain" not in payload:
        return findings
    try:
        active = interval_mask(payload["activity"], len(dates))
        uncertain = interval_mask(payload["uncertain"], len(dates))
    except ValueError:
        return findings
    for start, end in inclusive_spans(active & uncertain):
        findings.append({"kind": "activity_uncertain_overlap", "interval": [start, end],
                         "dates": [dates[start], dates[end]],
                         "message": f"第 {start}–{end} 日（{dates[start]} 至 {dates[end]}）同时属于确认活动和 uncertain；两类闭区间必须互斥。"})
    return findings


def replay_response(started, response, observed, dates, activity, kind, saved):
    """Reuse the experiment parsers; transport success and contract success stay separate."""
    result = {"status": "not_checked", "error": None, "details": [], "saved_matches": None}
    if response.get("status") != "returned_json":
        result.update(status="transport_failed", error=response.get("provider_error", response.get("error_type")))
        return result
    gated = kind != "visual_model_ungated"
    try:
        payload = json.loads(response["output"], object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        if kind in ACTIVITY_REQUEST_KINDS:
            result["details"] = activity_conflict_details(payload, dates)
            actual = parse_activity(payload, observed)
            result["saved_matches"] = bool(np.array_equal(actual, saved["activity"]))
        else:
            if kind == "single":
                if set(payload) != {"activity", "uncertain", "stages"}:
                    raise ValueError("Single-stage output requires exactly activity, uncertain, stages")
                activity = parse_activity({key: payload[key] for key in ("activity", "uncertain")}, observed)
            else:
                prefix = "Confirmed inclusive activity intervals: "
                lines = [line[len(prefix):] for line in started["prompt"].splitlines() if line.startswith(prefix)]
                if len(lines) != 1:
                    raise ValueError("Expected one logged activity gate")
                prompt_gate = interval_mask(json.loads(lines[0]), len(observed))
                if not np.array_equal(prompt_gate, activity == 1):
                    raise ValueError("Logged prompt gate differs from its saved source")
            result["details"] = contract_details(payload, activity, observed, dates, gated)
            stage_payload = {"stages": payload["stages"]} if kind == "single" else payload
            prediction = parse_stages(stage_payload, activity, observed, gated=gated)
            result["saved_matches"] = bool(
                np.array_equal(prediction.activity, saved["activity"])
                and np.array_equal(prediction.stage, saved["stage"]))
        result["status"] = "passed"
    except (ValueError, KeyError, TypeError) as error:
        result.update(status="failed", error=str(error))
    return result


def assemble_request(run_id, image_root, started_path, case, activity, saved, failures, out):
    started = read_json(started_path)
    response_path = started_path.with_name(started_path.name.replace(".started.", ".response."))
    response = read_json(response_path)
    if any(response.get(key) != value for key, value in started.items()):
        raise ValueError(f"Started/response receipts disagree: {started_path.name}")
    request_id = started["request_id"]
    kind = request_id[len(case["case_id"]) + 1:]
    values = np.array([row if row is not None else [np.nan] * 3 for row in case["displacement_mm"]])
    observed = np.isfinite(values).all(axis=1)
    ActivityPrediction(np.array(saved["activity"]), np.array(saved["stage"]), saved["status"],
                       saved["gated"], saved["stage_status"]).validate(len(observed))
    validation = replay_response(started, response, observed, case["dates"], activity, kind, saved)
    recorded_error = failures.get(request_id)
    validation["recorded_error"] = recorded_error
    validation["historical_status_matches"] = (recorded_error is None) == (validation["status"] == "passed")
    if not validation["historical_status_matches"] or validation["saved_matches"] is False:
        raise ValueError(f"Replay differs from the frozen experiment: {run_id}/{request_id}")
    if validation["status"] == "failed":
        expected_status = saved["status"] if kind == "single" or kind in ACTIVITY_REQUEST_KINDS else saved["stage_status"]
        if expected_status != "failed":
            raise ValueError(f"Rejected output was not marked failed: {run_id}/{request_id}")
    logs = out / "receipts" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    # Preserve only the known receipt schema; never copy credentials or provider configuration.
    allowed = {"request_id", "model", "prompt", "image_sha256", "request_sha256", "temperature",
               "top_p", "max_tokens", "no_retries", "http_status", "usage", "returned_model",
               "output", "finish_reason", "status", "seconds", "error_type", "provider_error"}
    for path, receipt in ((started_path, started), (response_path, response)):
        (logs / path.name).write_text(json.dumps({key: value for key, value in receipt.items() if key in allowed},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    return {"id": run_id + "/" + request_id, "request_id": request_id, "run": run_id,
            "case_id": case["case_id"], "kind": kind, "prompt": started["prompt"],
            "images": request_evidence(started, image_root, out), "request_sha256": started["request_sha256"],
            "request_hash_verified": True,
            "parameters": {key: started[key] for key in ("model", "temperature", "top_p", "max_tokens", "no_retries")},
            "response": {key: value for key, value in response.items() if key in allowed and key not in started},
            "validation": validation, "activity": activity.tolist(),
            "accepted": saved,
            "logs": {"started": f"receipts/{run_id}/{started_path.name}",
                     "response": f"receipts/{run_id}/{response_path.name}"}}


def build_trace(numeric_root, visual_root, reference_root, out):
    manifest = read_json(numeric_root / "manifest.json")
    visual_manifest = read_json(visual_root / "visual-manifest.json")
    reference_manifest = read_json(reference_root / "manifest.json")
    localization_root = Path(visual_manifest["localization_source"])
    checks = [
        (numeric_root / "manifest.json", visual_manifest["numeric_manifest_sha256"]),
        (numeric_root / "manifest.json", reference_manifest["numeric_manifest_sha256"]),
        (localization_root / "predictions.json", visual_manifest["localization_predictions_sha256"]),
        (visual_root / "predictions.json", reference_manifest["prior_predictions_sha256"]),
        (reference_root / "reference-activity-input.json", reference_manifest["reference_gate_sha256"]),
        (Path(manifest["labels"]), reference_manifest["reference_labels_sha256"]),
    ]
    for path, expected in checks:
        if sha256(path) != expected:
            raise ValueError(f"Experiment source SHA256 mismatch: {path}")
    cases = {}
    audit = read_json(reference_root / "existing-error-audit.json")
    for cid in visual_manifest["case_ids"]:
        path = Path(manifest["data"]) / "cases" / cid / "input.json"
        if sha256(path) != manifest["input_sha256"][cid]:
            raise ValueError(f"Observation SHA256 mismatch: {cid}")
        cases[cid] = read_json(path)
    roots = {"visual-v2": localization_root, "visual-v3": visual_root, "reference-v1": reference_root}
    predictions = {name: read_json(path / "predictions.json") for name, path in roots.items()}
    reference_activity = read_json(reference_root / "reference-activity-input.json")
    records = []
    for run_id, root in roots.items():
        failures = read_json(root / "failures.json")
        for path in sorted((root / "requests").glob("*.started.json")):
            request_id = path.name.removesuffix(".started.json")
            cid, kind = request_id.split("-", 1)
            method = {"locate": "visual_rule", "single": "single_visual",
                      "reference-stage": "reference_visual_model"}.get(kind, kind)
            saved = predictions[run_id][method][cid]
            if kind == "reference-stage":
                activity = reference_activity[cid]
            elif kind == "numeric_model":
                activity = predictions[run_id]["robust_rule"][cid]["activity"]
            elif kind == "single":
                activity = saved["activity"]
            else:
                activity = predictions[run_id]["visual_rule"][cid]["activity"]
            image_root = visual_root if run_id == "reference-v1" else root
            records.append(assemble_request(run_id, image_root, path, cases[cid],
                                            np.array(activity), saved, failures, out))
    case_evidence = {}
    for cid, case in cases.items():
        review_paths = [visual_root / f"{cid}-stages.png", reference_root / f"{cid}-reference-stages.png"]
        review_paths.extend(sorted(reference_root.glob(f"{cid}-local-*.png")))
        case_evidence[cid] = {
            "dates": case["dates"], "missing": [i for i, row in enumerate(case["displacement_mm"]) if row is None],
            "audit": audit[cid], "review_images": [checked_asset(path, out) for path in review_paths],
            "reference_activity": reference_activity[cid],
            "reference_prediction": predictions["reference-v1"]["reference_visual_model"][cid],
        }
    skipped = [cid for cid in cases if not any(row["case_id"] == cid and row["run"] == "reference-v1" for row in records)]
    for cid in skipped:
        saved = predictions["reference-v1"]["reference_visual_model"][cid]
        if np.any(np.array(reference_activity[cid]) == 1) or np.any(np.array(saved["stage"]) > 0):
            raise ValueError("A missing reference request is only valid for an empty activity gate")
    return {"schema": "landslide-visual-trace-v1", "cases": case_evidence, "requests": records,
            "reference_skipped": skipped, "localization_source": "visual-v2",
            "summary": {"requests": len(records), "contract_failures": sum(row["validation"]["status"] == "failed" for row in records),
                        "images_checked": sum(len(row["images"]) for row in records), "new_model_calls": 0},
            "sources": {name: str(path.resolve()) for name, path in roots.items()}}


def include_local_trace(report, local_root, out):
    """Append actual local-view calls; copied atlases become inputs only in these arms."""
    run = read_json(local_root / "manifest.json")
    if run["status"] != "complete":
        raise ValueError("Only completed local-view experiments can be displayed")
    if sha256(Path(run["visual_root"]) / "predictions.json") != run["fixed_predictions_sha256"]:
        raise ValueError("Local experiment's fixed activity source changed")
    if sha256(local_root / "packets.json") != run["packets_sha256"]:
        raise ValueError("Frozen local-view packets changed")
    predictions = read_json(local_root / "predictions.json")
    audit = read_json(local_root / "error-audit.json")
    method_kinds = {"activity_global": "locate-global", "activity_local": "locate-local",
                    "stage_global": "stage-global", "stage_local": "stage-local"}
    local_cases = {}
    for cid in run["case_ids"]:
        path = local_root / "inputs" / cid / "input.json"
        if sha256(path) != run["input_sha256"][cid]:
            raise ValueError("Local-view observation changed")
        case = read_json(path)
        if case["dates"] != report["cases"][cid]["dates"]:
            raise ValueError("Local-view calendar differs from the comparison")
        for method, kind in method_kinds.items():
            saved = predictions[method][cid]
            started = local_root / "requests" / f"{cid}-{kind}.started.json"
            report["requests"].append(assemble_request("local-v1", local_root, started, case,
                                                       np.array(saved["activity"]), saved,
                                                       read_json(local_root / "failures.json"), out))
        local_cases[cid] = {
            "audit": {kind: audit[method][cid] for method, kind in method_kinds.items()},
            "review_images": [checked_asset(local_root / f"{cid}{suffix}.png", out) for suffix in ("", "-stages")],
            "xgboost_prediction": predictions["visual_xgboost_stage"][cid],
        }
    report["local_experiment"] = {"cases": local_cases, "manifest": run,
                                    "results": read_json(local_root / "results.json")}
    report["sources"]["local-v1"] = str(local_root.resolve())
    report["summary"].update(requests=len(report["requests"]),
        contract_failures=sum(row["validation"]["status"] == "failed" for row in report["requests"]),
        images_checked=sum(len(row["images"]) for row in report["requests"]), new_model_calls=run["calls"])
    return report
