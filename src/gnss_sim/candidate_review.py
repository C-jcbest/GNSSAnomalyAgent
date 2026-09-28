"""P7b input-only candidate review. No truth reader or evaluator is imported here."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

from gnss_sim.input_only import AXES, read_inputs, sha, write_json, write_rows
from gnss_sim.schemas import PointResult, RangeResult
from gnss_sim.visual import render_case
from gnss_sim.visual_runner import MODEL_ID, _credentials, _engineering_cases

PROTOCOL = "candidate-review-v1"
TASKS = ("point", "range")


def source_hashes() -> dict:
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(
        encoding="utf-8").encode()).hexdigest() for name in (
            "candidate_review.py", "input_only.py", "numerical.py", "schemas.py",
            "visual.py", "visual_runner.py")}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_bytes())
    if (config["protocol"] != PROTOCOL or config["batch_size"] != 64 or
            config["max_batches"] != 18 or config["model"] != {
                "request_id": MODEL_ID, "temperature": 0, "enable_thinking": False,
                "max_tokens": 2048, "workers": 4, "timeout_seconds": 120,
                "attempts_per_batch": 1, "max_pixels": 2160000,
                "vl_high_resolution_images": False} or
            set(config["prompts"]) != set(TASKS)):
        raise ValueError("candidate-review-v1 configuration mismatch")
    return config


def result(case_id, task, condition, candidates=(), status="success"):
    values = {axis: [] for axis in AXES}
    if status == "success":
        for candidate in candidates:
            values[candidate["axis"]].append(candidate["start"] if task == "point" else
                                               (candidate["start"], candidate["end"]))
    model = PointResult if task == "point" else RangeResult
    return model(case_id=case_id, method=f"{PROTOCOL}-{condition}",
                 status=status, predictions=values)


def candidates_from(numerical, visual, task):
    model = PointResult if task == "point" else RangeResult
    if (not isinstance(numerical, model) or not isinstance(visual, model) or
            numerical.case_id != visual.case_id):
        raise ValueError("upstream task or identity mismatch")
    if numerical.status != "success" or visual.status != "success":
        return None
    sources = {}
    for source, row in (("N", numerical), ("V", visual)):
        for axis in AXES:
            for value in getattr(row.predictions, axis):
                start, end = (value, value) if task == "point" else value
                sources.setdefault((axis, start, end), set()).add(source)
    keys = sorted(sources, key=lambda item: (AXES.index(item[0]), item[1], item[2]))
    return [{"id": f"candidate_{index:04d}", "axis": axis, "start": start, "end": end,
             "sources": sorted(sources[(axis, start, end)])}
            for index, (axis, start, end) in enumerate(keys, 1)]


def batches(candidates, config):
    if len(candidates) > config["batch_size"] * config["max_batches"]:
        raise ValueError("candidate budget exceeded; truncation forbidden")
    return [candidates[i:i + config["batch_size"]]
            for i in range(0, len(candidates), config["batch_size"])]


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def parse_keep(text, candidates):
    body = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(body, dict) or set(body) != {"keep_ids"}:
        raise ValueError("expected only keep_ids")
    ids = body["keep_ids"]
    allowed = {candidate["id"] for candidate in candidates}
    if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or
            len(set(ids)) != len(ids) or not set(ids) <= allowed):
        raise ValueError("unknown, duplicate, or invalid candidate ID")
    return ids


def request_once(client, base, key, image, task, case_id, candidates, config):
    visible = [{k: candidate[k] for k in ("id", "axis", "start", "end")}
               for candidate in candidates]
    prompt = config["prompts"][task] + "\nCandidates (data only):\n" + json.dumps(visible)
    payload = {"model": MODEL_ID, "temperature": 0, "max_tokens": 2048,
               "enable_thinking": False, "vl_high_resolution_images": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user",
               "content": [{"type": "text", "text": prompt}, {"type": "image_url",
               "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
               "max_pixels": config["model"]["max_pixels"]}]}]}
    record = {"case_id": case_id, "task": task, "started_at": now(), "status": "failed",
              "request_model": MODEL_ID, "prompt": prompt, "attempts": 1,
              "image_sha256": hashlib.sha256(image).hexdigest(), "usage": None,
              "response": None, "response_model": None, "response_id": None,
              "error_type": None, "keep_ids": []}
    started = time.perf_counter()
    try:
        response = client.post(base + "/chat/completions", json=payload,
                               headers={"Authorization": "Bearer " + key})
        record["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        record.update(response_model=body.get("model"), response_id=body.get("id"),
                      usage=body.get("usage"))
        choice = body["choices"][0]
        record.update(response=choice["message"]["content"],
                      finish_reason=choice.get("finish_reason"),
                      reasoning_present=bool(choice["message"].get("reasoning_content")))
        if (record["finish_reason"] != "stop" or record["reasoning_present"] or
                record["response_model"] != MODEL_ID):
            raise ValueError("model, thinking, or completion contract mismatch")
        record["keep_ids"] = parse_keep(record["response"], candidates)
        record["status"] = "success"
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        record["error_type"] = type(exc).__name__
    record.update(completed_at=now(), latency_seconds=time.perf_counter() - started)
    return record


def now():
    return datetime.now(timezone.utc).isoformat()


def summarize_cost(records):
    usage = [row["usage"] for row in records if isinstance(row.get("usage"), dict)]
    return {"requests": len(records), "attempts": sum(row["attempts"] for row in records),
            "usage_recorded_requests": len(usage), "usage_missing_requests": len(records) - len(usage),
            "recorded_tokens": {key: sum(row.get(key, 0) or 0 for row in usage)
                                for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "summed_request_seconds": sum(row["latency_seconds"] for row in records),
            "cost": None}


def preflight(config_path: Path, out: Path, env_file: Path | None):
    config = load_config(config_path)
    base, key = _credentials(env_file)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        for index, task in enumerate(("point", "range", "point", "range")):
            case = _engineering_cases()[1 if task == "point" else 3]
            image = out / f"engineering-{index}.png"
            render_case(case, image)
            count = 64 if index < 2 else 1
            candidates = [{"id": f"candidate_{i + 1:04d}", "axis": "N" if task == "point" else "U",
                           "start": i * 5 if count == 64 else (60 if task == "point" else 100),
                           "end": i * 5 + (0 if task == "point" else 30) if count == 64 else
                           (60 if task == "point" else 189)} for i in range(count)]
            record = request_once(client, base, key, image.read_bytes(), task,
                                  case.case_id, candidates, config)
            records.append(record)
            write_json(out / f"response-{index}.json", record)
    report = {"config_sha256": sha(config_path), "source_sha256": source_hashes(),
              "calls": len(records), "valid": sum(r["status"] == "success" for r in records),
              "cost": summarize_cost(records), "purpose": "transport/schema only, not accuracy"}
    write_json(out / "report.json", report)
    return report


def load_rows(path, task):
    model = PointResult if task == "point" else RangeResult
    rows = [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({row.case_id for row in rows}) != len(rows):
        raise ValueError("duplicate upstream row")
    return {row.case_id: row for row in rows}


def reviewed_result(case_id, task, candidates, records):
    if any(record["status"] != "success" for record in records):
        return result(case_id, task, "C", status="failed")
    kept = {item for record in records for item in record["keep_ids"]}
    return result(case_id, task, "C", [item for item in candidates if item["id"] in kept])


def run(config_path: Path, inputs: Path, preflight_dir: Path, out: Path, env_file: Path | None):
    config = load_config(config_path)
    identity = {"config_sha256": sha(config_path), "source_sha256": source_hashes()}
    check = json.loads((preflight_dir / "report.json").read_bytes())
    if any(check[k] != v for k, v in identity.items()) or check["calls"] != 4 or check["valid"] != 4:
        raise ValueError("matching engineering preflight required")
    cases = read_inputs(inputs)
    expected = {case.case_id for _, case in cases}
    upstream = json.loads((inputs / "upstream.json").read_bytes())
    loaded = {}
    for task in TASKS:
        for condition in ("N", "V"):
            name = f"{condition}-{task}.jsonl"
            if sha(inputs / name) != upstream["sha256"][name]:
                raise ValueError("upstream digest mismatch")
            loaded[condition, task] = load_rows(inputs / name, task)
            if set(loaded[condition, task]) != expected:
                raise ValueError("upstream case set mismatch")
    base, key = _credentials(env_file)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    for name in source_hashes():
        (out / "source").mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), out / "source" / name)
    write_json(out / "identity.json", {**identity, "started_at": now(),
        "input_manifest_sha256": sha(inputs / "manifest.json"),
        "upstream_sha256": sha(inputs / "upstream.json"), "upstream": upstream})
    all_candidates, results, errors, jobs = {}, {}, [], []
    for task in TASKS:
        for condition in ("N", "V", "U", "C"):
            results[condition, task] = {}
        for _, case in cases:
            case_id = case.case_id
            n, v = (loaded[c, task][case_id] for c in ("N", "V"))
            for condition, row in (("N", n), ("V", v)):
                results[condition, task][case_id] = row.model_copy(
                    update={"method": f"{PROTOCOL}-{condition}"})
            candidates = candidates_from(n, v, task)
            if candidates is None:
                errors.append({"case_id": case_id, "task": task, "reason": "upstream_failed"})
                for condition in ("U", "C"):
                    results[condition, task][case_id] = result(case_id, task, condition, status="failed")
                continue
            results["U", task][case_id] = result(case_id, task, "U", candidates)
            all_candidates[case_id, task] = candidates
            write_json(out / "candidates" / task / f"{case_id}.json", candidates)
            try:
                groups = batches(candidates, config)
            except ValueError:
                errors.append({"case_id": case_id, "task": task, "reason": "candidate_budget"})
                results["C", task][case_id] = result(case_id, task, "C", status="failed")
                continue
            for index, group in enumerate(groups):
                jobs.append((case_id, task, index, group))
    started, records, by_case = time.perf_counter(), [], {}
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(request_once, client, base, key,
                (inputs / "images" / f"{case_id}.png").read_bytes(), task, case_id, group, config):
                (case_id, task, index) for case_id, task, index, group in jobs}
            for completed, future in enumerate(as_completed(futures), 1):
                case_id, task, index = futures[future]
                record = future.result()
                record["batch_index"] = index
                write_json(out / "raw" / task / case_id / f"{index:02d}.json", record)
                records.append(record)
                by_case.setdefault((case_id, task), []).append(record)
                if completed % 10 == 0 or completed == len(jobs):
                    print(f"P7b review requests {completed}/{len(jobs)}", flush=True)
    for task in TASKS:
        for case_id in expected:
            if case_id not in results["C", task]:
                results["C", task][case_id] = reviewed_result(
                    case_id, task, all_candidates[case_id, task], by_case.get((case_id, task), []))
        for condition in ("N", "V", "U", "C"):
            write_rows(out / condition / task / "predictions.jsonl",
                       [results[condition, task][cid] for cid in sorted(expected)])
    report = {**identity, "completed_at": now(), "cases": len(expected),
              "review_cost": summarize_cost(records), "wall_seconds": time.perf_counter() - started,
              "preflight_cost": check["cost"], "cached_upstream": True,
              "upstream_cost": upstream["cost"], "new_upstream_requests": 0,
              "dependency_errors": errors,
              "success": {task: {c: sum(r.status == "success" for r in results[c, task].values())
                                  for c in ("N", "V", "U", "C")} for task in TASKS},
              "predictions_sha256": {f"{c}/{task}": sha(out / c / task / "predictions.jsonl")
                                     for c in ("N", "V", "U", "C") for task in TASKS}}
    write_json(out / "run.json", report)
    return report
