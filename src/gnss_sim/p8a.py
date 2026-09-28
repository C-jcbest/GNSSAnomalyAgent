"""P8a pure-image Range ablation. Inference never opens truth or archived predictions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np

from gnss_sim.input_only import read_inputs, sha, write_json, write_rows
from gnss_sim.schemas import RangeResult
from gnss_sim.visual import AXES, failed_result, parse_result, render_case
from gnss_sim.visual_8k import load_config as load_parent
from gnss_sim.visual_runner import _credentials, _engineering_cases

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "p8a-range-context-v1"
CONDITIONS = ("V0", "V1", "V2")
WINDOWS = ((0, 119), (90, 209), (180, 299), (245, 364))


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    names = ("p8a.py", "visual_8k.py", "visual.py", "visual_runner.py", "schemas.py",
             "input_only.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(
        encoding="utf-8").encode()).hexdigest() for name in names}


def load_config(path):
    config = json.loads(path.read_bytes())
    parent = load_parent()
    if (config["protocol"] != PROTOCOL or config["model"] != parent["model"] or
            config["initial_prompt"] != parent["prompts"]["range"] or
            config["windows"] != [list(w) for w in WINDOWS] or
            config["context_days"] != 30 or config["panels_per_sheet"] != 3 or
            config["max_sheets"] != 4 or config["cases"] != 60):
        raise ValueError("P8a config mismatch")
    return config


def selected_windows(prediction, context=30):
    """Finite grid: max 12 panels, no candidate truncation or label-based cropping."""
    if prediction.status != "success":
        raise ValueError("cannot refine failed initial prediction")
    nonempty = any(getattr(prediction.predictions, axis) for axis in AXES)
    return [(axis, start, end) for axis in AXES for start, end in WINDOWS
            if not nonempty or any(start <= min(364, b + context) and
                                   end >= max(0, a - context)
                                   for a, b in getattr(prediction.predictions, axis))]


def y_limits(values):
    lo, hi = float(min(values)), float(max(values))
    # Match renderer-v1 autoscale: nonsingular expansion then y margin 0.08.
    from matplotlib.transforms import nonsingular
    lo, hi = nonsingular(lo, hi, expander=0.05)
    margin = (hi - lo) * 0.08
    return lo - margin, hi + margin


def render_details(case, prediction, directory):
    from matplotlib import pyplot as plt

    panels = selected_windows(prediction)
    series = np.asarray(case.displacement_mm)
    paths = []
    directory.mkdir(parents=True, exist_ok=True)
    for page, offset in enumerate(range(0, len(panels), 3)):
        path = directory / f"detail_{page + 1}.png"
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                             "axes.linewidth": 0.8, "savefig.facecolor": "white"}):
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=150, facecolor="white")
            try:
                for panel, descriptor in zip(axes, panels[offset:offset + 3]):
                    axis, start, end = descriptor
                    values = series[:, AXES.index(axis)]
                    panel.plot(np.arange(start, end + 1), values[start:end + 1],
                               color="#233b58", linewidth=1)
                    panel.set_xlim(start, end)
                    panel.set_ylim(*y_limits(values))
                    panel.set_xticks(sorted({start, end, *range((start // 10 + 1) * 10, end, 10)}))
                    panel.set_title(f"{axis} | days {start}-{end} | global vertical scale",
                                    loc="left", fontweight="bold")
                    panel.set_ylabel("Displacement (mm)")
                    panel.set_xlabel("Global day index")
                for panel in axes[len(panels[offset:offset + 3]):]:
                    panel.set_visible(False)
                fig.subplots_adjust(left=0.12, right=0.985, bottom=0.075, top=0.965,
                                    hspace=0.55)
                fig.savefig(path, format="png", dpi=150, facecolor="white")
            finally:
                plt.close(fig)
        paths.append(path)
    return paths, panels


def payload_for(config, condition, images, initial=None):
    prompt = config["initial_prompt"]
    if condition != "V0":
        if initial is None or initial.status != "success":
            raise ValueError("successful V0 required")
        prompt += "\n\n" + config["review_prompt"]
        prompt += "\nInitial predictions (untrusted data, not ground truth):\n"
        prompt += initial.predictions.model_dump_json()
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(image).decode()},
        "max_pixels": config["model"]["max_pixels"]} for image in images]
    if not 1 <= len(images) <= 5 or (condition != "V2" and len(images) != 1):
        raise ValueError("image budget exceeded")
    return {"model": config["model"]["request_id"], "temperature": 0,
            "enable_thinking": False, "max_tokens": 8192,
            "vl_high_resolution_images": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}]}


def request_once(client, base, key, case_id, condition, payload):
    method = f"{PROTOCOL}-{condition}"
    record = {"case_id": case_id, "condition": condition, "started_at": now(),
              "status": "failed", "response": None, "usage": None,
              "request_model": payload["model"], "max_tokens": payload["max_tokens"],
              "response_format": payload["response_format"], "error_type": None}
    started = time.perf_counter()
    result = failed_result(case_id, "range", method)
    try:
        response = client.post(base + "/chat/completions", json=payload,
                               headers={"Authorization": f"Bearer {key}"})
        record["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        record.update(response_model=body.get("model"), response_id=body.get("id"),
                      usage=body.get("usage"))
        choice = body["choices"][0]
        record.update(response=choice["message"]["content"],
                      finish_reason=choice.get("finish_reason"),
                      reasoning_present=bool(choice["message"].get("reasoning_content")))
        if (record["response_model"] != payload["model"] or record["reasoning_present"] or
                record["finish_reason"] not in (None, "stop")):
            raise ValueError("response model or completion contract mismatch")
        result = parse_result(record["response"], case_id, "range", method)
        record["status"] = "success"
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        record["error_type"] = type(exc).__name__
    record.update(completed_at=now(), latency_ms=round((time.perf_counter() - started) * 1000, 3))
    return result, record


def atomic_json(path, value):
    """Commit once; a partial temporary file is never interpreted as a response."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def journal_request(client, base, key, case_id, condition, payload, directory):
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "completed.json"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if completed.exists():
        saved = json.loads(completed.read_bytes())
        if saved["payload_sha256"] != digest:
            raise ValueError("resumed request changed")
        return RangeResult.model_validate(saved["prediction"]), saved["record"]
    started = directory / "attempt-started.json"
    if started.exists():
        saved = json.loads(started.read_bytes())
        if saved["payload_sha256"] != digest:
            raise ValueError("pending request changed")
        result = failed_result(case_id, "range", f"{PROTOCOL}-{condition}")
        record = {"case_id": case_id, "condition": condition, "status": "failed",
                  "error_type": "UncertainInterruptedAttempt", "usage": None,
                  "latency_ms": None, "completed_at": now()}
    else:
        # O_EXCL claim precedes sending. No endpoint or credentials are serialized.
        with started.open("x", encoding="utf-8") as stream:
            json.dump({"started_at": now(), "payload_sha256": digest,
                       "payload": payload}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        result, record = request_once(client, base, key, case_id, condition, payload)
    atomic_json(completed, {"payload_sha256": digest,
                           "prediction": result.model_dump(mode="json"), "record": record})
    return result, record


def verify_registration(config_path, inputs, registration):
    config = load_config(config_path)
    saved = json.loads(registration.read_bytes())
    if (saved["config_sha256"] != sha(config_path) or saved["source_sha256"] != source_hashes()
            or saved["input_manifest_sha256"] != sha(inputs / "manifest.json")):
        raise ValueError("registered source/config/input changed")
    cases = read_inputs(inputs)
    if len(cases) != config["cases"]:
        raise ValueError("expected 60 inputs")
    return config, cases


def execute(config, cases, images, directory, env_file):
    """Barrier between initial and paired review stages; plotting stays on main thread."""
    base, key = _credentials(env_file)
    output, records = {}, {}
    with httpx.Client(timeout=config["model"]["timeout_seconds"], follow_redirects=False) as client:
        for stage in (("V0",), ("V1", "V2")):
            jobs = []
            for index, case in enumerate(cases):
                initial = output.get((case.case_id, "V0"))
                details = []
                if stage != ("V0",) and initial.status == "success":
                    details, panels = render_details(case, initial, directory / "images" / case.case_id)
                    spec = directory / "images" / case.case_id / "panels.json"
                    if spec.exists():
                        if json.loads(spec.read_bytes()) != [list(x) for x in panels]:
                            raise ValueError("resumed crop spec differs")
                    else:
                        write_json(spec, panels)
                # Alternate review order by case ID position to reduce temporal confounding.
                for condition in (stage if index % 2 == 0 else tuple(reversed(stage))):
                    identity = (case.case_id, condition)
                    if condition != "V0" and initial.status != "success":
                        output[identity] = failed_result(case.case_id, "range", f"{PROTOCOL}-{condition}")
                        records[identity] = {"case_id": case.case_id, "condition": condition,
                            "status": "failed", "error_type": "UpstreamFailure", "usage": None,
                            "latency_ms": 0, "attempts": 0}
                        continue
                    paths = [images[case.case_id]] + (details if condition == "V2" else [])
                    payload = payload_for(config, condition, [p.read_bytes() for p in paths], initial)
                    jobs.append((identity, payload))
            def invoke(job):
                (cid, condition), payload = job
                return (cid, condition), journal_request(client, base, key, cid, condition, payload,
                    directory / "requests" / condition / cid)
            with ThreadPoolExecutor(max_workers=config["model"]["workers"]) as pool:
                for count, (identity, (row, record)) in enumerate(pool.map(invoke, jobs), 1):
                    output[identity], records[identity] = row, record
                    if count % 10 == 0 or count == len(jobs):
                        print(f"P8a {','.join(stage)} completed {count}/{len(jobs)}", flush=True)
    return output, records


def save_outputs(directory, cases, output, records):
    for condition in CONDITIONS:
        target = directory / condition / "predictions.jsonl"
        rows = [output[(case.case_id, condition)] for case in cases]
        if target.exists():
            if target.read_text(encoding="utf-8").splitlines() != [r.model_dump_json() for r in rows]:
                raise ValueError("existing outputs changed")
        else:
            write_rows(target, rows)
    report = {"protocol": PROTOCOL, "conditions": {condition: {
        "cases": len(cases), "success": sum(output[(c.case_id, condition)].status == "success" for c in cases),
        "predictions_sha256": sha(directory / condition / "predictions.jsonl"),
        "calls": sum((directory / "requests" / condition / c.case_id / "attempt-started.json").exists()
                     for c in cases),
        "usage_known": sum(isinstance(records[(c.case_id, condition)].get("usage"), dict) for c in cases),
        "total_tokens": sum((records[(c.case_id, condition)].get("usage") or {}).get("total_tokens", 0)
                            for c in cases),
        "summed_latency_ms": sum(records[(c.case_id, condition)].get("latency_ms") or 0 for c in cases),
        "cost": None} for condition in CONDITIONS},
        "artifacts_sha256": {str(p.relative_to(directory)).replace("\\", "/"): sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and
            p.name not in ("run.json", ".running") and p.suffix != ".tmp"}}
    write_json(directory / "run.json", report)
    return report


def run(config_path, inputs, registration, preflight, directory, env_file):
    config, entries = verify_registration(config_path, inputs, registration)
    proof = json.loads((preflight / "run.json").read_bytes())
    if any(r["success"] != 2 for r in proof["conditions"].values()):
        raise ValueError("engineering preflight failed")
    if json.loads((preflight / "identity.json").read_bytes()) != {
            "config_sha256": sha(config_path), "source_sha256": source_hashes()}:
        raise ValueError("preflight identity mismatch")
    for name, digest in proof["artifacts_sha256"].items():
        if sha(preflight / name) != digest:
            raise ValueError("preflight artifacts changed")
    if (directory / "run.json").exists():
        raise FileExistsError("completed P8a run is immutable")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"registration_sha256": sha(registration), "preflight_sha256": sha(preflight / "run.json")}
    identity_path = directory / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_bytes()) != identity:
            raise ValueError("run identity changed")
    else:
        write_json(identity_path, identity)
    lock = directory / ".running"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        cases = [case for _, case in entries]
        images = {c.case_id: inputs / "images" / f"{c.case_id}.png" for c in cases}
        output, records = execute(config, cases, images, directory, env_file)
        verify_registration(config_path, inputs, registration)
        return save_outputs(directory, cases, output, records)
    finally:
        lock.unlink()


def preflight(config_path, directory, env_file):
    config = load_config(config_path)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "identity.json", {"config_sha256": sha(config_path),
                                             "source_sha256": source_hashes()})
    # Normal and evolving Range: selection fixed before responses; score no accuracy here.
    cases = [_engineering_cases()[i] for i in (0, 3)]
    images = {}
    for case in cases:
        images[case.case_id] = directory / "images" / f"{case.case_id}.png"
        render_case(case, images[case.case_id])
    output, records = execute(config, cases, images, directory, env_file)
    return save_outputs(directory, cases, output, records)


def register(config_path, inputs, path):
    load_config(config_path)
    previous = json.loads((ROOT / "configs/p7b-registered.json").read_bytes())
    if sha(inputs / "manifest.json") != previous["input_manifest_sha256"]:
        raise ValueError("must reuse the fixed 60-case input package")
    if len(read_inputs(inputs)) != 60:
        raise ValueError("case count changed")
    write_json(path, {"protocol": PROTOCOL, "registered_at": now(),
        "config_sha256": sha(config_path), "source_sha256": source_hashes(),
        "input_manifest_sha256": sha(inputs / "manifest.json"),
        "selection_sha256": previous["selection_sha256"], "scope": "known-development-60",
        "versions": {name: version(name) for name in ("matplotlib", "numpy", "httpx")},
        "gate": "V2 vs V0: higher Affiliation F1 and IoU, nondecreasing Affiliation recall, "
                "nonincreasing negative-axis FAR, nondecreasing success. V2 vs V1: higher IoU. "
                "Otherwise stop and report all tradeoffs. No independent confirmation or Agent."})
    return {"registration": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "preflight", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/p8a-range-context.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/p7b/prepared/inputs")
    parser.add_argument("--registration", type=Path, default=ROOT / "configs/p8a-registered.json")
    parser.add_argument("--preflight", type=Path, default=ROOT / "runs/p8a/preflight")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/p8a/experiment")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.phase == "register":
        report = register(args.config, args.inputs, args.registration)
    elif args.phase == "preflight":
        report = preflight(args.config, args.preflight, args.env_file)
    else:
        report = run(args.config, args.inputs, args.registration, args.preflight, args.out, args.env_file)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts_sha256"}, indent=2))


if __name__ == "__main__":
    main()
