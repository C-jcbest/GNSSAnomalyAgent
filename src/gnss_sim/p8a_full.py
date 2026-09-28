"""Explicitly requested full-Pilot development extension, retaining frozen 60-case calls."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from gnss_sim import p8a
from gnss_sim.input_only import read_inputs, sha, write_json
from gnss_sim.visual_semantics import frozen_images

ROOT = p8a.ROOT
DEFAULT = ROOT / "runs/p8a-full"


def check_original():
    frozen = json.loads((ROOT / "configs/p8a-frozen.json").read_bytes())
    if p8a.source_hashes() != frozen["source_sha256"]:
        raise ValueError("frozen P8a implementation changed")
    for name, digest in frozen["sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"frozen P8a artifact changed: {name}")
    for directory in (ROOT / "runs/p8a/preflight", ROOT / "runs/p8a/experiment"):
        record = json.loads((directory / "run.json").read_bytes())
        for name, digest in record["artifacts_sha256"].items():
            if sha(directory / name) != digest:
                raise ValueError("original P8a artifact changed")


def prepare(directory):
    check_original()
    pilot = ROOT / "data/pilots/pilot-v1"
    config = json.loads((ROOT / "configs/p7a-visual-semantics.json").read_bytes())
    images = frozen_images(pilot, config, ROOT / "runs/p6/qwen3.8-flash")
    manifest = json.loads((pilot / "manifest.json").read_bytes())
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for entry in manifest["cases"]:
        cid = entry["case_id"]
        source = pilot / "cases" / cid / "input.json"
        if sha(source) != entry["input_sha256"]:
            raise ValueError("pilot input changed")
        target = directory / "inputs/cases" / cid / "input.json"
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        (directory / "inputs/images").mkdir(exist_ok=True)
        shutil.copyfile(images[cid], directory / "inputs/images" / f"{cid}.png")
        entries.append({"case_id": cid, "input_sha256": sha(target),
                        "image_sha256": sha(images[cid])})
    write_json(directory / "inputs/manifest.json", {"protocol": "input-only-v1", "cases": entries})
    original = ROOT / "runs/p8a/experiment"
    experiment = directory / "experiment"
    # Preserve original attempts and completed journals byte-for-byte, including failures.
    shutil.copytree(original / "requests", experiment / "requests")
    shutil.copytree(original / "images", experiment / "images")
    cache_hashes = {str(p.relative_to(experiment)).replace("\\", "/"): sha(p)
                    for p in sorted(experiment.rglob("*")) if p.is_file()}
    source_rows = [json.loads(line) for line in (original / "V0/predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    cached_ids = sorted(r["case_id"] for r in source_rows)
    if len(cached_ids) != 60 or len(entries) != 300:
        raise ValueError("full extension must have 60 cached and 300 total cases")
    identity = {"protocol": "p8a-full-pilot-v1", "scope": "known-development-300",
        "authorized_extension": "User explicitly selected full pilot-v1 after failed P8a gate; "
                                "this does not revise the original 60-case gate or mean independent testing.",
        "registered_at": p8a.now(), "cases": 300, "cached_cases": cached_ids,
        "new_case_budget": 240, "new_call_budget": 720,
        "config_sha256": sha(ROOT / "configs/p8a-range-context.json"),
        "parent_frozen_sha256": sha(ROOT / "configs/p8a-frozen.json"),
        "parent_run_sha256": sha(original / "run.json"),
        "input_manifest_sha256": sha(directory / "inputs/manifest.json"),
        "pilot_manifest_sha256": sha(pilot / "manifest.json"),
        "inference_source_sha256": p8a.source_hashes(),
        "extension_source_sha256": sha(Path(__file__)),
        "cache_sha256": cache_hashes}
    write_json(directory / "registered.json", identity)
    return {"cases": 300, "cached_cases": 60, "new_cases": 240, "max_new_calls": 720}


def validate(directory):
    registered = json.loads((directory / "registered.json").read_bytes())
    if (p8a.source_hashes() != registered["inference_source_sha256"] or
            sha(Path(__file__)) != registered["extension_source_sha256"] or
            sha(ROOT / "configs/p8a-range-context.json") != registered["config_sha256"] or
            sha(ROOT / "configs/p8a-frozen.json") != registered["parent_frozen_sha256"] or
            sha(directory / "inputs/manifest.json") != registered["input_manifest_sha256"]):
        raise ValueError("extension registration changed")
    for name, digest in registered["cache_sha256"].items():
        if sha(directory / "experiment" / name) != digest:
            raise ValueError("cached input, request or response changed")
    rows = read_inputs(directory / "inputs")
    if len(rows) != 300 or not set(registered["cached_cases"]) <= {c.case_id for _, c in rows}:
        raise ValueError("unexpected extension case set")
    return registered, rows


def run(directory, env_file):
    registered, entries = validate(directory)
    config = p8a.load_config(ROOT / "configs/p8a-range-context.json")
    experiment = directory / "experiment"
    if (experiment / "run.json").exists():
        raise FileExistsError("completed extension is immutable")
    lock = experiment / ".running"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        cases = [c for _, c in entries]
        images = {c.case_id: directory / "inputs/images" / f"{c.case_id}.png" for c in cases}
        output, records = p8a.execute(config, cases, images, experiment, env_file)
        validate(directory)
        report = p8a.save_outputs(experiment, cases, output, records)
        cached = set(registered["cached_cases"])
        new = [record for (cid, _), record in records.items()
               if cid not in cached and record.get("error_type") != "UpstreamFailure"]
        added = {"new_requests": len(new),
                 "total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0) for r in new),
                 "cached_cases": len(cached), "registered_sha256": sha(directory / "registered.json"),
                 "note": "300-case report includes frozen 60; new usage counts only other 240 cases."}
        if len(new) > registered["new_call_budget"]:
            raise ValueError("extension request budget exceeded")
        write_json(directory / "extension.json", added)
        return {"conditions": report["conditions"], "extension": added}
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    result = prepare(args.out) if args.phase == "prepare" else run(args.out, args.env_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
