"""Matched controls for frozen CFV requests; never repeat the image request."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from experiment_exp26 import check_frozen, curve_context, finite_rows, infer

from gnss_research.files import read_json, sha256, write_json

PROTOCOL = Path("docs/experiment-exp28-s1.md")
MODES = ("cf_flash_text", "cf_flash_raw", "cf_flash_noimage")


def prepare(visual, output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for phase in ("pilot", "remaining"):
        registration = read_json(visual / f"{phase}-registration.json")
        check_frozen(registration)
        records.extend(registration["records"])
    dataset = Path(registration["dataset"])
    jobs = []
    for record in records:
        station = record["station_id"]
        source = visual / "predictions/cf_flash_image" / station
        candidates = read_json(source / "candidates.json")
        graph, numeric = candidates["graph"], candidates["numerical_prediction"]
        if graph["pairs"]:
            observation = read_json(dataset / record["observation"])
            evidence, left, right = curve_context(observation, graph)
            vectors = read_json(visual / "inputs" / station / "candidates.json")["vectors"]
            text_data = {"start_index": left, "end_index_exclusive": right,
                "original_station_normalized": finite_rows(evidence.values[left:right] / evidence.noise),
                "historical_residual_station_normalized": finite_rows(evidence.residual[left:right] / evidence.noise),
                "onset_and_return_vectors": vectors}
            original = read_json(source / "request.json")
        for mode in MODES:
            work = output / "predictions" / mode / station
            work.mkdir(parents=True)
            write_json(work / "candidates.json", candidates)
            if not graph["pairs"]:
                write_json(work / "result.json", {"status": "success", "request_count": 0,
                           "pair_decision": {"pair_ids": [], "evidence": "No free candidate pairs"}})
                write_json(output / "final" / mode / station / "result.json", numeric)
                continue
            payload = copy.deepcopy(original)
            for message in payload["messages"]:
                if not isinstance(message["content"], list):
                    continue
                images = [item for item in message["content"] if item["type"] == "image_url"]
                message["content"] = [item for item in message["content"] if item["type"] != "image_url"]
                if mode == "cf_flash_raw":
                    message["content"].extend(images[:1])
                elif mode == "cf_flash_text":
                    message["content"].append({"type": "text", "text": json.dumps(text_data, allow_nan=False)})
            write_json(work / "request.json", payload)
            jobs.append({"station": station, "mode": mode, "model": payload["model"], "kind": "pair",
                         "request_sha256": sha256(work / "request.json")})
    frozen = {**registration["frozen_files"], str(Path(__file__)): sha256(Path(__file__)),
              str(PROTOCOL): sha256(PROTOCOL)}
    write_json(output / "full-registration.json", {**registration, "experiment_id": "GNSS-VLM-EXP28-S1",
        "phase": "full", "visual": str(visual), "records": records, "jobs": jobs, "frozen_files": frozen})
    source_snapshot = output / "frozen-source"
    source_snapshot.mkdir()
    for name in frozen:
        shutil.copyfile(name, source_snapshot / Path(name).name)
    print("registered supplement", len(records), "stations", len(jobs), "requests", flush=True)


def publish(output):
    if not (output / "full-completed.json").exists():
        raise ValueError("complete all supplemental requests first")
    registration = read_json(output / "full-registration.json")
    check_frozen(registration)
    visual = Path(registration["visual"])
    for mode in MODES:
        shutil.copytree(output / "final" / mode, visual / "final" / mode)
    write_json(output / "published.json", {"visual": str(visual), "methods": list(MODES)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "publish"))
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.visual, args.output)
    elif args.stage == "infer":
        infer(args.output, "full")
    else:
        publish(args.output)
