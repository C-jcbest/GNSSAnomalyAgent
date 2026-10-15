"""Frozen test continuation of CFV; the development implementation stays intact."""
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np
from experiment_exp08 import data_records
from experiment_exp26 import (
    DEPENDENCIES,
    MODELS,
    PROMPT,
    PairDecision,
    apply,
    check_frozen,
    curve_context,
    draw_panels,
    finite_rows,
    free_graph,
    infer,
    nearest,
    vectors,
)

from gnss_research.files import read_json, sha256, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM


def prepare(baseline: Path, output: Path, phase: str):
    config, dataset, manifest, records = data_records(baseline)
    if config["split"] != "test" or not (baseline / "execution-complete.json").exists():
        raise ValueError("completed test baselines required")
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / "pilot-completed.json").exists():
            raise ValueError("complete pilot first")
        check_frozen(read_json(output / "pilot-registration.json"))
    records = sorted(records, key=lambda record: record["station_id"])
    records = records[:24] if phase == "pilot" else records[24:]
    jobs = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        graph = free_graph(numeric)
        inputs = output / "inputs" / station
        inputs.mkdir(parents=True)
        local_vectors = vectors(observation, graph) if graph["pairs"] else []
        candidates = {"graph": graph, "numerical_prediction": numeric, "vectors": local_vectors}
        write_json(inputs / "candidates.json", candidates)
        write_json(output / "final/segment_numeric" / station / "result.json", numeric)
        write_json(output / "final/vector_nearest" / station / "result.json",
                   apply(observation, numeric, graph, nearest(graph, local_vectors)))
        raw_image = inputs / "original.png"
        images = []
        if graph["pairs"]:
            draw_panels(observation, graph, raw_image)
            vector_map = {item["id"]: item["delta"] for item in local_vectors}
            for pair in graph["pairs"]:
                image = inputs / (pair["id"] + ".png")
                draw_panels(observation, graph, image, pair, np.asarray(vector_map[pair["first_id"]]))
                images.append(image)
            evidence, left, right = curve_context(observation, graph)
            metadata = {"graph": graph, "missing_indices": np.flatnonzero(~evidence.observed).tolist(),
                        "correction_definition": "Subtract onset local median delta on inclusive [start,end]; unchanged elsewhere"}
            text_data = {"start_index": left, "end_index_exclusive": right,
                         "original_station_normalized": finite_rows(evidence.values[left:right] / evidence.noise),
                         "historical_residual_station_normalized": finite_rows(evidence.residual[left:right] / evidence.noise),
                         "onset_and_return_vectors": local_vectors}
        for family, model in MODELS.items():
            variants = ("image", "noimage") if family == "vl" else ("image", "noimage", "raw", "text")
            for variant in variants:
                mode = f"cf_{family}_{variant}"
                work = output / "predictions" / mode / station
                work.mkdir(parents=True)
                write_json(work / "candidates.json", candidates)
                if not graph["pairs"]:
                    write_json(work / "result.json", {"status": "success", "request_count": 0,
                               "pair_decision": {"pair_ids": [], "evidence": "No free candidate pairs"}})
                    write_json(output / "final" / mode / station / "result.json", numeric)
                    continue
                content = [{"type": "text", "text": json.dumps(metadata)}]
                if variant == "text":
                    content.append({"type": "text", "text": json.dumps(text_data, allow_nan=False)})
                if variant == "image":
                    selected_images = [raw_image, *images]
                elif variant == "raw":
                    selected_images = [raw_image]
                else:
                    selected_images = []
                for image in selected_images:
                    content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}})
                payload = {"model": model, "temperature": 0, "enable_thinking": False,
                           "response_format": {"type": "json_schema", "json_schema": {
                               "name": "gnss_counterfactual_pairs", "strict": True, "schema": PairDecision.model_json_schema()}},
                           "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
                write_json(work / "request.json", payload)
                jobs.append({"station": station, "mode": mode, "model": model, "kind": "pair",
                             "request_sha256": sha256(work / "request.json")})
        rf = read_json(baseline / "predictions/rf_split_temporal" / station / "result.json")
        for mode, view in (("sdr_flash", "slow_subtracted"), ("sdr_raw", "historical_residual")):
            work = output / "predictions" / mode / station
            payload = prepare_pair_focus(observation, rf, work, MODELS["flash"], SYSTEM, view)
            if payload is None:
                result = repair_return_pairs(observation, read_json(work / "result.json"), read_json(work / "candidates.json"))
                write_json(output / "final" / mode / station / "result.json", result)
            else:
                jobs.append({"station": station, "mode": mode, "model": MODELS["flash"], "kind": "sdr",
                             "request_sha256": sha256(work / "request.json")})
        print("prepared test", station, len(graph["pairs"]), "pairs", flush=True)
    dependencies = [*DEPENDENCIES, Path(__file__), Path("docs/experiment-exp26-test.md")]
    registration = {"experiment_id": "GNSS-VLM-EXP26-test", "phase": phase, "split": "test",
                    "baseline": str(baseline), "dataset": str(dataset), "seed": manifest["seed"],
                    "records": records, "jobs": jobs,
                    "frozen_files": {str(path): sha256(path) for path in dependencies}}
    write_json(output / (phase + "-registration.json"), registration)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for path in dependencies:
        if not (frozen / path.name).exists():
            shutil.copyfile(path, frozen / path.name)
    print("registered", len(jobs), "requests", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "remaining"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.baseline, args.output, args.phase)
    else:
        infer(args.output, args.phase)
