"""Compare observed and onset-only counterfactual GNSS trajectories."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, apply, call, free_graph
from exp15_vector import nearest, vectors
from experiment_exp08 import data_records
from experiment_exp25 import settings

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual

PROTOCOL = Path("docs/experiment-exp26.md")
MODELS = {"vl": "qwen3-vl-plus", "flash": "qwen3.8-flash"}
DEPENDENCIES = [Path(__file__), PROTOCOL, Path("scripts/exp14_fingerprint.py"),
                Path("scripts/exp15_vector.py"), Path("scripts/experiment_exp25.py")]
PROMPT = """Identify temporary GNSS observation offsets and their returns among candidate pair IDs.
Return {"pair_ids":[],"evidence":"one short observational sentence"}. Empty is valid.
Do not assume any supplied candidate is true. Compare ALL alternatives jointly across N/E/U.
Independent permanent steps and continuous physical displacement can coexist with a temporary
offset; an independent step can occur just before, inside, or after the temporary interval.
Do not automatically choose the earliest pair, neighboring edges, or the shortest interval.

Available evidence may include original displacement, normal-history residuals, and hypothetical
corrections. A counterfactual subtracts the onset's locally estimated offset ONLY inside that
candidate's interval. Its amplitude is estimated at the ONSET only, never fitted at the return.
If the pair is real, this operation should remove BOTH boundary discontinuities while preserving
smooth movement and unrelated jumps. If wrong, a visible discontinuity remains or is introduced
at its proposed return. Compare this behavior across components and against the original curves.
Correction is a diagnostic hypothesis, NOT an observation, fitted truth, or guarantee of smoothness.
Local offset estimates can be biased by noise, nearby jumps, spikes, and ongoing motion. Slight
remaining drift is not by itself evidence against a temporary offset. A smooth bend without an
abrupt shift-and-return is not a temporary observation offset. Do not flatten genuine movement.
Missing samples stay absent. Do not invent changes inside gaps. Normal-history extrapolation
can drift; verify the ORIGINAL rather than demanding return to an old absolute baseline.

Only select supplied pair IDs; selected pairs cannot share an endpoint. Other numerical events
are handled independently. Do not estimate dates, label activity, force an anomaly count, or apply
a universal amplitude/speed cutoff. If observational evidence does not resolve a pair, leave it
unselected. If no curve observations are supplied, do not invent visual evidence.
"""


def curve_context(observation, graph):
    evidence = numerical_evidence(observation)
    indices = [jump["index"] for jump in graph["jumps"]]
    left = max(evidence.background_end + 1, min(indices) - 30)
    right = min(len(evidence.values), max(indices) + 31)
    return evidence, left, right


def draw_panels(observation, graph, path, pair=None, offset=None):
    evidence, left, right = curve_context(observation, graph)
    raw = evidence.values / evidence.noise
    if pair is None:
        columns = (raw, evidence.residual / evidence.noise)
        titles = ("Original observations", "Normal-history residual; auxiliary")
    else:
        left = max(evidence.background_end + 1, pair["start"] - 30)
        right = min(len(raw), pair["end"] + 32)
        corrected = raw.copy()
        corrected[pair["start"]:pair["end"] + 1] -= offset
        columns = (raw, corrected)
        titles = ("Original observations", f"Hypothesis {pair['id']}: subtract onset offset")
    time = np.arange(left, right)
    figure, panels = plt.subplots(3, 2, figsize=(13, 7), dpi=150, sharex=True, sharey="row")
    for axis in range(3):
        center = np.nanmedian(raw[left:min(right, left + 10), axis])
        for column, values in enumerate(columns):
            panel = panels[axis, column]
            # Within a counterfactual comparison, preserve identical scales and centers.
            shift = center if pair is not None or column == 0 else np.nanmedian(values[left:min(right, left + 10), axis])
            panel.plot(time, values[left:right, axis] - shift, ".-", linewidth=.8, markersize=2.5)
            for jump in graph["jumps"]:
                if left <= jump["index"] < right:
                    target = pair is not None and jump["index"] in (pair["start"], pair["end"] + 1)
                    panel.axvline(jump["index"], color="#ad503e" if target else "#888888", linestyle=":", linewidth=1)
                    panel.text(jump["index"], .95, jump["id"], transform=panel.get_xaxis_transform(), fontsize=8)
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
            panel.grid(alpha=.2)
            if axis == 0:
                panel.set_title(titles[column])
            if axis == 2:
                panel.set_xlabel("Original daily index; missing days unfilled")
    figure.suptitle("Raw versus hypothesis; other jumps and physical motion remain" if pair else "Original local context and historical reference")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def finite_rows(values):
    return [[round(float(value), 5) if np.isfinite(value) else None for value in row] for row in values]


def prepare(baseline: Path, output: Path, phase: str):
    config, dataset, manifest, records = data_records(baseline)
    if config["split"] != "validation" or not (baseline / "execution-complete.json").exists():
        raise ValueError("completed development baselines required")
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / "pilot-completed.json").exists():
            raise ValueError("finish pilot before expansion")
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
        write_json(inputs / "candidates.json", {"graph": graph, "numerical_prediction": numeric,
                                                "vectors": local_vectors})
        write_json(output / "final/segment_numeric" / station / "result.json", numeric)
        numeric_response = nearest(graph, local_vectors)
        write_json(output / "final/vector_nearest" / station / "result.json",
                   apply(observation, numeric, graph, numeric_response))
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
            for variant in ("image", "noimage", "raw", "text"):
                mode = f"cf_{family}_{variant}"
                work = output / "predictions" / mode / station
                work.mkdir(parents=True)
                shutil.copyfile(inputs / "candidates.json", work / "candidates.json")
                if not graph["pairs"]:
                    response = {"status": "success", "request_count": 0, "pair_decision": {
                        "pair_ids": [], "evidence": "No free candidate pairs"}}
                    write_json(work / "result.json", response)
                    write_json(output / "final" / mode / station / "result.json", numeric)
                    continue
                content = [{"type": "text", "text": json.dumps(metadata)}]
                if variant == "text":
                    content.append({"type": "text", "text": json.dumps(text_data, allow_nan=False)})
                selected_images = [raw_image, *images] if variant == "image" else [raw_image] if variant == "raw" else []
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
        print("prepared", station, len(graph["pairs"]), "pairs", flush=True)
    registration = {"experiment_id": "GNSS-VLM-EXP26", "phase": phase, "baseline": str(baseline),
                    "dataset": str(dataset), "seed": manifest["seed"], "records": records, "jobs": jobs,
                    "frozen_files": {str(path): sha256(path) for path in DEPENDENCIES}}
    write_json(output / (phase + "-registration.json"), registration)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for path in DEPENDENCIES:
        if not (frozen / path.name).exists():
            shutil.copyfile(path, frozen / path.name)
    print("registered", len(jobs), "requests", flush=True)


def check_frozen(registration):
    if any(sha256(Path(path)) != value for path, value in registration["frozen_files"].items()):
        raise ValueError("experiment inputs changed")


def infer(output: Path, phase: str):
    registration = read_json(output / (phase + "-registration.json"))
    check_frozen(registration)
    marker = output / (phase + "-started.json")
    if marker.exists():
        raise ValueError("already started; no implicit retries")
    write_json(marker, {"retry_count": 0, "jobs": len(registration["jobs"])})
    records = {record["station_id"]: record for record in registration["records"]}

    def process(job):
        work = output / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(Path(registration["dataset"]) / records[job["station"]]["observation"])
        candidates = read_json(work / "candidates.json")
        if job["kind"] == "pair":
            response = call(observation, work, settings(job["model"]))
            result = apply(observation, candidates["numerical_prediction"], candidates["graph"], response)
        else:
            response = call_visual(observation, job["mode"], work, settings(job["model"]))
            result = repair_return_pairs(observation, response, candidates)
        result = copy.deepcopy(result)
        result["evaluation_dependency_workdirs"] = [str(work)]
        write_json(output / "final" / job["mode"] / job["station"] / "result.json", result)
        print(job["station"], job["mode"], response["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(process, registration["jobs"]))
    write_json(output / (phase + "-completed.json"), {"jobs": len(registration["jobs"])})


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
