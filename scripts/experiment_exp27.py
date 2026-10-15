"""Separate edge magnification from trend-aware counterfactual displacement review."""
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, apply, free_graph
from exp15_vector import nearest, vectors
from experiment_exp08 import data_records
from experiment_exp26 import (
    DEPENDENCIES as CF_DEPENDENCIES,
)
from experiment_exp26 import (
    PROMPT as CF_PROMPT,
)
from experiment_exp26 import (
    check_frozen,
    curve_context,
    draw_panels,
    finite_rows,
    infer,
)
from scipy.stats import theilslopes

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json

PROTOCOL = Path("docs/experiment-exp27.md")
DEPENDENCIES = [*CF_DEPENDENCIES, Path(__file__), PROTOCOL,
                Path("scripts/report_exp27.py")]
MODES = {
    "cf_flash_image": ("qwen3.8-flash", "median", "cf"),
    "edge_flash_image": ("qwen3.8-flash", "median", "image"),
    "edge_flash_text": ("qwen3.8-flash", "median", "text"),
    "edge_flash_raw": ("qwen3.8-flash", "median", "raw"),
    "edge_flash_noimage": ("qwen3.8-flash", "median", "noimage"),
    "trend_flash_image": ("qwen3.8-flash", "trend", "image"),
    "trend_flash_text": ("qwen3.8-flash", "trend", "text"),
    "trend_flash_raw": ("qwen3.8-flash", "trend", "raw"),
    "trend_flash_noimage": ("qwen3.8-flash", "trend", "noimage"),
    "trend_vl_image": ("qwen3-vl-plus", "trend", "image"),
    "trend_vl_noimage": ("qwen3-vl-plus", "trend", "noimage"),
}
PROMPT = CF_PROMPT + """
Additional close-ups, when supplied, show BOTH proposed boundaries separately, with
original and hypothetically corrected curves side by side on identical local scales.
Use the original full context first to distinguish a smooth motion bend from a jump.
Then compare the immediate observations on BOTH sides of each boundary in all axes.
Finally compare alternative candidate corrections. A true return should undo the same
offset direction; correcting the wrong return leaves or creates a visible local break.
Keep any intervening independent step. Do not demand a globally flat corrected trace.
Some corrections use a local trend-separated onset estimate, others local median changes;
neither is ground truth. Do not accept or reject based on the estimator name. Missing
or unestimable corrections supply no evidence of continuity. Text samples, if supplied,
give the same original points and offsets and can be checked with the same logic.
"""


def trend_vectors(observation: dict, graph: dict) -> list[dict]:
    evidence = numerical_evidence(observation)
    values = evidence.values / evidence.noise
    indices = [jump["index"] for jump in graph["jumps"]]
    result = []
    for position, jump in enumerate(graph["jumps"]):
        day = jump["index"]
        previous = indices[position - 1] if position else evidence.background_end + 1
        following = indices[position + 1] if position + 1 < len(indices) else len(values)
        left = max(previous, day - 14)
        right = min(following, day + 14)
        times = np.arange(left, right)
        window = values[left:right]
        observed = np.isfinite(window).all(axis=1)
        before = observed & (times < day)
        after = observed & (times >= day)
        slopes, offsets = [], []
        estimable = before.sum() >= 2 and after.sum() >= 2
        if estimable:
            for axis in range(3):
                before_slope = theilslopes(window[before, axis], times[before]).slope
                after_slope = theilslopes(window[after, axis], times[after]).slope
                slope = float(np.median([before_slope, after_slope]))
                detrended = window[:, axis] - slope * (times - day)
                offset = np.median(detrended[after]) - np.median(detrended[before])
                slopes.append(slope)
                offsets.append(float(offset))
        result.append({"id": jump["id"], "index": day, "delta": offsets if estimable else [None] * 3,
                       "slope": slopes if estimable else [None] * 3, "estimable": bool(estimable),
                       "window": [left, right], "before_observed": int(before.sum()),
                       "after_observed": int(after.sum())})
    return result


def draw_edges(observation: dict, graph: dict, pair: dict, offset, path: Path):
    evidence = numerical_evidence(observation)
    raw = evidence.values / evidence.noise
    corrected = raw.copy()
    if offset is not None:
        corrected[pair["start"]:pair["end"] + 1] -= offset
    columns = 4 if offset is not None else 2
    figure, panels = plt.subplots(3, columns, figsize=(4.1 * columns, 7.5), dpi=150, squeeze=False)
    for edge_number, day in enumerate((pair["start"], pair["end"] + 1)):
        left, right = max(evidence.background_end + 1, day - 14), min(len(raw), day + 15)
        times = np.arange(left, right)
        views = (raw, corrected) if offset is not None else (raw,)
        for view_number, values in enumerate(views):
            column = edge_number * len(views) + view_number
            for axis in range(3):
                panel = panels[axis, column]
                center = np.nanmedian(raw[left:day, axis])
                panel.plot(times, raw[left:right, axis] - center, ".-", color="#bbbbbb",
                           linewidth=.6, markersize=2, label="Original reference")
                panel.plot(times, values[left:right, axis] - center, ".-", color="#246e8c",
                           linewidth=1, markersize=3)
                for jump in graph["jumps"]:
                    if left <= jump["index"] < right:
                        panel.axvline(jump["index"], color="#ad503e" if jump["index"] == day else "#999999",
                                      linestyle=":" if jump["index"] != day else "--", linewidth=.9)
                        panel.text(jump["index"], .94, jump["id"], fontsize=7,
                                   transform=panel.get_xaxis_transform())
                # Original and corrected views share limits; vertical motion remains visible.
                combined = np.concatenate((raw[left:right, axis], corrected[left:right, axis])) - center
                finite = combined[np.isfinite(combined)]
                if finite.size:
                    margin = max(float(np.ptp(finite)) * .08, .2)
                    panel.set_ylim(float(finite.min()) - margin, float(finite.max()) + margin)
                panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
                panel.grid(alpha=.2)
                if axis == 0:
                    boundary = "ONSET" if edge_number == 0 else "RETURN"
                    view = "original" if view_number == 0 else "corrected"
                    panel.set_title(f"{boundary} at {day}: {view}")
                if axis == 2:
                    panel.set_xlabel("Original daily index; gaps unfilled")
    figure.suptitle(f"Candidate {pair['id']}: compare BOTH edges; correction is a hypothesis")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def request_payload(model, prompt, metadata, images, text_data=None):
    content = [{"type": "text", "text": json.dumps(metadata, allow_nan=False)}]
    if text_data is not None:
        content.append({"type": "text", "text": json.dumps(text_data, allow_nan=False)})
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                        + base64.b64encode(image.read_bytes()).decode("ascii")}})
    return {"model": model, "temperature": 0, "enable_thinking": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "gnss_counterfactual_pairs", "strict": True,
                "schema": PairDecision.model_json_schema()}},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}]}


def prepare(baseline: Path, output: Path, phase: str):
    config, dataset, manifest, records = data_records(baseline)
    if not (baseline / "execution-complete.json").exists():
        raise ValueError("complete numerical baseline first")
    if config["split"] == "test":
        selection = read_json(Path("docs/exp27-test-selection.json"))
        if selection["script_sha256"] != sha256(Path(__file__)):
            raise ValueError("test method differs from selected development source")
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / "pilot-completed.json").exists():
            raise ValueError("complete and assess pilot before expansion")
        check_frozen(read_json(output / "pilot-registration.json"))
    records = sorted(records, key=lambda record: record["station_id"])
    records = records[:24] if phase == "pilot" else records[24:]
    jobs = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        graph = free_graph(numeric)
        original_vectors = vectors(observation, graph) if graph["pairs"] else []
        local_trend_vectors = trend_vectors(observation, graph) if graph["pairs"] else []
        inputs = output / "inputs" / station
        inputs.mkdir(parents=True)
        write_json(inputs / "estimates.json", {"median": original_vectors, "trend": local_trend_vectors})
        write_json(output / "final/segment_numeric" / station / "result.json", numeric)
        for mode, local_vectors in (("vector_nearest", original_vectors), ("trend_nearest", local_trend_vectors)):
            numeric_vectors = [{**item, "delta": np.asarray(item["delta"], dtype=float)}
                               for item in local_vectors]
            result = apply(observation, numeric, graph, nearest(graph, numeric_vectors))
            write_json(output / "final" / mode / station / "result.json", result)
        image_sets = {}
        text_sets = {}
        metadata = {"graph": graph, "correction_definition": "Subtract onset-only local offset on inclusive [start,end]; other points unchanged"}
        if graph["pairs"]:
            evidence, left, right = curve_context(observation, graph)
            metadata["missing_indices"] = np.flatnonzero(~evidence.observed).tolist()
            original_image = inputs / "original.png"
            draw_panels(observation, graph, original_image)
            for estimator, local_vectors in (("median", original_vectors), ("trend", local_trend_vectors)):
                image_sets[estimator] = {"image": [original_image], "raw": [original_image], "cf": [original_image]}
                text_sets[estimator] = {"start_index": left, "end_index_exclusive": right,
                    "original_station_normalized": finite_rows(evidence.values[left:right] / evidence.noise),
                    "historical_residual_station_normalized": finite_rows(evidence.residual[left:right] / evidence.noise),
                    "onset_and_return_vectors": local_vectors}
                vector_map = {item["id"]: item for item in local_vectors}
                for pair in graph["pairs"]:
                    offset = np.asarray(vector_map[pair["first_id"]]["delta"], dtype=float)
                    raw_edges = inputs / f"{estimator}-{pair['id']}-raw-edges.png"
                    draw_edges(observation, graph, pair, None, raw_edges)
                    image_sets[estimator]["raw"].append(raw_edges)
                    if not np.isfinite(offset).all():
                        image_sets[estimator]["image"].append(raw_edges)
                        continue
                    full = inputs / f"{estimator}-{pair['id']}-cf.png"
                    zoom = inputs / f"{estimator}-{pair['id']}-edges.png"
                    draw_panels(observation, graph, full, pair, offset)
                    draw_edges(observation, graph, pair, offset, zoom)
                    image_sets[estimator]["image"].extend([full, zoom])
                    image_sets[estimator]["cf"].append(full)
        for mode, (model, estimator, variant) in MODES.items():
            work = output / "predictions" / mode / station
            work.mkdir(parents=True)
            write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numeric})
            if not graph["pairs"]:
                write_json(work / "result.json", {"status": "success", "request_count": 0,
                           "pair_decision": {"pair_ids": [], "evidence": "No free candidate pairs"}})
                write_json(output / "final" / mode / station / "result.json", numeric)
                continue
            images = image_sets[estimator].get(variant, [])
            text = text_sets[estimator] if variant == "text" else None
            prompt = CF_PROMPT if variant == "cf" else PROMPT
            request_metadata = metadata
            if variant == "cf":
                request_metadata = {**metadata, "correction_definition":
                    "Subtract onset local median delta on inclusive [start,end]; unchanged elsewhere"}
            payload = request_payload(model, prompt, request_metadata, images, text)
            write_json(work / "request.json", payload)
            jobs.append({"station": station, "mode": mode, "model": model, "kind": "pair",
                         "request_sha256": sha256(work / "request.json")})
        print("prepared", station, len(graph["pairs"]), "pairs", flush=True)
    registration = {"experiment_id": "GNSS-VLM-EXP27", "phase": phase, "split": config["split"],
                    "baseline": str(baseline), "dataset": str(dataset), "seed": manifest["seed"],
                    "records": records, "jobs": jobs,
                    "frozen_files": {str(path): sha256(path) for path in DEPENDENCIES}}
    write_json(output / f"{phase}-registration.json", registration)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for path in DEPENDENCIES:
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
