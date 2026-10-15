"""Observed local distributions and training-only contrastive visual references."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, free_graph
from exp14_fingerprint import apply as apply
from exp14_fingerprint import call as call
from exp15_vector import nearest as nearest
from exp15_vector import vectors
from exp17_resolution import image_item, payload

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json

MODES = ("distribution", "reference_distribution", "distribution_text", "no_evidence")
PROMPT = """Review proposed temporary GNSS observation-offset pairs. Select supplied target
P IDs only; return the requested JSON decision and a short evidence statement.
The observations are shown as N/E/U before/after distributions around each edge. Orange
is the onset; blue is the return AFTER sign reversal. Each edge is centered on its own
before median. A return pair normally shows a similar abrupt before-to-after change in
orange and blue. Every dot is a real local observation; boxes summarize its distribution.
The dot spread is observational variability, not an additional event. Missing is unknown.
Look for a coherent joint change in the reliable components. Do NOT require precise
equality on all three axes: a weak component dominated by local scatter provides little
evidence either way. A mismatch between two clear, strong changes is counterevidence;
a tiny fluctuation cannot explain a clear jump. Permit noise and smooth underlying motion.
An independent permanent step can lie between onset and return, so global endpoint levels
need not coincide. All target panels of a given axis share a scale; the training reference
is a different station with its own scale. Training labels show how to read the comparison,
not the target's event count, dates or magnitudes. Consider ALL candidates jointly. Prefer
the most convincing return when candidates share an edge, and never select overlapping
endpoints. If none is supported, select none. Use no universal amplitude or speed threshold.
Do not revise spikes, sustained motion, existing transients or dates. This is verification
of local offset pairing only, not an inference of instrument causes or slope failure.
"""


def distributions(observation: dict, graph: dict) -> list[dict]:
    evidence = numerical_evidence(observation)
    values = evidence.values / evidence.noise
    indices = [jump["index"] for jump in graph["jumps"]]
    neighborhoods = {}
    for position, jump in enumerate(graph["jumps"]):
        origin = jump["index"]
        previous = indices[position - 1] if position else observation["background_end"]
        following = indices[position + 1] if position + 1 < len(indices) else len(values)
        left, right = max(previous, origin - 7), min(following, origin + 7)
        center = np.nanmedian(values[left:origin], axis=0)
        neighborhoods[jump["id"]] = {"absolute_index": origin,
            "before_indices": list(range(left, origin)), "after_indices": list(range(origin, right)),
            "before": values[left:origin] - center, "after": values[origin:right] - center}
    result = []
    for pair in graph["pairs"]:
        row = {"pair_id": pair["id"], "edges": []}
        for role, key, sign in (("onset", "first_id", 1), ("minus_return", "second_id", -1)):
            neighborhood = neighborhoods[pair[key]]
            edge = {"role": role, "jump_id": pair[key]}
            for name, value in neighborhood.items():
                if name in ("before", "after"):
                    edge[name] = [[float(x) if np.isfinite(x) else None for x in sample]
                                  for sample in value * sign]
                else:
                    edge[name] = value
            row["edges"].append(edge)
        result.append(row)
    return result


def training_reference(dataset: Path, records: list[dict]) -> dict:
    for record in sorted(records, key=lambda item: item["station_id"]):
        if record["split"] != "train":
            continue
        truth = read_json(dataset / record["truth"])
        steps = [event for event in truth["events"] if event["kind"] == "step"]
        transients = [event for event in truth["events"] if event["kind"] == "transient"]
        for transient in transients:
            for step in steps:
                if not transient["start"] < step["start"] < transient["end"]:
                    continue
                graph = {"jumps": [{"id": "RJ1", "index": transient["start"]},
                                   {"id": "RJ2", "index": step["start"]},
                                   {"id": "RJ3", "index": transient["end"] + 1}],
                         "pairs": [{"id": "R_positive", "first_id": "RJ1", "second_id": "RJ3"},
                                   {"id": "R_negative", "first_id": "RJ1", "second_id": "RJ2"}]}
                observation = read_json(dataset / record["observation"])
                samples = distributions(observation, graph)
                if any(not any(all(value is not None for value in sample) for sample in edge[side])
                       for pair in samples for edge in pair["edges"] for side in ("before", "after")):
                    continue
                return {"station_id": record["station_id"], "split": "train",
                    "graph": graph, "samples": samples,
                    "labels": {"R_positive": "temporary onset and opposite return: matched",
                               "R_negative": "onset and independent permanent step: unmatched"},
                    "observation_sha256": sha256(dataset / record["observation"]),
                    "truth_sha256": sha256(dataset / record["truth"])}
    raise ValueError("No eligible training-only contrastive reference; do not invent an example")


def render(samples: list[dict], path: Path, reference: bool):
    figure, panels = plt.subplots(3, len(samples), figsize=(6 * len(samples), 7),
                                 sharey="row", squeeze=False)
    colors = ("#c27435", "#347fa0")
    for column, pair in enumerate(samples):
        for axis in range(3):
            panel = panels[axis, column]
            for edge_index, (edge, color) in enumerate(zip(pair["edges"], colors)):
                for group, side in enumerate(("before", "after")):
                    values = np.asarray(edge[side], dtype=float)[:, axis]
                    values = values[np.isfinite(values)]
                    if not len(values):
                        continue
                    position = group + (-.16 if edge_index == 0 else .16)
                    boxes = panel.boxplot([values], positions=[position], widths=.25,
                        patch_artist=True, showfliers=False,
                        boxprops={"facecolor": color, "alpha": .22},
                        medianprops={"color": color, "linewidth": 2})
                    for median in boxes["medians"]:
                        median.set_zorder(4)
                    panel.scatter(position + np.linspace(-.06, .06, len(values)), values,
                                  color=color, s=16, alpha=.8, zorder=3,
                                  label=edge["role"] if group == 0 else None)
                    panel.annotate(f"{np.median(values):+.2f}", (position, np.median(values)),
                                   xytext=(0, 7), textcoords="offset points", ha="center",
                                   color=color, fontsize=8)
            panel.axhline(0, color="#555555", linewidth=.6)
            panel.set_xticks([0, 1], ["BEFORE", "AFTER"])
            panel.set_xlim(-.55, 1.55)
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
            panel.grid(axis="y", alpha=.25)
            if axis == 0:
                title = pair["pair_id"]
                if reference:
                    title += " : MATCHED RETURN" if pair["pair_id"] == "R_positive" else " : INDEPENDENT STEP"
                panel.set_title(title)
                panel.legend(fontsize=8, loc="best")
    figure.suptitle("TRAINING REFERENCE" if reference else "TARGET: observed local distributions; blue return already negated")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare(observation: dict, numeric: dict, reference: dict, work: Path, mode: str, model: str):
    work.mkdir(parents=True, exist_ok=False)
    graph = free_graph(numeric)
    local_vectors = vectors(observation, graph)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numeric,
                                         "local_vectors": local_vectors})
    if not graph["pairs"]:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
            "pair_decision": {"pair_ids": [], "evidence": "No unpaired step edges"}})
        return None
    content = [{"type": "text", "text": json.dumps({"target_candidates": graph})}]
    samples = distributions(observation, graph)
    images = []
    if mode in ("reference_distribution", "distribution_text"):
        content.append({"type": "text", "text": json.dumps({"training_reference": {
            "station_id": reference["station_id"], "split": "train", "labels": reference["labels"]}})})
        if mode == "reference_distribution":
            path = work / "reference.png"
            render(reference["samples"], path, True)
            content.append(image_item(path, True))
            images.append(path)
        else:
            content.append({"type": "text", "text": json.dumps({"reference_samples": reference["samples"]})})
    if mode in ("distribution", "reference_distribution"):
        path = work / "target.png"
        render(samples, path, False)
        content.append(image_item(path, True))
        images.append(path)
    elif mode == "distribution_text":
        content.append({"type": "text", "text": json.dumps({"same_plotted_target_samples": samples})})
    write_json(work / "local-samples.json", samples)
    request = payload(model, PROMPT, content, PairDecision)
    write_json(work / "request.json", request)
    from PIL import Image
    dimensions = {}
    for path in images:
        with Image.open(path) as picture:
            dimensions[path.name] = {"width": picture.width, "height": picture.height,
                                     "sha256": sha256(path)}
    write_json(work / "input-receipt.json", {"mode": mode, "images": dimensions,
        "request_sha256": sha256(work / "request.json"),
        "reference_station": reference["station_id"] if mode in ("reference_distribution", "distribution_text") else None})
    return request
