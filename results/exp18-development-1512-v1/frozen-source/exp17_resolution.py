"""Controlled visual input and readout variants; numerical ownership stays fixed."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, free_graph
from exp14_fingerprint import apply as apply
from exp14_fingerprint import call as call
from exp15_vector import nearest as nearest
from exp15_vector import vectors
from exp16_overlap import PROMPT, contrasts
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from pydantic import BaseModel, ConfigDict

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.provider import send_request

MODES = ("bar_high", "bar_grid", "axis_prompt", "pair_tiles", "local_coarse",
         "local_daily", "local_text", "staged_readout", "no_image")
AXIS_PROMPT = PROMPT + """
Inspect every supplied candidate, not only the first panel. For each N/E/U component,
compare the orange ONSET bar with the blue MINUS RETURN bar using the zero line and ticks.
Check both sign AND relative size; a sign contradiction on a clearly visible component
is counterevidence even when another component matches. Assess the joint three-component
fingerprint. Do not prefer earlier IDs or the nearest dates. Do not force any pair count.
"""
LOCAL_PROMPT = """Identify temporary GNSS observation-offset onset/return pairs using ONLY
supplied candidate P IDs. Return {"pair_ids":[],"evidence":"one short observation"}.
Each candidate has three N/E/U panels. Orange is raw local onset displacement, centered by
its pre-edge 7-day median. Blue is the raw return neighborhood, centered in the same way
and THEN multiplied by -1. The sign reversal is already done; do not invert it again.
Horizontal coordinates are actual days relative to each edge, with edge at 0. Missing
samples remain missing; do not join or infer values in gaps. A temporary offset normally
has similar abrupt changes in the orange and blue curves near 0 across the three axes.
Smooth slopes and single-sample excursions are not the abrupt offset. Dotted vertical
lines mark neighboring candidate edges, not the edge under comparison. An independent
permanent step can occur inside a transient; global levels need not return to their old
values. Compare local change near 0, allowing continuous deformation underneath it.
Large disagreement in a clear axis, or a large jump paired to an ordinary fluctuation,
is counterevidence. Inspect every candidate and use cross-axis agreement and neighboring
observations; do not choose the first or nearest pair automatically. Select none when
evidence is absent or unclear. Pairs cannot share endpoints; no required pair count.
Use no shared mm/speed cutoff. Activity, spikes, existing events and numerical dates stay
independent. The task is local pair verification, not diagnosing physical causes.
"""
READOUT_PROMPT = """Read the supplied GNSS comparison bar charts into observations only.
For EVERY candidate P ID, output approximate orange onset and blue MINUS RETURN bar
heights in N,E,U order, using the numeric axis and zero line. The return is already negated.
Keep signs and component order. Do not select pairs or infer dates. Include one short
visible observation per candidate. Output JSON {"rows":[{"pair_id":"P...",
"onset":[0.0,0.0,0.0],"minus_return":[0.0,0.0,0.0],"observation":"..."}]}.
"""


class ReadoutRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    pair_id: str
    onset: list[float]
    minus_return: list[float]
    observation: str


class Readout(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    rows: list[ReadoutRow]


def payload(model: str, prompt: str, content: list, schema=PairDecision) -> dict:
    return {"model": model, "temperature": 0, "enable_thinking": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                         *content]}]}


def image_item(path: Path, high: bool) -> dict:
    item = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii")}}
    if high:
        item["max_pixels"] = 8388608
    return item


def render_bars(pairs: list[dict], path: Path, fine: bool, dpi: int = 260):
    figure, panels = plt.subplots(len(pairs), 1, figsize=(8, max(3, 2.4 * len(pairs))),
                                 sharey=True, squeeze=False)
    for panel, pair in zip(panels[:, 0], pairs):
        positions = np.arange(3)
        orange = panel.bar(positions - .18, pair["onset"], .36, color="#c27435", label="Onset")
        blue = panel.bar(positions + .18, pair["negated_return"], .36,
                         color="#347fa0", label="Minus return")
        panel.axhline(0, color="#333333", linewidth=.7)
        panel.set_xticks(positions, list("NEU"))
        panel.set_title(f"{pair['id']}: {pair['first_id']} vs minus {pair['second_id']}")
        panel.set_ylabel("Local change / station MAD")
        panel.grid(axis="y", alpha=.2)
        if fine:
            panel.yaxis.set_minor_locator(AutoMinorLocator(5))
            panel.grid(axis="y", which="minor", alpha=.12)
            panel.bar_label(orange, fmt="%+.2f", padding=3, fontsize=9)
            panel.bar_label(blue, fmt="%+.2f", padding=3, fontsize=9)
            panel.margins(y=.2)
    panels[0, 0].legend()
    figure.suptitle("Candidate return comparison: MATCHED shapes support a pair; no required count")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def local_samples(observation: dict, graph: dict) -> list[dict]:
    evidence = numerical_evidence(observation)
    values = evidence.values / evidence.noise
    jumps = {jump["id"]: jump["index"] for jump in graph["jumps"]}
    pairs = []
    for pair in graph["pairs"]:
        record = {"pair_id": pair["id"], "edges": []}
        for role, edge_id in (("onset", pair["first_id"]), ("minus_return", pair["second_id"])):
            origin = jumps[edge_id]
            left, right = max(0, origin - 14), min(len(values), origin + 15)
            before = values[max(0, origin - 7):origin]
            centered = values[left:right] - np.nanmedian(before, axis=0)
            if role == "minus_return":
                centered *= -1
            record["edges"].append({"role": role, "jump_id": edge_id,
                "absolute_index": origin, "relative_days": list(range(left - origin, right - origin)),
                "values_NEU": [[float(x) if np.isfinite(x) else None for x in row] for row in centered],
                "other_edges_relative": [day - origin for key, day in jumps.items()
                                         if key != edge_id and left <= day < right]})
        pairs.append(record)
    return pairs


def render_local(pair: dict, path: Path, daily: bool):
    figure, panels = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for axis, panel in enumerate(panels):
        for edge, color in zip(pair["edges"], ("#c27435", "#347fa0")):
            values = np.array(edge["values_NEU"], dtype=float)[:, axis]
            panel.plot(edge["relative_days"], values, "o-", color=color, linewidth=1.3,
                       markersize=3, label=f"{edge['role']} {edge['jump_id']}")
            for offset in edge["other_edges_relative"]:
                panel.axvline(offset, color=color, linestyle=":", alpha=.6)
        panel.axvline(0, color="#222222", linewidth=1)
        panel.axhline(0, color="#555555", linewidth=.6)
        panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
        panel.grid(alpha=.25)
        panel.legend(fontsize=9)
        panel.xaxis.set_major_locator(MultipleLocator(2 if daily else 7))
        if daily:
            panel.xaxis.set_minor_locator(MultipleLocator(1))
            panel.yaxis.set_minor_locator(AutoMinorLocator(2))
            panel.grid(which="minor", alpha=.13)
    panels[-1].set_xlabel("Observed days relative to each proposed edge (0)")
    panels[-1].set_xlim(-14.5, 14.5)
    figure.suptitle(f"{pair['pair_id']}: original local trajectory, return sign already reversed")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def prepare(observation: dict, numeric: dict, work: Path, model: str, mode: str):
    work.mkdir(parents=True, exist_ok=False)
    graph = free_graph(numeric)
    local_vectors = vectors(observation, graph)
    pairs = contrasts(graph, local_vectors)
    candidates = {"graph": graph, "numerical_prediction": numeric,
                  "local_vectors": local_vectors, "contrasts": pairs}
    write_json(work / "candidates.json", candidates)
    if not pairs:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
            "pair_decision": {"pair_ids": [], "evidence": "No unpaired step edges"}})
        return None
    content = [{"type": "text", "text": json.dumps(graph)}]
    images = []
    prompt = PROMPT
    if mode in ("bar_high", "bar_grid", "axis_prompt"):
        images = [work / "bars.png"]
        render_bars(pairs, images[0], mode != "bar_high")
        if mode == "axis_prompt":
            prompt = AXIS_PROMPT
    elif mode in ("pair_tiles", "staged_readout"):
        prompt = AXIS_PROMPT
        for pair in pairs:
            path = work / f"{pair['id']}.png"
            render_bars([pair], path, True)
            images.append(path)
        if mode == "staged_readout":
            prompt = READOUT_PROMPT
    else:
        prompt = LOCAL_PROMPT
        if mode != "no_image":
            samples = local_samples(observation, graph)
            write_json(work / "local-samples.json", samples)
            if mode == "local_text":
                content.append({"type": "text", "text": json.dumps({"same_plotted_samples": samples})})
            else:
                for pair in samples:
                    path = work / f"{pair['pair_id']}.png"
                    render_local(pair, path, mode == "local_daily")
                    images.append(path)
    content.extend(image_item(path, True) for path in images)
    schema = Readout if mode == "staged_readout" else PairDecision
    request = payload(model, prompt, content, schema)
    write_json(work / "request.json", request)
    from PIL import Image
    dimensions = {}
    for path in images:
        with Image.open(path) as picture:
            dimensions[path.name] = {"width": picture.width, "height": picture.height,
                                     "sha256": sha256(path)}
    write_json(work / "input-receipt.json", {"mode": mode,
        "request_sha256": sha256(work / "request.json"), "images": dimensions,
        "max_pixels_requested_per_image": 8388608})
    return request


def call_staged(observation: dict, work: Path, settings: tuple) -> dict:
    receipt, content = send_request(read_json(work / "request.json"), work, settings)
    if receipt["status"] == "success":
        try:
            readout = Readout.model_validate_json(content).model_dump()
            candidates = read_json(work / "candidates.json")
            expected = {pair["id"] for pair in candidates["graph"]["pairs"]}
            rows = readout["rows"]
            if {row["pair_id"] for row in rows} != expected or len(rows) != len(expected):
                raise ValueError("readout must cover all supplied candidates")
            if any(len(row[key]) != 3 for row in rows for key in ("onset", "minus_return")):
                raise ValueError("readout requires N/E/U values")
            receipt["visual_readout"] = readout
        except (ValueError, KeyError, TypeError) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
    write_json(work / "readout-result.json", receipt)
    if receipt["status"] != "success":
        write_json(work / "result.json", receipt)
        return receipt
    decision_work = work / "decision"
    decision_work.mkdir(exist_ok=False)
    write_json(decision_work / "candidates.json", candidates)
    content = [{"type": "text", "text": json.dumps(candidates["graph"])},
               {"type": "text", "text": json.dumps({"visual_readout": readout})}]
    write_json(decision_work / "request.json", payload(settings[2], AXIS_PROMPT, content))
    result = call(observation, decision_work, settings)
    result["request_count"] += receipt["request_count"]
    result["stage_one_status"] = receipt["status"]
    write_json(work / "result.json", result)
    return result
