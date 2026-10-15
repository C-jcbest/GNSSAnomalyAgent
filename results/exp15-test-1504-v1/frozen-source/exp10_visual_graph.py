"""Joint visual ownership of numerical GNSS jumps; displacement activity stays fixed."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.provider import send_request
from gnss_research.schema import Prediction, event_masks, observed_values
from gnss_research.segmented_review import stabilized_displacement

GRAPH_PROMPT = """You interpret GNSS monitoring displacement with superposed observation offsets.
Numerical jump locations are fallible proposals, not known anomalies or ground truth.
Select a jointly consistent explanation of the observed level changes. Sustained physical
motion and isolated spikes are handled separately and must not be relabelled here.
The left panels show ORIGINAL displacement, scaled only by this station's historical noise.
The right panels subtract a fitted continuous slow trajectory; verify against the ORIGINAL.
Each vertical line has a jump ID. Candidate pair IDs join an onset to a later observed return.
Read ALL neighboring jumps in a single continuous context, rather than pairing the first
two jumps automatically. A permanent step can occur INSIDE a transient platform. Such an
interior step belongs to its own jump, not necessarily to the transient's return. Look at
the direction and continuing trajectory on all three axes before, during and after the pair.
Choose candidate pair IDs only for visible abrupt shift-and-return patterns. Different axes
can have opposite signed offsets. A smoothly bending original curve is physical movement,
not evidence for a step. A changed cumulative displacement plateau after motion stops is
not itself a quality offset. Retain standalone step IDs when persistent abrupt changes remain.
You may retain a permanent step at a transient edge only if a separate persistent component
is visible. Do not invent an anomaly count or a shared mm/speed threshold. Do not infer a jump
inside a gap. If the numerical level explanation cannot be resolved, choose keep_numeric.
If replace_levels is chosen, list the COMPLETE selected pair IDs and standalone step IDs;
unlisted level proposals will be rejected. No two selected pairs may share a jump endpoint.
Training examples, when supplied, illustrate shapes only: their dates, amplitudes and event
counts do not apply to the target station. Return the requested concise structured decision.
"""


class GraphDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["keep_numeric", "replace_levels"]
    pair_ids: list[str]
    step_ids: list[str]
    evidence: str


def build_graph(numerical: dict, max_separation: int = 120) -> dict:
    jumps = [{"id": f"J{position + 1}", "index": jump["index"]}
             for position, jump in enumerate(numerical["jumps"])]
    pairs = []
    for position, first in enumerate(jumps):
        for second in jumps[position + 1:]:
            if second["index"] - first["index"] > max_separation:
                break
            pairs.append({"id": f"P{len(pairs) + 1}", "first_id": first["id"],
                          "second_id": second["id"], "start": first["index"],
                          "end": second["index"] - 1})
    return {"jumps": jumps, "pairs": pairs}


def select_training_references(dataset: Path, records: list[dict]) -> list[dict]:
    training = sorted((record for record in records if record["split"] == "train"),
                      key=lambda record: record["station_id"])
    selected = []
    for scenario in ("step", "transient", "deformation", "mixed"):
        choices = []
        for record in training:
            truth = read_json(dataset / record["truth"])
            if truth["scenario"] == scenario:
                choices.append((record, truth))
        if not choices:
            raise ValueError("training reference scenario is unavailable")
        chosen = choices[0]
        if scenario == "mixed":
            for record, truth in choices:
                steps = [event for event in truth["events"] if event["kind"] == "step"]
                transients = [event for event in truth["events"] if event["kind"] == "transient"]
                if any(transient["start"] < step["start"] < transient["end"]
                       for step in steps for transient in transients):
                    chosen = (record, truth)
                    break
        record, truth = chosen
        selected.append({"station_id": record["station_id"], "split": "train",
                         "observation": record["observation"], "scenario": scenario,
                         "events": truth["events"]})
    return selected


def render_references(dataset: Path, references: list[dict], path: Path):
    figure, panels = plt.subplots(4, 3, figsize=(15, 10), dpi=120)
    for row, reference in enumerate(references):
        observation = read_json(dataset / reference["observation"])
        evidence = numerical_evidence(observation)
        events = reference["events"]
        left = max(observation["background_end"] + 1, min(event["start"] for event in events) - 45)
        right = min(len(evidence.values) - 1, max(event["end"] for event in events) + 45)
        time = np.arange(left, right + 1)
        for axis in range(3):
            panel = panels[row, axis]
            values = evidence.values[left:right + 1, axis] / evidence.noise[axis]
            panel.plot(time, values - np.nanmedian(values[:10]), ".-", markersize=2, linewidth=.8)
            for event in events:
                if event["kind"] == "step":
                    panel.axvline(event["start"], color="#bc4141", linewidth=1.2, label="step")
                elif event["kind"] == "transient":
                    panel.axvspan(event["start"], event["end"], color="#cc9a33", alpha=.2)
                    panel.axvline(event["start"], color="#cc9a33", linestyle=":")
                    panel.axvline(event["end"] + 1, color="#cc9a33", linestyle=":")
            panel.set_title(f"Train {row + 1}: {reference['scenario']}, {'NEU'[axis]}")
            panel.set_ylabel("Raw / station historical MAD")
            panel.grid(alpha=.2)
    figure.suptitle("TRAINING REFERENCES: red = permanent step; gold interval = transient; remaining smooth motion is activity")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def render_graph(observation: dict, numerical: dict, graph: dict, path: Path):
    evidence = numerical_evidence(observation)
    raw = evidence.values / evidence.noise
    auxiliary = stabilized_displacement(observation, numerical) / evidence.noise
    left = max(observation["background_end"] + 1, graph["jumps"][0]["index"] - 45)
    right = min(len(raw) - 1, graph["jumps"][-1]["index"] + 45)
    time = np.arange(left, right + 1)
    figure, panels = plt.subplots(3, 2, figsize=(15, 8), dpi=135, sharex=True)
    for axis in range(3):
        for column, values in enumerate((raw, auxiliary)):
            panel = panels[axis, column]
            local = values[left:right + 1, axis]
            panel.plot(time, local - np.nanmedian(local[:10]), ".-", markersize=2, linewidth=.8,
                       color="#245c70" if column == 0 else "#965527")
            for position, jump in enumerate(graph["jumps"]):
                panel.axvline(jump["index"], color="#666666", linestyle=":", linewidth=.7)
                panel.text(jump["index"], .97 - .12 * (position % 2), jump["id"],
                           transform=panel.get_xaxis_transform(), va="top", ha="center", fontsize=9,
                           bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
            panel.grid(alpha=.2)
        panels[axis, 0].set_title("ORIGINAL displacement; common context for ALL jumps")
        panels[axis, 1].set_title("Auxiliary slow-trajectory-subtracted displacement")
    for panel in panels[-1]:
        panel.set_xlabel("Original daily index; missing observations remain gaps")
        panel.set_xlim(left, right)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare_graph(observation: dict, numerical: dict, work: Path, model: str,
                  mode: str, references: list[dict], reference_image: Path) -> dict | None:
    work.mkdir(parents=True, exist_ok=False)
    graph = build_graph(numerical)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numerical})
    if not graph["pairs"]:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
                   "review_mode": "not_requested_no_pair",
                   "graph_decision": {"decision": "keep_numeric", "pair_ids": [], "step_ids": [],
                                      "evidence": "No candidate return pair"}})
        return None
    target = work / "joint-context.png"
    if mode != "text_graph":
        render_graph(observation, numerical, graph, target)
    instructions = {"jumps": graph["jumps"], "candidate_pairs": graph["pairs"],
                    "length": len(observation["dates"]),
                    "missing_indices": np.flatnonzero(~np.isfinite(observed_values(observation)).all(axis=1)).tolist()}
    content = []
    images = []
    if mode != "zero_graph":
        content.append({"type": "text", "text": "TRAINING EXAMPLES, different stations: " + json.dumps(references)})
        if mode != "text_graph":
            images.append(reference_image)
    if mode != "text_graph":
        images.append(target)
    content.append({"type": "text", "text": "TARGET STATION, classify these hypotheses: " + json.dumps(instructions)})
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                        + base64.b64encode(path.read_bytes()).decode("ascii")}})
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "gnss_jump_graph", "strict": True, "schema": GraphDecision.model_json_schema()}},
               "messages": [{"role": "system", "content": GRAPH_PROMPT},
                            {"role": "user", "content": content}]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {"mode": mode, "request_sha256": sha256(work / "request.json"),
               "images": {str(path.resolve()): sha256(path) for path in images}})
    return payload


def apply_graph(observation: dict, numerical: dict, graph: dict, response: dict) -> dict:
    events = copy.deepcopy(numerical["prediction"]["events"])
    if response["status"] == "success":
        decision = GraphDecision.model_validate(response["graph_decision"])
        if decision.decision == "replace_levels":
            pairs = {pair["id"]: pair for pair in graph["pairs"]}
            jumps = {jump["id"]: jump for jump in graph["jumps"]}
            if not set(decision.pair_ids) <= pairs.keys() or not set(decision.step_ids) <= jumps.keys():
                raise ValueError("visual decision refers to an unavailable hypothesis")
            if len(set(decision.pair_ids)) != len(decision.pair_ids) or len(set(decision.step_ids)) != len(decision.step_ids):
                raise ValueError("duplicate hypothesis IDs")
            used = set()
            selected = []
            for pair_id in decision.pair_ids:
                pair = pairs[pair_id]
                endpoints = {pair["first_id"], pair["second_id"]}
                if used & endpoints:
                    raise ValueError("selected pairs share a jump endpoint")
                used.update(endpoints)
                selected.append({"kind": "transient", "start": pair["start"], "end": pair["end"],
                                 "status": "confirmed", "evidence": decision.evidence, "phases": []})
            selected.extend({"kind": "step", "start": jumps[jump_id]["index"], "end": jumps[jump_id]["index"],
                             "status": "confirmed", "evidence": decision.evidence, "phases": []}
                            for jump_id in decision.step_ids)
            events = [event for event in events if event["kind"] not in ("step", "transient")]
            events.extend(selected)
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {**numerical, "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0, "graph_response_status": response["status"],
            "graph_decision": response.get("graph_decision"),
            "derivation": "Visual joint reassignment of fixed numerical level candidates; spikes and activity unchanged"}


def call_graph(observation: dict, numerical: dict, work: Path, settings: tuple) -> dict:
    receipt, content = send_request(read_json(work / "request.json"), work, settings)
    if receipt["status"] == "success":
        try:
            decision = GraphDecision.model_validate_json(content)
            receipt["graph_decision"] = decision.model_dump()
            graph = read_json(work / "candidates.json")["graph"]
            apply_graph(observation, numerical, graph, receipt)
        except (ValueError, KeyError, TypeError) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
    write_json(work / "result.json", receipt)
    return receipt
