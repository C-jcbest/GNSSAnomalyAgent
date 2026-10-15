"""Visual pairing of free numerical step edges using aligned raw neighborhoods."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp10_visual_graph import build_graph
from pydantic import BaseModel, ConfigDict, Field

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.plots import render_observation
from gnss_research.provider import send_request
from gnss_research.schema import Prediction, event_masks, observed_values

PROMPT = """Identify which proposed pairs of GNSS displacement jumps are a temporary offset
and its opposite return. Reply with exactly one JSON object {"pair_ids":[],"evidence":"..."}.
Only choose supplied pair IDs. A permanent independent step can lie INSIDE a temporary
platform, and continuous physical motion can continue underneath. Therefore absolute raw
levels before the onset and after the return need NOT be equal. Compare the LOCAL jump
fingerprints in N, E and U: an onset and return have opposing changes across the same axes;
an intervening independent step has its own different cross-axis pattern. Do not pair the
first two jumps automatically. Some axes may have very small or noisy changes.
The full image shows ORIGINAL observations. In the local image each column is a different
jump, each row the same component, and x is days relative to THAT jump. A solid vertical line
at zero marks its proposed edge; dotted lines are OTHER jumps, not the target edge. Gray is
raw observed displacement; blue is a 3-day median for readability, never an extra observation.
Each column has only been vertically centered on its own pre-jump neighborhood. The same
row uses a shared scale across columns. Compare the near-edge direction and magnitude,
allowing smooth local drift, nearby independent jumps and an isolated outlier to coexist.
Do not infer unseen jumps in gaps. Use no universal amplitude/speed cutoff or expected count.
Existing numerical events are not shown as truth. Select no pair if the return is unclear;
unselected events will remain numerical. No two selected pairs may share an endpoint.
Do not label activity or estimate dates. Keep evidence to one short observational sentence.
"""


class PairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    pair_ids: list[str]
    evidence: str = Field(min_length=1)


def free_graph(numeric: dict) -> dict:
    graph = build_graph(numeric)
    steps = {e["start"] for e in numeric["prediction"]["events"] if e["kind"] == "step"}
    graph["pairs"] = [p for p in graph["pairs"] if p["start"] in steps and p["end"] + 1 in steps]
    used = {p[key] for p in graph["pairs"] for key in ("first_id", "second_id")}
    graph["jumps"] = [j for j in graph["jumps"] if j["id"] in used]
    return graph


def render_fingerprints(observation: dict, graph: dict, path: Path):
    evidence = numerical_evidence(observation)
    normalized = evidence.values / evidence.noise
    figure, panels = plt.subplots(3, len(graph["jumps"]), figsize=(5 * len(graph["jumps"]), 8),
                                  dpi=130, sharey="row", squeeze=False)
    for column, jump in enumerate(graph["jumps"]):
        origin = jump["index"]
        left, right = max(0, origin - 14), min(len(normalized), origin + 15)
        relative = np.arange(left, right) - origin
        for axis in range(3):
            panel = panels[axis, column]
            values = normalized[left:right, axis].copy()
            values -= np.nanmedian(values[(relative >= -12) & (relative <= -4)])
            smooth = values.copy()
            for index in range(1, len(values) - 1):
                if np.isfinite(values[index]):
                    smooth[index] = np.nanmedian(values[index - 1:index + 2])
            panel.plot(relative, values, ".-", linewidth=.7, markersize=3, color="#aaaaaa", label="Original")
            panel.plot(relative, smooth, linewidth=1.6, color="#245c70", label="3-day median")
            panel.axvline(0, color="#b63737", linewidth=1.2)
            panel.axhline(0, color="#555555", linewidth=.6)
            for other in graph["jumps"]:
                offset = other["index"] - origin
                if other != jump and -14 <= offset <= 14:
                    panel.axvline(offset, color="#777777", linestyle=":")
                    panel.text(offset, .9, other["id"], transform=panel.get_xaxis_transform())
            panel.set_title(f"{jump['id']}: index {origin}, {'NEU'[axis]}")
            panel.set_ylabel("Original / station MAD; locally centered")
            panel.grid(alpha=.2)
            if axis == 2:
                panel.set_xlabel("Days relative to this jump")
    panels[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare(observation: dict, numeric: dict, work: Path, model: str, mode: str):
    work.mkdir(parents=True, exist_ok=False)
    graph = free_graph(numeric)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numeric})
    if not graph["pairs"]:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
                   "pair_decision": {"pair_ids": [], "evidence": "No unpaired step edges"}})
        return None
    images = []
    if mode != "text":
        images.append(work / "original.png")
        render_observation(observation, images[-1])
        if mode == "fingerprint":
            images.append(work / "fingerprints.png")
            render_fingerprints(observation, graph, images[-1])
    content = [{"type": "text", "text": json.dumps(graph)}]
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                        + base64.b64encode(path.read_bytes()).decode("ascii")}})
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "gnss_free_pair",
            "strict": True, "schema": PairDecision.model_json_schema()}},
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {"mode": mode, "request_sha256": sha256(work / "request.json"),
               "images": {p.name: sha256(p) for p in images}})
    return payload


def apply(observation: dict, numeric: dict, graph: dict, response: dict):
    events = copy.deepcopy(numeric["prediction"]["events"])
    if response["status"] == "success":
        decision = PairDecision.model_validate(response["pair_decision"])
        pairs = {p["id"]: p for p in graph["pairs"]}
        if len(set(decision.pair_ids)) != len(decision.pair_ids) or not set(decision.pair_ids) <= pairs.keys():
            raise ValueError("invalid or duplicated pair IDs")
        used = set()
        for pair_id in decision.pair_ids:
            pair = pairs[pair_id]
            edges = {pair["start"], pair["end"] + 1}
            if used & edges:
                raise ValueError("pairs share an endpoint")
            used.update(edges)
            events = [e for e in events if e["kind"] != "step" or e["start"] not in edges]
            events.append({"kind": "transient", "start": pair["start"], "end": pair["end"],
                           "status": "confirmed", "phases": [], "evidence": decision.evidence})
    prediction = Prediction.model_validate({"events": events})
    masks = event_masks(prediction, np.isfinite(observed_values(observation)).all(axis=1))
    return {**numeric, "prediction": prediction.model_dump(), "generic_score": None,
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "request_count": 0, "pair_visual_status": response["status"],
            "pair_decision": response.get("pair_decision")}


def call(observation: dict, work: Path, settings: tuple):
    receipt, content = send_request(read_json(work / "request.json"), work, settings)
    if receipt["status"] == "success":
        try:
            receipt["pair_decision"] = PairDecision.model_validate_json(content).model_dump()
            candidates = read_json(work / "candidates.json")
            apply(observation, candidates["numerical_prediction"], candidates["graph"], receipt)
        except (ValueError, KeyError, TypeError) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
    write_json(work / "result.json", receipt)
    return receipt
