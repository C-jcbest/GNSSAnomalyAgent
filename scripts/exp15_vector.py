"""Local observed displacement vector plots and explicit numerical comparator."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, free_graph
from exp14_fingerprint import apply as apply
from exp14_fingerprint import call as call

from gnss_research.features import numerical_evidence
from gnss_research.files import sha256, write_json
from gnss_research.plots import render_observation

PROMPT = """Pair GNSS temporary-offset onset and return among supplied candidate pair IDs.
Output one JSON object {"pair_ids":[],"evidence":"one short sentence"}. Select pair IDs
(P1 etc.), never jump IDs (J1 etc.). The two local displacement-change vectors should be
approximately opposite across N/E/U, allowing noise and smooth motion. Each vector is
computed from neighboring ORIGINAL observed medians, not from a global extrapolated fit.
Positive means displacement goes UP at that edge; negative means DOWN. Compare all axes;
one axis can be weak. A separate permanent step may lie between onset and return, so do
not require absolute global levels to return or pair adjacent edges automatically.
The original full curve provides persistence context. Local statistics can be distorted
by spikes, short neighboring intervals and ongoing motion. If these prevent a clear pair,
select none. Never infer unavailable data or impose a universal mm/speed threshold.
Numerical positions are fixed and fallible. Already accepted events are handled elsewhere.
Only supplied pairs can be selected and selected pairs cannot share endpoints. If no
observational evidence is provided, select none rather than inventing directions.
"""


def vectors(observation: dict, graph: dict) -> list[dict]:
    evidence = numerical_evidence(observation)
    values = evidence.values / evidence.noise
    result = []
    indices = [j["index"] for j in graph["jumps"]]
    for position, jump in enumerate(graph["jumps"]):
        day = jump["index"]
        previous = indices[position - 1] if position else observation["background_end"]
        following = indices[position + 1] if position + 1 < len(indices) else len(values)
        before = values[max(previous, day - 7):day]
        after = values[day:min(following, day + 7)]
        displacement = np.nanmedian(after, axis=0) - np.nanmedian(before, axis=0)
        result.append({"id": jump["id"], "index": day, "delta": displacement.tolist(),
                       "before_observed": int(np.isfinite(before).all(axis=1).sum()),
                       "after_observed": int(np.isfinite(after).all(axis=1).sum())})
    return result


def nearest(graph: dict, local_vectors: list[dict]) -> dict:
    mapping = {v["id"]: np.asarray(v["delta"]) for v in local_vectors}
    ranked = []
    for pair in graph["pairs"]:
        a, b = mapping[pair["first_id"]], mapping[pair["second_id"]]
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.dot(a, b) >= 0:
            continue
        cost = np.linalg.norm(a + b) / (np.linalg.norm(a) + np.linalg.norm(b))
        ranked.append((float(cost), pair["id"], pair))
    used, selected = set(), []
    for _, pair_id, pair in sorted(ranked):
        endpoints = {pair["first_id"], pair["second_id"]}
        if not used & endpoints:
            used.update(endpoints)
            selected.append(pair_id)
    return {"status": "success", "pair_decision": {"pair_ids": selected,
        "evidence": "Disclosed untuned numerical opposite-vector nearest pairing"}}


def render_vectors(local_vectors: list[dict], path: Path):
    figure, panels = plt.subplots(len(local_vectors), 3, figsize=(10, 2.2 * len(local_vectors)), squeeze=False)
    for row, vector in enumerate(local_vectors):
        for axis in range(3):
            panel = panels[row, axis]
            value = vector["delta"][axis]
            panel.barh([0], [value], color="#b34a37" if value >= 0 else "#346fa4")
            panel.axvline(0, color="black", linewidth=.7)
            panel.set_yticks([])
            panel.set_title(f"{vector['id']} at {vector['index']}: {'NEU'[axis]} = {value:+.2f}")
            panel.set_xlabel("Local change / station MAD")
            panel.grid(axis="x", alpha=.2)
    figure.suptitle("Observed local jump vectors; + means UP, - means DOWN; compare opposite directions")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def prepare(observation: dict, numeric: dict, work: Path, model: str, mode: str):
    work.mkdir(parents=True, exist_ok=False)
    graph = free_graph(numeric)
    local_vectors = vectors(observation, graph)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numeric,
                                        "local_vectors": local_vectors})
    if not graph["pairs"]:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
                   "pair_decision": {"pair_ids": [], "evidence": "No unpaired step edges"}})
        return None
    content = [{"type": "text", "text": json.dumps(graph)}]
    images = []
    if mode == "vector_visual":
        images = [work / "original.png", work / "vectors.png"]
        render_observation(observation, images[0])
        render_vectors(local_vectors, images[1])
    elif mode == "vector_text":
        content.append({"type": "text", "text": json.dumps({"local_vectors": local_vectors})})
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
