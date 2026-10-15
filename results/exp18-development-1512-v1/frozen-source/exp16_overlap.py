"""Direct visual contrast of onset and sign-reversed candidate return vectors."""
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

from gnss_research.files import sha256, write_json

PROMPT = """Select GNSS temporary-offset onset/return pairs from the supplied pair IDs.
Reply exactly {"pair_ids":[],"evidence":"one short sentence"}. Choose P IDs, never J IDs.
For each candidate we compare the local ONSET displacement-change vector with the RETURN
vector AFTER MULTIPLYING IT BY MINUS ONE. This sign reversal is already done: do not invert
it again. A good returning offset gives SIMILAR paired bar lengths and directions in N/E/U,
as the onset and original return are opposite. The two displayed bars should OVERLAP in
shape, not oppose. Compare all three components jointly, allowing station noise and smooth
movement. If one displayed vector is tiny and the other large, this is not a convincing
matched return, even when the signs agree. All panels share the same vertical scale.
Local observed medians were computed separately around each edge; permanent interior steps
and ongoing deformation can coexist. Candidate pairs compete and cannot share endpoints.
Do not force a pair count. If no candidate is clearly matched, or observational evidence is
absent, return an empty pair_ids list. Apply no universal displacement or speed cutoff.
Do not detect activity or estimate dates; those numerical decisions remain independent.
"""


def contrasts(graph: dict, local_vectors: list[dict]) -> list[dict]:
    mapping = {v["id"]: v["delta"] for v in local_vectors}
    return [{"id": p["id"], "first_id": p["first_id"], "second_id": p["second_id"],
             "onset": mapping[p["first_id"]],
             "negated_return": [-x for x in mapping[p["second_id"]]]} for p in graph["pairs"]]


def render_contrasts(pairs: list[dict], path: Path):
    figure, panels = plt.subplots(len(pairs), 1, figsize=(8, max(3, 2.4 * len(pairs))),
                                  sharey=True, squeeze=False)
    positions = np.arange(3)
    for panel, pair in zip(panels[:, 0], pairs):
        panel.bar(positions - .18, pair["onset"], .36, color="#c27435", label="Onset")
        panel.bar(positions + .18, pair["negated_return"], .36, color="#347fa0", label="Minus return")
        panel.axhline(0, color="#333333", linewidth=.7)
        panel.set_xticks(positions, list("NEU"))
        panel.set_title(f"{pair['id']}: {pair['first_id']} vs minus {pair['second_id']}")
        panel.set_ylabel("Local change / station MAD")
        panel.grid(axis="y", alpha=.2)
    panels[0, 0].legend()
    figure.suptitle("Candidate return comparison: MATCHED shapes support a pair; no required count")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def prepare(observation: dict, numeric: dict, work: Path, model: str, mode: str):
    work.mkdir(parents=True, exist_ok=False)
    graph = free_graph(numeric)
    local_vectors = vectors(observation, graph)
    pairs = contrasts(graph, local_vectors)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": numeric,
                                        "local_vectors": local_vectors, "contrasts": pairs})
    if not pairs:
        write_json(work / "result.json", {"status": "success", "request_count": 0,
                   "pair_decision": {"pair_ids": [], "evidence": "No unpaired step edges"}})
        return None
    content = [{"type": "text", "text": json.dumps(graph)}]
    images = []
    if mode == "pair_overlap":
        images = [work / "pair-comparison.png"]
        render_contrasts(pairs, images[0])
    elif mode == "pair_text":
        content.append({"type": "text", "text": json.dumps({"same_plotted_vectors": pairs})})
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                        + base64.b64encode(path.read_bytes()).decode("ascii")}})
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "gnss_free_pair",
            "strict": True, "schema": PairDecision.model_json_schema()}},
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, *content]}]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {"mode": mode, "request_sha256": sha256(work / "request.json"),
               "images": {p.name: sha256(p) for p in images}})
    return payload
