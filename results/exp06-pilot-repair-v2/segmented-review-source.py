"""One visual review of competing displacement shapes, then numerical boundaries."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from gnss_research.features import numerical_evidence
from gnss_research.files import sha256, write_json
from gnss_research.guided import render_local
from gnss_research.plots import render_observation
from gnss_research.schema import Prediction, VisualDecision, event_masks, observed_values
from gnss_research.segmentation import segment_proposals

SEGMENT_REVIEW = """
Review ORIGINAL raw N/E/U curves with the provided local views. Numerical proposals come
from segmented displacement regression, not ground truth. Accept, reject, or select an
alternative morphology using the curves. Empty events is valid. All proposals are fallible.
Quality artifacts and real displacement activity may coexist: inspect them separately.
A transient is an abrupt level shift followed by the opposite return, possibly superposed
on an evolving displacement trajectory. Compare the level before the first jump and after
the second after allowing the background motion to continue. Do not confuse an elevated
temporary platform with sustained movement, or report its two edges as permanent steps.
Alternative return pairs may be false: two independent permanent jumps are also possible.
Persistent steps remain after their onset. A lone jump at the end without an observed return
cannot establish a transient. Missing intervals cannot establish exact unseen jumps.
The bounded-trajectory proposal estimates sustained activity and its arrest from ORIGINAL
displacement. Reject it if the raw curve is stationary except for jumps. A platform after
arrest is not active. An artifact inside activity does not cancel the surrounding movement.
When confirming a supplied shape, use its numerical start/end indices as your proposed
boundaries rather than estimating crowded plot tick positions. You may also propose your
own boundaries if the supplied fit disagrees with the raw curve. Do not assume any fixed
anomaly count, expected type, or physical threshold. Dashed historical background is only
an extrapolation of the declared normal prefix, not truth.
"""


def review_candidates(numerical: dict) -> list[dict]:
    events = [dict(event) for event in numerical["prediction"]["events"]
              if event["kind"] != "spike"]
    existing = {(event["kind"], event["start"], event["end"]) for event in events}
    for pair in numerical["pair_receipts"]:
        key = ("transient", pair["first"], pair["second"] - 1)
        if key in existing:
            continue
        events.append({"kind": key[0], "start": key[1], "end": key[2],
                       "status": "uncertain", "phases": [],
                       "evidence": "Alternative return pair; numerical information criterion did not select it"})
        existing.add(key)
    return [{"id": f"shape{index + 1}", **event} for index, event in enumerate(events)]


def prepare_segmented(observation: dict, numerical: dict, work: Path,
                      model: str, system: str) -> dict:
    work.mkdir(parents=True, exist_ok=False)
    evidence = numerical_evidence(observation)
    segmented = segment_proposals(observation, numerical, evidence)
    proposals = review_candidates(segmented)
    reference = evidence.values - evidence.residual
    images = [work / "input.png"]
    render_observation(observation, images[0], reference_curve=reference)
    normal = {"id": "normal-reference", "start": evidence.background_end - 120,
              "end": evidence.background_end}
    path = work / "normal-reference.png"
    render_local(observation, normal, path, reference)
    images.append(path)
    windows = []
    bounds_seen = set()
    for proposal in proposals:
        context = 45 if proposal["kind"] == "deformation" else 30
        left = max(evidence.background_end + 1, proposal["start"] - context)
        right = min(len(evidence.values) - 1, proposal["end"] + context)
        if (left, right) in bounds_seen:
            continue
        bounds_seen.add((left, right))
        window = {"id": proposal["id"], "start": left, "end": right}
        windows.append(window)
        path = work / (window["id"] + ".png")
        render_local(observation, window, path, reference)
        images.append(path)
    write_json(work / "candidates.json", {"numerical_prediction": segmented,
                                          "proposals": proposals, "windows": windows})
    instructions = {"length": len(evidence.values),
                    "normal_historical_reference": [0, evidence.background_end],
                    "missing_indices": np.flatnonzero(~evidence.observed).tolist(),
                    "shape_proposals": proposals, "local_views": windows,
                    "segmentation_parameters": segmented["parameters"],
                    "note": "Statistical model comparison is station normalized; no mm alarm threshold. "
                            "Alternative pairs compete with individual steps, not additional required events."}
    prompt = system + SEGMENT_REVIEW
    content = [{"type": "text", "text": json.dumps(instructions, allow_nan=False)}]
    content.extend({"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}
        for path in images)
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "gnss_events", "strict": True,
                   "schema": VisualDecision.model_json_schema()}},
               "messages": [{"role": "system", "content": prompt},
                            {"role": "user", "content": content}]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {
        "method": "segment_guided", "selector": "PELT + displacement shape comparison",
        "images": {path.name: sha256(path) for path in images},
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "request_sha256": sha256(work / "request.json")})
    return payload


def segmented_fusion(observation: dict, visual: dict, candidates: dict,
                     visual_bounds: bool = False) -> dict:
    if visual["status"] != "success":
        return {"status": "failed", "error_type": "VisualDependencyFailed", "request_count": 0}
    numerical = candidates["numerical_prediction"]
    proposals = candidates["proposals"]
    decisions = [event for event in visual["prediction"]["events"]
                 if event["status"] == "confirmed"]
    events = [event for event in numerical["prediction"]["events"] if event["kind"] == "spike"]
    used = set()
    accepted_pairs = []
    # A visually confirmed return pair owns its two edges; those edges are not also steps.
    for kind in ("transient", "deformation", "step"):
        for decision in decisions:
            if decision["kind"] != kind:
                continue
            possible = []
            for proposal in proposals:
                if proposal["kind"] != kind or proposal["id"] in used:
                    continue
                if kind == "step":
                    supported = abs(proposal["start"] - decision["start"]) <= 21
                    supported &= proposal["start"] not in accepted_pairs
                elif kind == "transient":
                    supported = abs(proposal["start"] - decision["start"]) <= 7
                    supported &= abs(proposal["end"] - decision["end"]) <= 7
                else:
                    supported = max(proposal["start"], decision["start"]) <= min(
                        proposal["end"], decision["end"])
                if supported:
                    possible.append(proposal)
            if not possible:
                continue
            proposal = min(possible, key=lambda item: abs(item["start"] - decision["start"])
                           + abs(item["end"] - decision["end"]))
            used.add(proposal["id"])
            selected = decision if visual_bounds and kind != "step" else proposal
            events.append({key: value for key, value in selected.items() if key != "id"}
                          | {"status": "confirmed", "evidence":
                             "Visual raw-displacement shape confirmation; "
                             + ("visual interval bounds" if visual_bounds and kind != "step"
                                else "numerical displacement bounds") + ": " + decision["evidence"]})
            if kind == "transient":
                accepted_pairs.extend([proposal["start"], proposal["end"] + 1])
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {"status": "success", "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0,
            "derivation": "same-response segmented shape confirmation; RF spikes; "
                          + ("visual interval bounds" if visual_bounds else "numerical bounds")}


def repair_return_pairs(observation: dict, visual: dict, candidates: dict) -> dict:
    """Visual semantics repair ambiguous type ownership, not numerical event presence.

    The independent numerical path always produces its decisions. A failed visual
    path supplies no repair and is explicitly recorded; it does not erase those
    decisions. This differs from the visual-veto fusion and is reported separately.
    """
    events = [dict(event) for event in candidates["numerical_prediction"]["prediction"]["events"]]
    repairs = []
    if visual["status"] == "success":
        decisions = [event for event in visual["prediction"]["events"]
                     if event["status"] == "confirmed" and event["kind"] == "transient"]
        for decision in decisions:
            step_dates = {event["start"] for event in events if event["kind"] == "step"}
            alternatives = [proposal for proposal in candidates["proposals"]
                            if proposal["kind"] == "transient"
                            and proposal["status"] == "uncertain"
                            and proposal["start"] in step_dates
                            and proposal["end"] + 1 in step_dates
                            and abs(proposal["start"] - decision["start"]) <= 7
                            and abs(proposal["end"] - decision["end"]) <= 7]
            if not alternatives:
                continue
            proposal = min(alternatives, key=lambda item: abs(item["start"] - decision["start"])
                           + abs(item["end"] - decision["end"]))
            edges = {proposal["start"], proposal["end"] + 1}
            events = [event for event in events
                      if event["kind"] != "step" or event["start"] not in edges]
            events.append({key: value for key, value in proposal.items() if key != "id"}
                          | {"status": "confirmed", "evidence":
                             "Visual shift-and-return semantics resolve an unpaired numerical jump pair: "
                             + decision["evidence"]})
            repairs.append({"proposal_id": proposal["id"], "edges": sorted(edges)})
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {"status": "success", "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0,
            "visual_dependency_status": visual["status"], "semantic_repairs": repairs,
            "derivation": "independent numerical detection with optional visual repair of ambiguous return pairs; "
                          "failed visual path leaves numerical decisions and is separately disclosed"}
