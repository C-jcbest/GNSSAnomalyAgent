"""Visual activity presence; all event dates remain numerical displacement fits."""
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

ACTIVITY_PROMPT = """Determine whether a GNSS monitoring record contains sustained displacement
ACTIVITY during the analysis period after its declared normal historical prefix.
Answer one presence decision only. You are not detecting quality offsets or point outliers.
Look at the ORIGINAL N/E/U displacement together over time. The blue shaded prefix is this
station's normal reference, which can include ordinary drift and annual variation. Sustained
new movement changes displacement continuously for a substantial interval compared with
that ordinary behavior. Its direction and magnitude can differ across stations and axes.
An isolated spike, an instantaneous permanent offset, or an abrupt shift followed by a
return is an observation artifact, not sustained activity. Such artifacts can coexist with
true slow activity; do not let them cancel surrounding gradual movement. A stationary
plateau after movement arrests is cumulative displacement, not continuing activity.
The auxiliary normal-history residual is a fallible extrapolation; do not label activity
solely because that residual drifts when the ORIGINAL curve is stationary or merely seasonal.
Missing samples are gaps, never evidence of movement. Do not use a universal mm or speed
cutoff, assume an event count, or copy training example dates. If the observed raw curves
cannot distinguish sustained new movement from ordinary behavior, answer uncertain.
Output present, absent, or uncertain with a short observation-based explanation. Do not
estimate activity dates or label internal speed phases. Numerical fitting owns the dates.
"""


class ActivityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    activity: Literal["present", "absent", "uncertain"]
    evidence: str


def select_references(dataset: Path, records: list[dict]) -> list[dict]:
    selected = {}
    for record in sorted(records, key=lambda item: item["station_id"]):
        if record["split"] != "train":
            continue
        truth = read_json(dataset / record["truth"])
        scenario = truth["scenario"]
        if scenario not in selected:
            selected[scenario] = {"station_id": record["station_id"], "split": "train",
                "observation": record["observation"], "scenario": scenario,
                "activity_intervals": [{"start": event["start"], "end": event["end"]}
                                       for event in truth["events"] if event["kind"] == "deformation"]}
    return [selected[scenario] for scenario in ("normal", "spike", "step", "transient", "deformation", "mixed")]


def render_references(dataset: Path, references: list[dict], path: Path):
    figure, panels = plt.subplots(2, 3, figsize=(16, 9), dpi=120)
    for panel, reference in zip(panels.flat, references):
        observation = read_json(dataset / reference["observation"])
        evidence = numerical_evidence(observation)
        values = evidence.values / evidence.noise
        time = np.arange(len(values))
        for axis in range(3):
            panel.plot(time, values[:, axis] - np.nanmedian(values[:10, axis]),
                       linewidth=.7, label="NEU"[axis], alpha=.85)
        panel.axvspan(0, observation["background_end"], color="#a1c5dc", alpha=.3)
        for interval in reference["activity_intervals"]:
            panel.axvspan(interval["start"], interval["end"], color="#d0a338", alpha=.25)
        present = bool(reference["activity_intervals"])
        panel.set_title(f"Train {reference['scenario']}: activity {'PRESENT (gold)' if present else 'ABSENT'}")
        panel.set_ylabel("Original displacement / station MAD")
        panel.set_xlabel("Daily index; blue = normal prefix")
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
    figure.suptitle("TRAINING OBSERVATIONS: quality jumps/spikes are not sustained activity; gradual motion may coexist")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def render_activity_observation(observation: dict, path: Path):
    evidence = numerical_evidence(observation)
    time = np.arange(len(evidence.values))
    figure, panels = plt.subplots(3, 2, figsize=(15, 8), dpi=130, sharex=True)
    for axis in range(3):
        original = evidence.values[:, axis] / evidence.noise[axis]
        panels[axis, 0].plot(time, original - np.nanmedian(original[:10]), ".-", markersize=1.5, linewidth=.65)
        panels[axis, 1].plot(time, evidence.residual[:, axis] / evidence.noise[axis],
                            ".-", markersize=1.5, linewidth=.65, color="#965527")
        for column in range(2):
            panel = panels[axis, column]
            panel.axvspan(0, evidence.background_end, color="#a1c5dc", alpha=.3)
            panel.axvline(evidence.background_end + 1, color="#555555", linestyle=":")
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
            panel.grid(alpha=.2)
        panels[axis, 0].set_title("ORIGINAL displacement; normal prefix shaded blue")
        panels[axis, 1].set_title("Auxiliary residual of normal-history model; extrapolation can be wrong")
    for panel in panels[-1]:
        panel.set_xlabel("Original daily index; gaps remain missing")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare_activity(observation: dict, work: Path, model: str, mode: str,
                     references: list[dict], reference_image: Path, target_image: Path):
    work.mkdir(parents=True, exist_ok=False)
    content = []
    images = []
    if mode != "activity_zero":
        labels = [{"scenario": reference["scenario"], "activity_intervals": reference["activity_intervals"]}
                  for reference in references]
        content.append({"type": "text", "text": "TRAINING EXAMPLES, different stations: " + json.dumps(labels)})
        if mode != "activity_text":
            images.append(reference_image)
    content.append({"type": "text", "text": json.dumps({"target_length": len(observation["dates"]),
        "normal_historical_prefix": [0, observation["background_end"]],
        "analysis_period": [observation["background_end"] + 1, len(observation["dates"]) - 1]})})
    if mode != "activity_text":
        images.append(target_image)
    for path in images:
        content.append({"type": "text", "text": "TRAINING REFERENCE IMAGE" if path == reference_image else "TARGET STATION IMAGE"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}})
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "gnss_activity_presence", "strict": True, "schema": ActivityDecision.model_json_schema()}},
               "messages": [{"role": "system", "content": ACTIVITY_PROMPT}, {"role": "user", "content": content}]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {"mode": mode, "request_sha256": sha256(work / "request.json"),
        "images": {str(path.resolve()): sha256(path) for path in images}})


def call_activity(work: Path, settings: tuple):
    receipt, content = send_request(read_json(work / "request.json"), work, settings)
    if receipt["status"] == "success":
        try:
            receipt["activity_decision"] = ActivityDecision.model_validate_json(content).model_dump()
        except (ValueError, TypeError) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
    write_json(work / "result.json", receipt)
    return receipt


def apply_activity(observation: dict, source: dict, numeric: dict, response: dict) -> dict:
    events = copy.deepcopy(source["prediction"]["events"])
    if response["status"] == "success":
        decision = ActivityDecision.model_validate(response["activity_decision"])
        if decision.activity != "uncertain":
            events = [event for event in events if event["kind"] != "deformation"]
            if decision.activity == "present":
                fit = numeric["activity_fit"]
                events.append({"kind": "deformation", "start": fit["start"], "end": fit["end"],
                               "status": "confirmed", "phases": [], "evidence": decision.evidence})
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {**source, "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0, "activity_visual_status": response["status"],
            "activity_decision": response.get("activity_decision"),
            "derivation": "Visual raw-displacement activity presence with unchanged numerical candidate bounds"}
