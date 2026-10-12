"""Numerical localization and multiscale raw-NEU review in one visual-model call."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_research.features import numerical_evidence
from gnss_research.files import sha256, write_json
from gnss_research.plots import render_observation
from gnss_research.schema import EVENT_TYPES, VisualDecision, observed_values

GUIDED_METHODS = ("guided_fusion", "guided_no_local", "guided_no_scores")
REVIEW = """
Review the global raw N/E/U plot and, when provided, high-resolution local views.
All local indices are ORIGINAL daily indices. Local windows were selected by a supervised
balanced random forest trained on other simulated stations. Selection is not an anomaly
decision; every station, including normal controls, receives review windows.
Treat classifier suggestions as fallible. You may reject, change, or add events.
Inspect quality artifacts (spike, step, transient) separately from sustained motion: a
quality artifact superposed on deformation still needs its own event. Look at neighboring
raw samples for exact point indices; a transient has a shift AND a return, whereas a step
is only the onset, not the elevated platform. A transient must not automatically yield two
step labels. Weak isolated noise is not necessarily a spike.
For deformation, compare original displacement before, within AND AFTER the activity.
Stop the activity when sustained movement stops; elevated stationary residuals after arrest
are not continuing motion. Compare against historical trend and seasonality. Numerical
centered-window responses can lead or lag actual displacement and are not final boundaries.
Do not simply copy proposals or assume the six inspection windows contain six anomalies.
The dashed curve is a linear + annual seasonal background fitted ONLY on the normal prefix.
It is contextual extrapolation, not ground truth. Persistent offset relative to it is not
itself active motion. Compare ordinary noise with the close-up of the historical reference.
"""


def inspection_windows(observation: dict, numerical: dict) -> list[dict]:
    values = observed_values(observation)
    observed = np.isfinite(values).all(axis=1)
    first = observation["background_end"] + 1
    windows = []
    for kind in EVENT_TYPES:
        probability = np.asarray(numerical["type_scores"][kind])
        eligible = observed & (np.arange(len(values)) >= first)
        count = 2 if kind in ("spike", "step") else 1
        for _ in range(count):
            scores = np.where(eligible, probability, -np.inf)
            center = int(scores.argmax())
            if not np.isfinite(scores[center]):
                break
            half = 21 if kind in ("spike", "step") else 60
            left = max(first, center - half)
            right = min(len(values) - 1, center + half)
            if kind in ("transient", "deformation"):
                proposals = [event for event in numerical["prediction"]["events"]
                             if event["kind"] == kind]
                if proposals:
                    # Rank by total classifier evidence, not truth or an expected event count.
                    event = max(proposals, key=lambda item: float(probability[
                        item["start"]:item["end"] + 1][
                            observed[item["start"]:item["end"] + 1]].sum()))
                    left = max(first, event["start"] - 45)
                    right = min(len(values) - 1, event["end"] + 45)
            windows.append({"id": f"view{len(windows) + 1}", "start": left, "end": right,
                            "inspection_kind": kind, "peak_index": center,
                            "peak_classifier_score": float(probability[center])})
            eligible[max(first, center - 21):min(len(values), center + 22)] = False
    return windows


def render_local(observation: dict, window: dict, path: Path,
                 reference: np.ndarray) -> None:
    values = observed_values(observation)
    left, right = window["start"], window["end"]
    time = np.arange(left, right + 1)
    figure, panels = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True, dpi=120)
    for column, panel in enumerate(panels):
        panel.plot(time, values[left:right + 1, column], ".-", markersize=2.5,
                   linewidth=0.8, color="#23495b")
        panel.plot(time, reference[left:right + 1, column], "--", color="#be7739", linewidth=1)
        panel.set_ylabel(f"{'NEU'[column]} (mm)")
        panel.grid(alpha=0.3)
    panels[0].set_title(f"{window['id']} — raw displacement, indices {left} to {right}")
    panels[-1].set_xlim(left, right)
    panels[-1].set_xlabel("Original daily index; missing values remain gaps")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare_guided(observation: dict, numerical: dict, method: str, work: Path,
                   model: str, system: str) -> dict:
    work.mkdir(parents=True, exist_ok=False)
    evidence = numerical_evidence(observation)
    reference = evidence.values - evidence.residual
    windows = inspection_windows(observation, numerical)
    write_json(work / "candidates.json", {"selector": "rf_split_temporal", "windows": windows,
                                         "numerical_prediction": numerical})
    global_plot = work / "input.png"
    render_observation(observation, global_plot, reference_curve=reference)
    images = [global_plot]
    instructions = {
        "length": len(observation["dates"]),
        "normal_historical_reference": [0, observation["background_end"]],
        "missing_indices": [day for day, row in enumerate(observation["displacement_mm"])
                            if row is None],
        "views": [{key: window[key] for key in ("id", "start", "end")} for window in windows],
        "task": "Independently review quality artifacts and displacement activity; both may coexist",
    }
    if method != "guided_no_scores":
        instructions["classifier_proposals"] = numerical["prediction"]["events"]
        instructions["inspection_evidence"] = windows
        instructions["point_neighbors"] = [
            {"index": day, "raw_neu_mm": observation["displacement_mm"][day]}
            for day in sorted({day for window in windows[:4]
                               for day in range(max(0, window["peak_index"] - 2),
                                                min(len(observation["dates"]),
                                                    window["peak_index"] + 3))})]
        instructions["score_limitations"] = (
            "Uncalibrated RF posterior, not physical alarm probability. Duration transitions "
            "are learned on training stations; missing-day emissions are neutral. "
            "No universal physical threshold and no test-label threshold selection.")
    if method != "guided_no_local":
        background_window = {"id": "normal-reference", "start": observation["background_end"] - 120,
                             "end": observation["background_end"]}
        background_plot = work / "normal-reference.png"
        render_local(observation, background_window, background_plot, reference)
        images.append(background_plot)
        for window in windows:
            path = work / f"{window['id']}.png"
            render_local(observation, window, path, reference)
            images.append(path)
    prompt = system + REVIEW
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
        "method": method, "selector": "rf_split_temporal",
        "images": {path.name: sha256(path) for path in images},
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "request_sha256": sha256(work / "request.json")})
    return payload
