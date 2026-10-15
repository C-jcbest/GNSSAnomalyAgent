"""Direct range localization in raw GNSS plots, followed by visual boundary zooms."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from exp17_resolution import image_item, payload
from experiment_exp07 import rebuild
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, ConfigDict

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.provider import send_request
from gnss_research.schema import Interval

PROMPT = """Identify anomaly INTERVALS directly from the supplied GNSS N/E/U observations.
Output only transient and deformation ranges, with inclusive ORIGINAL daily indices.
There is no known event count. Use the historical normal prefix as this station's
reference. Different stations have different noise, drift and seasonal displacement.
Do not impose a universal displacement or speed alarm threshold.

transient: an abrupt observation-level shift persists for an interval and then returns
by an opposite change. Use first shifted day through last day BEFORE the return.
Ordinary smooth physical motion may underlie a temporary offset, and a permanent
step may occur during the interval. Judge the three components jointly.
deformation: a sustained new continuous displacement trajectory relative to ordinary
history. First establish an active period from the ORIGINAL displacement. Start at
the observed departure into sustained movement; end at the last active day before
arrest. A stationary plateau after movement stops is NOT continuing activity.
Do not label an instantaneous permanent step, isolated spike, ordinary drift or annual
variation as an interval. These can coexist with an interval and do not erase it.

The left panels are ORIGINAL displacement (blue) with an extrapolated historical
normal fit (dashed orange); the right panels are original minus that historical fit.
The fit uses only the normal prefix and can be wrong outside it. Residual drift alone
cannot establish activity when ORIGINAL displacement remains ordinary or stationary.
Raw curves take precedence. Missing samples stay gaps; never infer an event solely
inside a gap. Read the numeric daily axis rather than guessing proportional dates.
Do not assign internal acceleration phases. If a boundary is unresolved use uncertain;
do not extend a confirmed interval to unexplained missing days or an arrested plateau.
Return an empty events list if no interval is supported or observation evidence is absent.
Give a short explanation of visible behavior for each interval. No numerical detection
proposals or target truth are supplied. Output the requested JSON only.
"""
ZOOM_PROMPT = """
This is a second observation pass. The supplied coarse ranges are YOUR previous
estimates, not ground truth. Inspect the overview and the local windows at proposed
onset/end boundaries. Each local plot retains original daily indices, major grid at
5 days and minor ticks at 1 day. Locate the actual abrupt transient edges, or the
departure into sustained raw motion and its arrest. Correct onset and end, reject
unsupported ranges, and retain all supported ranges in the COMPLETE final output.
The last active day is before the stationary plateau; the last transient day is before
the return edge. No need to change a correct estimate. Do not narrow an interval
merely because a local window is narrow; global context determines the whole range.
"""


class RangeEvent(Interval):
    kind: Literal["transient", "deformation"]
    status: Literal["confirmed", "uncertain"]
    evidence: str


class RangeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    events: list[RangeEvent]


def render(observation: dict, path: Path, window: tuple[int, int] | None = None):
    evidence = numerical_evidence(observation)
    length = len(evidence.values)
    left, right = window if window is not None else (0, length - 1)
    days = np.arange(left, right + 1)
    reference = evidence.values - evidence.residual
    figure, panels = plt.subplots(3, 2, figsize=(8, 6) if window else (13, 7.5), dpi=220,
                                  sharex=True)
    for axis in range(3):
        raw, residual = panels[axis]
        raw.plot(days, evidence.values[left:right + 1, axis], ".-", markersize=1.2, linewidth=.7)
        raw.plot(days, reference[left:right + 1, axis], "--", color="#c18437", linewidth=.7)
        residual.plot(days, evidence.residual[left:right + 1, axis], ".-", markersize=1.2,
                      linewidth=.7, color="#2c766c")
        residual.axhline(0, color="#555555", linewidth=.5)
        for panel in (raw, residual):
            if left <= evidence.background_end:
                panel.axvspan(left, min(right, evidence.background_end), color="#b4cfdf", alpha=.3)
            panel.set_ylabel(f"{'NEU'[axis]} (mm)")
            panel.set_xlim(left, right)
            panel.xaxis.set_major_locator(MultipleLocator(5 if window else 50))
            panel.xaxis.set_minor_locator(MultipleLocator(1 if window else 10))
            panel.grid(alpha=.25)
    panels[0, 0].set_title("Original displacement / dashed normal fit")
    panels[0, 1].set_title("Historical-normal residual (auxiliary)")
    for panel in panels[-1]:
        panel.set_xlabel("Original daily index (gaps retained)")
    figure.suptitle(f"Observed daily GNSS; days {left} to {right}; blue shade = normal prefix")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def prepare(observation: dict, work: Path, model: str, mode: str, overview: Path,
            coarse: dict | None = None):
    work.mkdir(parents=True, exist_ok=False)
    evidence = numerical_evidence(observation)
    metadata = {"length": len(evidence.values), "normal_prefix": [0, evidence.background_end],
                "analysis_period": [evidence.background_end + 1, len(evidence.values) - 1],
                "missing_indices": np.flatnonzero(~evidence.observed).tolist()}
    content = [{"type": "text", "text": json.dumps(metadata)}]
    images = []
    prompt = PROMPT
    if mode in ("range_visual", "range_zoom"):
        images.append(overview)
    elif mode == "range_text":
        values = []
        for day in range(len(evidence.values)):
            values.append([day, *evidence.values[day].tolist(), *evidence.residual[day].tolist()]
                          if evidence.observed[day] else [day, None, None, None, None, None, None])
        content.append({"type": "text", "text": json.dumps({
            "table_columns": ["day", "raw_N", "raw_E", "raw_U", "residual_N", "residual_E", "residual_U"],
            "daily_values_mm": values}, allow_nan=False, separators=(",", ":"))})
    if mode == "range_zoom":
        prompt += ZOOM_PROMPT
        content.append({"type": "text", "text": json.dumps({"your_coarse_ranges": coarse["range_decision"]})})
        seen = set()
        for event in coarse["range_decision"]["events"][:4]:
            for boundary in (event["start"], event["end"]):
                window = (max(0, boundary - 35), min(len(evidence.values) - 1, boundary + 35))
                if window in seen:
                    continue
                seen.add(window)
                path = work / f"boundary-{boundary}.png"
                render(observation, path, window)
                images.append(path)
    content.extend(image_item(path, True) for path in images)
    write_json(work / "request.json", payload(model, prompt, content, RangeDecision))
    write_json(work / "input-receipt.json", {"mode": mode, "images": {str(p): sha256(p) for p in images},
        "request_sha256": sha256(work / "request.json"), "coarse_sha256": sha256(work.parent / "range_visual/result.json")
        if coarse is not None else None})


def call(observation: dict, work: Path, settings: tuple) -> dict:
    receipt, content = send_request(read_json(work / "request.json"), work, settings)
    if receipt["status"] == "success":
        try:
            decision = RangeDecision.model_validate_json(content)
            if any(event.end >= len(observation["dates"]) or event.start <= observation["background_end"]
                   for event in decision.events):
                raise ValueError("range must remain inside analysis period")
            receipt["range_decision"] = decision.model_dump()
        except (ValueError, TypeError, KeyError) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
    write_json(work / "result.json", receipt)
    return receipt


def apply(observation: dict, numeric: dict, response: dict) -> dict:
    if response["status"] != "success":
        return {**numeric, "range_visual_status": response["status"], "range_fallback": "independent_numeric"}
    events = [copy.deepcopy(event) for event in numeric["prediction"]["events"]
              if event["kind"] in ("spike", "step")]
    for event in response["range_decision"]["events"]:
        events.append({**event, "phases": []})
    return {**rebuild(observation, numeric, events), "range_visual_status": response["status"],
            "range_decision": response["range_decision"], "type_scores": {}}
