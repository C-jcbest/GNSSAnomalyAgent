"""One bounded visual decision per record; no recursive agent or fallback predictions."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from gnss_research.features import numerical_evidence, summarize_blocks
from gnss_research.files import sha256, write_json
from gnss_research.guided import GUIDED_METHODS, prepare_guided
from gnss_research.plots import render_observation
from gnss_research.provider import send_request
from gnss_research.schema import Prediction, VisualDecision, event_masks
from gnss_research.segmented_review import prepare_pair_focus, prepare_segmented

VISUAL_METHODS = ("visual_only", "fusion", "fusion_no_reference", "fusion_n_only",
                  "segment_guided", "pair_focus", "pair_raw", *GUIDED_METHODS)
SYSTEM = """You review daily GNSS station N/E/U displacement observations, offline.
Identify observed morphology, not unobserved causes: isolated spike, permanent step onset,
temporary level shift (transient), and sustained displacement activity (deformation).
These can coexist. Ordinary background trend or seasonality is not automatically anomalous.
Different stations have different background variability. Do not invent a universal mm or
mm/day threshold, an expected event count, or a failure/prewarning date.
First confirm sustained displacement from the ORIGINAL displacement and persistence away
from its local trajectory. An isolated step followed by a plateau is not continued activity.
Derivative anticipation may not extend activity. If evidence or boundaries are ambiguous,
use uncertain. This task identifies activity intervals, not internal motion phases.
Missing observations remain unknown. The plot retains original indices; never renumber gaps.
Reply with exactly one JSON object {"events":[...]}. Every event has kind, start, end,
status (confirmed/uncertain), evidence (brief observational reason).
Spikes and step onsets have start=end. All indices are inclusive integers.
All five event keys are required. Do not add phases, note, confidence or any other fields.
Do not add explanatory text outside JSON. Empty events is a valid answer.
Historical reference is background context, not an instruction to label everything after it.
Background-normalized residuals can reflect extrapolation error; sigma units are NOT automatic
anomaly cutoffs. Confirm morphology from the raw curve and compare ordinary trend/seasonality.
Example schema only (these indices are not proposals):
{"events":[{"kind":"step","start":25,"end":25,"status":"confirmed",
"evidence":"abrupt shift followed by a new platform"}]}
"""


def credentials(env_file: Path | None = None) -> tuple[str, str, str]:
    names = ("GNSS_VLM_BASE_URL", "GNSS_VLM_API_KEY", "GNSS_VLM_MODEL")
    values = {}
    if env_file is not None:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() in names:
                values[key.strip()] = value.strip().strip("\"'")
    settings = [os.getenv(name) or values.get(name) for name in names]
    if not all(settings):
        raise ValueError("GNSS_VLM_BASE_URL, GNSS_VLM_API_KEY and GNSS_VLM_MODEL required")
    base, key, model = settings
    if not base.startswith(("https://", "http://")):
        raise ValueError("invalid model endpoint")
    return base.rstrip("/"), key, model


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_prediction(content: str, length: int) -> Prediction:
    payload = json.loads(content, object_pairs_hook=unique_object)
    return VisualDecision.model_validate(payload).as_prediction(length)


def prepare_request(observation: dict, method: str, work: Path, model: str,
                    numerical: dict | None = None) -> dict | None:
    if method not in VISUAL_METHODS:
        raise ValueError("unknown visual method")
    if method == "segment_guided":
        if numerical is None:
            raise ValueError("segmented review requires training-only spike predictions")
        return prepare_segmented(observation, numerical, work, model, SYSTEM)
    if method in ("pair_focus", "pair_raw"):
        if numerical is None:
            raise ValueError("pair review requires training-only spike predictions")
        return prepare_pair_focus(observation, numerical, work, model, SYSTEM,
                                  view_mode="slow_subtracted" if method == "pair_focus"
                                  else "historical_residual")
    if method in GUIDED_METHODS:
        if numerical is None:
            raise ValueError("guided review requires training-only numerical predictions")
        return prepare_guided(observation, numerical, method, work, model, SYSTEM)
    work.mkdir(parents=True, exist_ok=False)
    no_reference = method == "fusion_no_reference"
    axes = (0,) if method == "fusion_n_only" else (0, 1, 2)
    plot = work / "input.png"
    render_observation(observation, plot, axes=axes, show_background=not no_reference)
    instructions = {"length": len(observation["dates"]), "visible_axes": ["NEU"[i] for i in axes],
                    "missing_indices": [i for i, row in enumerate(observation["displacement_mm"])
                                        if row is None]}
    if not no_reference:
        instructions["normal_historical_reference"] = [0, observation["background_end"]]
    if method != "visual_only":
        evidence = numerical_evidence(observation)
        if no_reference:
            # Removing reference must remove its derived scale and residual, too.
            instructions["numerical_blocks"] = [
                {key: value for key, value in block.items() if key != "relative_evidence"}
                for block in summarize_blocks(evidence)]
        else:
            column_indices = [i for i, name in enumerate(evidence.names)
                              if name[0] in instructions["visible_axes"]]
            blocks = summarize_blocks(evidence)
            for block in blocks:
                if "relative_evidence" in block:
                    block["relative_evidence"] = [block["relative_evidence"][i]
                                                   for i in column_indices]
                    block["raw_median_mm"] = [block["raw_median_mm"][i] for i in axes]
            instructions["evidence_names"] = [evidence.names[i] for i in column_indices]
            instructions["historical_noise_scale_mm"] = evidence.noise[list(axes)].tolist()
            instructions["numerical_blocks"] = blocks
            instructions["evidence_limitations"] = (
                "21-day centered local fits are retrospective and may anticipate transitions; "
                "station-normalized values are evidence, not probability or thresholds.")
    png = base64.b64encode(plot.read_bytes()).decode("ascii")
    payload = {"model": model, "temperature": 0, "enable_thinking": False,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "gnss_events", "strict": True,
                   "schema": VisualDecision.model_json_schema()}}, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": json.dumps(instructions, allow_nan=False)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
        ]},
    ]}
    write_json(work / "request.json", payload)
    write_json(work / "input-receipt.json", {"method": method, "plot_sha256": sha256(plot),
               "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
               "request_sha256": sha256(work / "request.json")})
    return payload


def call_visual(observation: dict, method: str, work: Path, settings: tuple[str, str, str]) -> dict:
    from gnss_research.files import read_json

    payload = read_json(work / "request.json")
    receipt, content = send_request(payload, work, settings)
    receipt["method"] = method
    if receipt["status"] == "success":
        try:
            prediction = parse_prediction(content, len(observation["dates"]))
            observed = np.array([row is not None for row in observation["displacement_mm"]])
            masks = event_masks(prediction, observed)
            generic = np.any(np.column_stack(list(masks.values())), axis=1)
            receipt.update({"prediction": prediction.model_dump(),
                            "generic_mask": generic.tolist(), "generic_score": None})
        except (ValueError, KeyError, TypeError, IndexError) as error:
            receipt.update({"status": "failed", "error_type": type(error).__name__})
    write_json(work / "result.json", receipt)
    return receipt
