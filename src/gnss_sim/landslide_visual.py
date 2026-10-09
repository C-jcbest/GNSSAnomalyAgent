"""Bounded, replayable visual activity localization and gated evolution inference."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.visual import _unique_object, credentials

ACTIVITY_TASK = """Review the observed multi-year GNSS displacement record retrospectively.
Locate sustained cumulative displacement activity, including weak/slow movement when visible.
Activity may start or stop anywhere, recur, accelerate, decelerate or become steady.
A stationary elevated plateau AFTER movement is no longer activity. Isolated spikes,
a single abrupt instrument-like step, noise, or a derivative peak alone do not confirm
sustained displacement. First require evidence of continuing change in raw displacement.
Do not infer generating parameters, event counts, landslide failure, or unobserved movement.
Use uncertain intervals where the raw displacement is inconclusive. Missing gaps do not
compress time. All interval endpoints are INCLUSIVE integer day indices, using the printed
horizontal axis. Return sorted, non-overlapping intervals, and never use indices outside
0..{last}. Confirmed and uncertain intervals must be disjoint. Empty lists are valid.
Respond with a JSON object matching the requested schema, without Markdown or explanations.
"""
EVOLUTION_TASK = """Evolution labels: A=increasing displacement rate, S=approximately steady
nonzero rate, D=decreasing rate. Low speed is NOT automatically S: slow movement can also
accelerate/decelerate. Do not label A because speed is high; assess how it changes over time.
Centered derivative estimates may respond before raw movement starts and are auxiliary only.
Do not force a label when evidence is weak or near transitions; leave those days unlabelled.
Do not include isolated measurement spikes in a definite phase.
Every stage endpoint is INCLUSIVE. Adjacent stages MUST NOT share a day: use
[100,149,"A"],[150,200,"S"], NEVER [100,150,"A"],[150,200,"S"].
"""


def interval_mask(intervals, days):
    if not isinstance(intervals, list):
        raise ValueError("Intervals must be a JSON list")
    mask = np.zeros(days, dtype=bool)
    previous_end = -1
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("An interval requires two endpoints")
        start, end = interval
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < days:
            raise ValueError("Invalid inclusive calendar endpoints")
        if start <= previous_end:
            raise ValueError("Intervals overlap or are not sorted")
        mask[start:end + 1] = True
        previous_end = end
    return mask


def parse_activity(payload, observed):
    if not isinstance(payload, dict) or set(payload) != {"activity", "uncertain"}:
        raise ValueError("Expected exactly activity and uncertain lists")
    active = interval_mask(payload["activity"], len(observed))
    uncertain = interval_mask(payload["uncertain"], len(observed))
    if np.any(active & uncertain):
        raise ValueError("Activity and uncertainty must be disjoint")
    activity = np.zeros(len(observed), dtype=int)
    activity[active] = 1
    activity[uncertain | ~observed] = -1
    return activity


def parse_stages(payload, activity, observed, gated=True):
    if not isinstance(payload, dict) or set(payload) != {"stages"} or not isinstance(payload["stages"], list):
        raise ValueError("Expected exactly a stages list")
    stage = np.full(len(activity), -1, dtype=int)
    stage[activity == 0] = 0
    previous_end = -1
    codes = {"A": 1, "S": 2, "D": 3}
    for item in payload["stages"]:
        if (not isinstance(item, list) or len(item) != 3
                or not isinstance(item[2], str) or item[2] not in codes):
            raise ValueError("Stage must be [start,end,A|S|D]")
        mask = interval_mask([item[:2]], len(activity))
        if item[0] <= previous_end:
            raise ValueError("Stage intervals overlap or are unsorted")
        if gated and np.any(mask & observed & (activity != 1)):
            raise ValueError("Stage extends beyond confirmed displacement; no automatic clipping")
        stage[mask & observed] = codes[item[2]]
        previous_end = item[1]
    stage[~observed] = -1
    prediction = ActivityPrediction(activity.copy(), stage, gated=gated)
    prediction.validate(len(activity))
    return prediction


class VisualRequests:
    def __init__(self, directory: Path, env_file: Path, model: str, budget: int):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.base, self.key = credentials(env_file)
        self.model = model
        self.budget = budget

    def ask(self, request_id, prompt, images):
        if len(list(self.directory.glob("*.started.json"))) >= self.budget:
            raise RuntimeError("Frozen model request budget exhausted")
        content = [{"type": "text", "text": prompt}]
        for path in images:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048,
                   "enable_thinking": False, "response_format": {"type": "json_object"}}
        receipt = {"request_id": request_id, "model": self.model, "prompt": prompt,
                   "image_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in images},
                   "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                   "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048, "no_retries": True}
        stem = self.directory / request_id
        with stem.with_suffix(".started.json").open("x", encoding="utf-8") as target:
            json.dump(receipt, target, ensure_ascii=False, indent=2)
        started = time.monotonic()
        try:
            with httpx.Client(timeout=150) as client:
                response = client.post(self.base + "/chat/completions", json=payload,
                                       headers={"Authorization": "Bearer " + self.key})
            receipt["http_status"] = response.status_code
            if response.status_code != 200:
                try:
                    provider_error = response.json().get("error", {})
                    message = str(provider_error.get("message", ""))
                    receipt["provider_error"] = message.replace(self.key, "[redacted]").replace(self.base, "[redacted]")[:300]
                except ValueError:
                    receipt["provider_error"] = "Non-JSON error response"
                raise RuntimeError(f"Provider HTTP {response.status_code}")
            body = response.json()
            choice = body["choices"][0]
            receipt.update(usage=body.get("usage", {}), returned_model=body.get("model"),
                           output=choice["message"]["content"], finish_reason=choice.get("finish_reason"))
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete model response")
            result = json.loads(receipt["output"], object_pairs_hook=_unique_object)
            receipt["status"] = "returned_json"
            return result
        except Exception as error:
            receipt.update(status="failed", error_type=type(error).__name__)
            raise RuntimeError(f"Visual request failed: {request_id}; see receipt") from None
        finally:
            receipt["seconds"] = time.monotonic() - started
            stem.with_suffix(".response.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
