"""One recorded provider request; no implicit retries and no secret-bearing receipts."""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from gnss_research.files import write_json


def send_request(payload: dict, work: Path, settings: tuple[str, str, str]) -> tuple[dict, str | None]:
    base, key, model = settings
    start = time.perf_counter()
    receipt = {"status": "failed", "requested_model": model, "request_count": 1,
               "retry_count": 0}
    content = None
    try:
        with httpx.Client(timeout=120, transport=httpx.HTTPTransport(retries=0)) as client:
            response = client.post(base + "/chat/completions", json=payload,
                                   headers={"Authorization": "Bearer " + key})
            receipt["http_status"] = response.status_code
            response.raise_for_status()
            raw = response.json()
            write_json(work / "response.json", raw)
            receipt["served_model"] = raw.get("model")
            receipt["usage"] = raw.get("usage")
            content = raw["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("text model response required")
            receipt["status"] = "success"
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as error:
        receipt["error_type"] = type(error).__name__
    receipt["elapsed_seconds"] = time.perf_counter() - start
    return receipt, content
