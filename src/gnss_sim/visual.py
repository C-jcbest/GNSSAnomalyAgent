"""Strict visual output parsing and local credential access."""
import json
import os
import re
from pathlib import Path

from gnss_sim.schemas import PointResult, RangeResult

AXES = ("N", "E", "U")
FENCE = re.compile(r"\A```(?:json)?\r?\n(.*?)\r?\n```\Z", re.DOTALL)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_result(response: str, case_id: str, task: str,
                 method: str = "visual-v1") -> PointResult | RangeResult:
    """Only accept a complete JSON object or one enclosing JSON code fence."""
    if task not in ("point", "range"):
        raise ValueError("unknown visual task")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("empty model response")
    body = response.strip()
    if body.startswith("```"):
        match = FENCE.fullmatch(body)
        if match is None:
            raise ValueError("invalid JSON fence")
        body = match.group(1)
    payload = json.loads(body, object_pairs_hook=_unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValueError(f"invalid JSON constant: {value}")))
    if not isinstance(payload, dict) or set(payload) != set(AXES):
        raise ValueError("response must have exactly N/E/U keys")
    if any(not isinstance(payload[axis], list) for axis in AXES):
        raise ValueError("each axis prediction must be a list")
    model = PointResult if task == "point" else RangeResult
    return model(case_id=case_id, method=method, status="success", predictions=payload)


def failed_result(case_id: str, task: str,
                  method: str = "visual-v1") -> PointResult | RangeResult:
    model = PointResult if task == "point" else RangeResult
    return model(case_id=case_id, method=method, status="failed",
                 predictions={axis: [] for axis in AXES})


def credentials(env_file: Path | None) -> tuple[str, str]:
    values = {}
    if env_file is not None:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() in ("QWEN_BASE_URL", "QWEN_API_KEY"):
                values[key.strip()] = value.strip().strip("\"'")
    base = os.getenv("QWEN_BASE_URL") or values.get("QWEN_BASE_URL")
    key = os.getenv("QWEN_API_KEY") or values.get("QWEN_API_KEY")
    if not base or not key:
        raise ValueError("QWEN_BASE_URL and QWEN_API_KEY are required")
    if not base.startswith(("https://", "http://")):
        raise ValueError("invalid QWEN_BASE_URL")
    return base.rstrip("/"), key
