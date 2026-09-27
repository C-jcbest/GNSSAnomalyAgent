"""Frozen N/E/U image renderer and strict P6 answer parser."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np

from gnss_sim.schemas import CaseInput, PointResult, RangeResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

AXES = ("N", "E", "U")
DAYS = 365
TICKS = (0, 60, 120, 180, 240, 300, 364)
FENCE = re.compile(r"\A```(?:json)?\r?\n(.*?)\r?\n```\Z", re.DOTALL)


def render_case(case: CaseInput, output: Path) -> None:
    """Render only signed input displacement; the interface cannot accept truth."""
    series = np.asarray(case.displacement_mm, dtype=np.float64)
    if series.shape != (DAYS, 3) or not np.isfinite(series).all():
        raise ValueError("expected 365 finite N/E/U displacement samples")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 0.8, "savefig.facecolor": "white"}):
        fig, panels = plt.subplots(3, 1, figsize=(12, 8), dpi=150, sharex=True,
                                   facecolor="white")
        try:
            for index, (axis, panel) in enumerate(zip(AXES, panels)):
                panel.plot(np.arange(DAYS), series[:, index], color="#233b58", linewidth=1)
                panel.set_xlim(0, DAYS - 1)
                panel.set_xticks(TICKS)
                panel.tick_params(axis="x", labelbottom=True)
                panel.set_ylabel("Displacement (mm)")
                panel.set_title(axis, loc="left", fontweight="bold")
                panel.margins(y=0.08)
            panels[-1].set_xlabel("Day index")
            fig.subplots_adjust(left=0.12, right=0.985, bottom=0.075, top=0.965,
                                hspace=0.42)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, format="png", dpi=150, facecolor="white")
        finally:
            plt.close(fig)


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
