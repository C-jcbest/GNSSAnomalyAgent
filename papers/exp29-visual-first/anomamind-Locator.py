# -*- coding: utf-8 -*-
"""
Visual coarse localization for ``AnomaAgent``: **WorkflowState in → segment updates out** (writes visual_* only).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.core.workflowstate import WorkflowState

PLAIN_PREVIEW_NAME = "segment_preview_plain.png"
FINAL_PREVIEW_NAME = "segment_final.png"

LOCATOR_SYSTEM = (
    "You are a vision-language assistant for time-series screening. "
    "Read axes carefully; follow index conventions exactly. "
    "Reply with a short <locator_description> block then a JSON array as instructed."
)

def _png_to_data_url(png_path: Path) -> str:
    raw = png_path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _load_meta_index(segment_dir: Path) -> tuple[int, int, dict[str, Any]]:
    meta_path = segment_dir / "meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta_info.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idx = meta.get("index") or {}
    start = int(idx["start"])
    end = int(idx["end"])
    if end <= start:
        raise ValueError(f"invalid index in meta: start={start} end={end}")
    return start, end, meta


def _strip_code_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _parse_intervals_json(text: str) -> tuple[list[dict[str, Any]], Optional[str]]:

    cleaned = _strip_code_fences(text)
    m = re.search(r"\[\s*(\{|\])", cleaned, flags=re.DOTALL)
    if not m:
        return [], "no JSON array found"
    start = m.start()
    depth = 0
    end_pos = -1
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if end_pos < 0:
        return [], "unclosed JSON array"
    try:
        arr = json.loads(cleaned[start:end_pos])
    except json.JSONDecodeError as e:
        return [], f"json decode error: {e}"
    if not isinstance(arr, list):
        return [], "root is not a list"
    out: list[dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        iv = item.get("interval")
        if isinstance(iv, list) and len(iv) >= 1:
            try:
                a = int(iv[0])
                b = int(iv[1]) if len(iv) > 1 else a
            except (TypeError, ValueError):
                continue
            if b < a:
                a, b = b, a
            out.append(
                {
                    "interval": [a, b],
                    "rationale": str(item.get("rationale", item.get("note", "")))[:500],
                }
            )
            continue
        if "start" in item and "end" in item:
            try:
                a = int(item["start"])
                b = int(item["end"])
            except (TypeError, ValueError):
                continue
            if b < a:
                a, b = b, a
            out.append(
                {
                    "interval": [a, b],
                    "rationale": str(item.get("rationale", item.get("note", "")))[:500],
                }
            )
    return out, None


def _strip_locator_description_prefix(raw: str) -> tuple[str, str]:
    """If the model emits <locator_description>…</locator_description>, split brief text from trailing JSON."""
    m = re.search(
        r"<locator_description>\s*(.*?)\s*</locator_description>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return "", raw
    brief = (m.group(1) or "").strip()
    rest = raw[m.end() :].strip()
    return brief, rest

def _build_user_prompt(
    *,
    start: int,
    end: int,
    dataset_type: str,
    visual_knowledge: str,
    has_reference_strip: bool = False,
) -> str:
    # Quick test: edit prompt text below; only {start} / {end} / {dataset_type} are substituted per segment.
    ref_block = ""
    if has_reference_strip:
        ref_block = (
            "\n## Composite figure (reference + segment)\n"
            "- The figure is **stacked vertically**: the **upper** region is an **Anomaly Reference** "
            "mosaic (representative anomaly shapes from clustering on this dataset). "
            "The **lower** region is the **target segment** line chart you must judge.\n"
            "- **Use only the lower panel’s x-axis** for your JSON intervals: global indices **"
            f"{start}** … **{end - 1}**. Ignore x-axes in the upper reference strip.\n"
        )
    return f"""The attached image is a **line chart** of one preprocessed time-series segment.
{ref_block}
## Read the axes carefully (avoid index mistakes)
- **X-axis = global integer index**, from **{start}** (left) through **{end - 1}** (right). Every tick label is a real index; align each candidate interval to what you **see** on the axis—do **not** shift ranges by guesswork.
- **Y-axis = preprocessed value** (scaled / detrended), not raw physical units. Judge **shape** (sudden jumps), not slow drift.

## What counts as an anomaly here (strict)
{visual_knowledge}

## Dataset context
- Declared dataset type: **{dataset_type}**

## Your task
List **coarse** candidate anomaly index ranges (global indices) that match the rules above.
- You need try your best to match the possible anomaly with reference
- Prefer **few, wide** intervals; merge nearby bursts that belong to one event.
- If nothing abrupt appears, say so in the description and use an empty JSON array.



## Output format
1) First, a visual summary  inside **exactly** these tags (no markdown fences inside):
<locator_description>
Your short description for the series.
</locator_description>

2) Immediately after, output **only** a JSON array (no markdown fences), each element one object, for example:
[
  {{"interval": [12045, 12080], "rationale": "detailed reason and description (eg. how the anomaly begin, in what shape and value, and recover the what"}},
  {{"interval": [12100, 12140], "rationale": "..."}}
]
If no anomalies: <locator_description>...</locator_description> then [] on the next line.
"""


class Locator:

    def __init__(self, llm=None, knowledge_base: KnowledgeBase | None = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or KnowledgeBase()

    @staticmethod
    def resolve_plain_preview_path(segment_dir: Path) -> Path:
        segment_dir = segment_dir.resolve()
        p = segment_dir / PLAIN_PREVIEW_NAME
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {PLAIN_PREVIEW_NAME} under {segment_dir}. "
                "Run scripts_new/preprocess.py to generate segment previews."
            )
        return p

    def locate(self, state: WorkflowState) -> dict[str, Any]:
        seg = Path(state["path"]).resolve()
        png_path = self.resolve_plain_preview_path(seg)
        start, end, meta = _load_meta_index(seg)
        dtype = (state.get("type") or meta.get("type") or "UNKNOWN")
        dtype = str(dtype).strip().upper()
        visual_knowledge = self.knowledge_base.get_locator_visual_knowledge(dtype)
        user_text = _build_user_prompt(
            start=start,
            end=end,
            dataset_type=dtype,
            visual_knowledge=visual_knowledge,
        )
        data_url = _png_to_data_url(png_path)

        human = HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        resp = self.llm.invoke([SystemMessage(content=LOCATOR_SYSTEM), human])
        raw_text = getattr(resp, "content", "") or ""
        if isinstance(raw_text, list):
            parts = []
            for block in raw_text:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            raw_text = "".join(parts)

        brief, json_part = _strip_locator_description_prefix(raw_text)
        intervals, _err = _parse_intervals_json(json_part)
        pairs: list[list[int]] = []
        items: list[dict[str, Any]] = []
        for it in intervals:
            iv = it.get("interval")
            if not isinstance(iv, list) or len(iv) < 2:
                continue
            a, b = int(iv[0]), int(iv[1])
            a = max(a, start)
            b = min(b, end - 1)
            if a <= b:
                pairs.append([a, b])
                items.append(
                    {
                        "interval": [a, b],
                        "rationale": str(it.get("rationale", "") or "")[:2000],
                    }
                )

        return {
            "visual_description": str(brief or "").strip(),
            "visual_localization": pairs,
            "visual_localization_items": items,
        }
