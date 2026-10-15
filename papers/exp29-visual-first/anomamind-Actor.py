# -*- coding: utf-8 -*-
"""
Actor: tool-calling-only stage.

It sees Locator hints, decides what tools to call, and stops once it has enough
evidence for the downstream Detector stage.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from AnomaMind.core.Detector import DETECTOR_ROLE
from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.core.tool import create_detector_tools, get_detector_tools_prompt_block
from AnomaMind.core.utils import get_column_names
from AnomaMind.core.workflowstate import WorkflowState
ACTOR_STAGE_OBJECTIVE = """
## Your job in this stage
- You are the **Actor** stage. Your only job is to decide which tools to call and inspect tool results.
- When you have enough evidence, write a short plain-text evidence summary.
- Do not call grep for those interval without evidence to support.
""".strip()

ACTOR_FINAL_OUTPUT = """
## Output format (after you stop calling tools)
- Output a short plain-text evidence summary
""".strip()


def _append_redo_suggestion(prompt: str, state: WorkflowState) -> str:
    if not state.get("redo"):
        return prompt
    suggestion = str(state.get("suggestion") or "").strip()
    if not suggestion:
        return prompt
    return f"{prompt.rstrip()}\n\nsuggestion: {suggestion}"


def _normalize_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _default_first_round_tool_calls(state: WorkflowState) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = [
        {"name": "stats", "args": {}, "id": "auto_stats"},
        {"name": "value_zscore", "args": {"thresholds": [3.5]}, "id": "auto_value_zscore"},
        {"name": "diff_zscore", "args": {"thresholds": [5.0]}, "id": "auto_diff_zscore"},
    ]

    localization = state.get("visual_localization") or []
    if not isinstance(localization, list) or not localization:
        return tool_calls

    try:
        starts = [int(one[0]) for one in localization if isinstance(one, (list, tuple)) and len(one) >= 2]
        ends = [int(one[1]) for one in localization if isinstance(one, (list, tuple)) and len(one) >= 2]
        if not starts or not ends:
            return tool_calls
        start = min(starts)
        end = max(ends)
        grep_start = max(0, start - 10)
        grep_end = max(end, end + 10)
        tool_calls.append(
            {"name": "grep", "args": {"start": grep_start, "end": grep_end}, "id": "auto_grep"}
        )
    except Exception:
        return tool_calls
    return tool_calls


def _actor_context_block(state: WorkflowState, knowledge_base: KnowledgeBase) -> str:
    """
    Make Actor prompt look like the legacy (single-agent) detector prompt:
    tool description + domain knowledge + locator hints.
    """
    description = str(state.get("visual_description") or "").strip()
    dataset_key = str(state.get("type") or "UNKNOWN").strip().upper()
    localization = state.get("visual_localization") or []
    if not isinstance(localization, list):
        localization = []
    localization_json = json.dumps(localization, ensure_ascii=False)

    localization_items = state.get("visual_localization_items") or []
    if not isinstance(localization_items, list):
        localization_items = []
    localization_items_json = json.dumps(localization_items, ensure_ascii=False)

    kb_raw = (
        knowledge_base.get_agent_knowledge("planning_agent", dataset_key)
        or knowledge_base.get_agent_knowledge("detector", dataset_key)
    )
    domain_knowledge = str(kb_raw).strip() if kb_raw else "(none)"

    # Provide explicit global index bounds so the Actor chooses correct grep ranges.
    index_min: Optional[int] = None
    index_max: Optional[int] = None
    try:
        data = state.get("data")
        if data is not None:
            ts_col, _val_col = get_column_names(data)
            if ts_col in data.columns:
                idxs = data[ts_col].astype(int)
                if len(idxs) > 0:
                    index_min = int(idxs.min())
                    index_max = int(idxs.max())
            else:
                # Fallback to positional indices if timestamp column is missing.
                n = int(len(data))
                if n > 0:
                    index_min = 0
                    index_max = n - 1
    except Exception:
        index_min = None
        index_max = None

    seg_range = (
        f"Segment Index: start:{index_min}, end: {index_max}\n"
        if index_min is not None and index_max is not None
        else ""
    )
    return "".join(
        [
            seg_range,
            "Here is domain knowledge:\n",
            f"{domain_knowledge}\n\n",
            "Locator (vision, coarse index ranges)\n",
            "Treat the Locator's visual localization as hint for where anomalies are. Use zscore/grep tools to confirm, reject, or refine those localized intervals based on domain kwnoledge.\n",
            f"VisualDescription: {description or '(none)'}\n",
            f"VisualLocalization (index pairs): {localization_json}\n",
            f"VisualLocalizationRationale (interval + rationale from Locator JSON): {localization_items_json}",
        ]
    )
class Actor:
    def __init__(self, knowledge_base: KnowledgeBase, llm=None):
        self.knowledge_base = knowledge_base
        self.llm = llm

    def _build_messages(self, state: WorkflowState) -> list[BaseMessage]:
        tool_description = get_detector_tools_prompt_block()
        prompt = "\n\n".join(
            part
            for part in (
                DETECTOR_ROLE,
                "Here is tool description:\n" + tool_description.strip(),
                _actor_context_block(state, self.knowledge_base).strip(),
                ACTOR_STAGE_OBJECTIVE,
                ACTOR_FINAL_OUTPUT,
            )
            if part
        )
        prompt = _append_redo_suggestion(prompt, state)
        return [HumanMessage(content=prompt)]

    def act(self, state: WorkflowState, *, max_turns: int = 8) -> dict[str, Any]:
        if state.get("has_error", False):
            return {}
        if self.llm is None:
            return {
                "has_error": True,
                "error_message": "actor llm is not configured",
            }

        seg_path = str(state.get("path") or "").strip()
        data = state["data"]
        tools = create_detector_tools(data, segment_folder_path=seg_path)
        tools_by_name = {tool.name: tool for tool in tools}
        llm_tools = self.llm.bind_tools(tools)

        messages = list(state.get("actor_messages") or [])
        if not messages:
            messages = self._build_messages(state)

        default_val_thresholds = [3.5]
        default_diff_thresholds = [5.0]

        for _turn_index in range(max(1, int(max_turns))):
            try:
                ai = llm_tools.invoke(messages)
                if not isinstance(ai, AIMessage):
                    content = getattr(ai, "content", ai)
                    if not isinstance(content, str):
                        content = str(content or "")
                    ai = AIMessage(content=content)
            except Exception as exc:
                err = str(exc)
                return {
                    "actor_messages": messages,
                    "has_error": True,
                    "error_message": err,
                }

            tool_calls = getattr(ai, "tool_calls", None) or []
            messages.append(ai)

            if not tool_calls:
                if _turn_index == 0:
                    tool_calls = _default_first_round_tool_calls(state)
                else:
                    break

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    name = str(tool_call.get("name") or "").strip()
                    tool_call_id = str(tool_call.get("id") or "")
                    args = _normalize_tool_args(tool_call.get("args"))
                else:
                    name = str(getattr(tool_call, "name", "") or "").strip()
                    tool_call_id = str(getattr(tool_call, "id", "") or "")
                    args = _normalize_tool_args(getattr(tool_call, "args", None))

                tool_fn = tools_by_name.get(name)
                if tool_fn is None:
                    body = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
                else:
                    try:
                        if name == "stats":
                            body = tool_fn.invoke({})
                        elif name == "grep":
                            body = tool_fn.invoke(
                                {
                                    "start": int(args.get("start", 0)),
                                    "end": int(args.get("end", 0)),
                                }
                            )
                        elif name in {"value_zscore", "diff_zscore"}:
                            thresholds = args.get("thresholds")
                            if not thresholds:
                                thresholds = (
                                    default_val_thresholds
                                    if name == "value_zscore"
                                    else default_diff_thresholds
                                )
                            body = tool_fn.invoke({"thresholds": thresholds})
                        else:
                            body = tool_fn.invoke(args)
                    except Exception as exc:
                        body = json.dumps({"error": str(exc)}, ensure_ascii=False)
                # Some providers (e.g., Gemini) require tool response to include the tool name.
                messages.append(ToolMessage(content=body, tool_call_id=tool_call_id, name=name or None))

        return {
            "actor_messages": messages,
        }
