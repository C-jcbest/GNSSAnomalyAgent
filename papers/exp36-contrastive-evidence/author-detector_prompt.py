from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

DETECTOR_ROLE = "You are an expert in **time series anomaly detection**."

DETECTOR_FINAL_OUTPUT = """
## Final output (after you stop calling tools)
1) Write `<think>...</think>` with brief reasoning.
2) Immediately after `</think>`, output **exactly one** JSON array (no markdown fences), e.g. `[]` or:
[{"interval": [start, end], "type": "name", "explanation": "..."}]
Rules:
- Use this segment's **global Index**.
- Use `[]` if no anomaly.
- Include `confidence` (integer **1 or 2**) for each anomaly.
- After the JSON array, output **nothing else**.
- Do not add any extra words, notes, comments, headings, markdown, code fences, or trailing text before or after the JSON array.
- The final answer must end exactly at the closing `]` of the JSON array.
""".strip()

def build_actor_replay_text(messages: list[Any]) -> str:
    if not messages:
        return ""

    user_prompt: str | None = None
    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                user_prompt = content
                break
        if isinstance(message, dict) and (message.get("role") == "user"):
            content = str(message.get("content") or "").strip()
            if content:
                user_prompt = content
                break

    def _sanitize_user_prompt(text: str) -> str:
        prompt = (text or "").strip()
        if not prompt:
            return ""
        for cut in (
            "\n\n## Your job in this stage",
            "\n## Your job in this stage",
            "\n\n## Output format (after you stop calling tools)",
            "\n## Output format (after you stop calling tools)",
        ):
            index = prompt.find(cut)
            if index != -1:
                prompt = prompt[:index].rstrip()
        if prompt.startswith("User:\n"):
            prompt = prompt[len("User:\n") :].lstrip()
        return prompt

    lines: list[str] = []
    if user_prompt:
        lines.append("User:")
        lines.append(_sanitize_user_prompt(user_prompt))

    id_to_name: dict[str, str] = {}
    for message in messages:
        tool_calls = None
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
        elif isinstance(message, dict) and (message.get("role") == "assistant"):
            tool_calls = message.get("tool_calls") or []
        if not tool_calls or not isinstance(tool_calls, list):
            continue
        for one in tool_calls:
            if not isinstance(one, dict):
                continue
            tool_id = str(one.get("id") or "").strip()
            tool_name = str(one.get("name") or "").strip()
            if tool_id and tool_name:
                id_to_name[tool_id] = tool_name

    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                lines.append("")
                lines.append("Assistant:")
                lines.append("(tool calls)")
                for one in tool_calls:
                    if not isinstance(one, dict):
                        continue
                    try:
                        args_text = json.dumps(one.get("args", {}), ensure_ascii=False)
                    except Exception:
                        args_text = str(one.get("args", {}))
                    lines.append(f"- name={one.get('name','')} args={args_text}")
            continue
        if isinstance(message, ToolMessage):
            tool_id = str(getattr(message, "tool_call_id", "") or "").strip()
            tool_name = id_to_name.get(tool_id, "tool")
            lines.append("")
            lines.append(f"Tool ({tool_name}):")
            lines.append(str(getattr(message, "content", "") or ""))
            continue
        if isinstance(message, dict) and (message.get("role") == "tool"):
            tool_id = str(message.get("tool_call_id") or "").strip()
            tool_name = id_to_name.get(tool_id, "tool")
            lines.append("")
            lines.append(f"Tool ({tool_name}):")
            lines.append(str(message.get("content") or ""))
            continue
    return "\n".join(lines).strip()
