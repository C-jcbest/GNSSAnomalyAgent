"""Bounded LangChain agent and matched review controls; observations only."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith import tracing_context
from pydantic import Field

from gnss_sim import localization, numerical
from gnss_sim.artifacts import write_json
from gnss_sim.detection import make_config
from gnss_sim.report import intervals
from gnss_sim.schemas import CaseInput, PointResult, RangeResult
from gnss_sim.visual import _unique_object

ARMS = ("candidate_visual", "fixed_fusion", "langchain_agent", "boundary_agent")
SYSTEM = (
    "Review candidate GNSS anomalies using the supplied observation plot and available evidence. "
    "Candidate dates are proposals, not truth. Assess every candidate independently. "
    "Keep a candidate only when its morphology and temporal extent match the requested task. "
    "Do not infer the number, duration, amplitude, axis or type of injected events. "
    "Never invent dates or candidate IDs. Final output must be exactly a JSON object "
    'mapping EVERY candidate ID to true (keep) or false (reject), e.g. {"c001":true}. '
    "No explanations or extra keys."
)
BOUNDARY_SYSTEM = SYSTEM.replace(
    "Keep a candidate only when its morphology and temporal extent match the requested task.",
    "Retain the observed active portion of a candidate when its morphology matches the task."
).replace(
    "Never invent dates or candidate IDs.",
    "Never invent candidates or extend a candidate beyond its proposed dates."
).replace(
    'mapping EVERY candidate ID to true (keep) or false (reject), e.g. {"c001":true}.',
    'mapping EVERY candidate ID to null (reject) or an inclusive [start,end] integer pair, '
    'e.g. {"c001":[100,150]}. To keep the whole candidate, return its original endpoints. '
    "To shrink it, both endpoints must remain inside that candidate. Do not split candidates. "
    "Shrink only when the observations support excluding part of the proposed interval; "
    "do not shorten every interval automatically. A stable residual offset is not active evolution."
)
TOOL_GUIDANCE = (
    "First request one or both numerical tools in ONE batch, at most one call per tool. "
    "Choose candidate IDs where that evidence is useful; all IDs may be selected. "
    "Then combine tool results with the plot and return your final decisions."
)
REVIEW_GUIDANCE = (
    "Give an initial decision for every candidate. A final review of the same evidence follows."
)
FINAL_GUIDANCE = (
    "Now recheck every candidate against the original plot, task and supplied evidence. "
    "Return the final JSON decisions. Do not call further tools."
)
LOCALIZATION_SYSTEM = (
    "Use the observed GNSS morphology and numerical boundary proposals to identify active ranges. "
    "Assess every candidate independently; proposals are not truth. Do not infer injection "
    "parameters, event counts, durations or types. Numerical scores are supporting evidence, "
    "not mandatory decisions. Reject ordinary fluctuations and isolated spikes as ranges. "
    "For gradual evolution stop at the stable residual plateau; for a temporary shifted level "
    "include the last shifted day. Return exactly one JSON object mapping EVERY candidate ID "
    'to null (reject), "original", "level", or "trend" (select its supplied boundary proposal). '
    "Never return date pairs, booleans, new IDs or extra keys. No explanations."
)
LOCALIZATION_GUIDANCE = (
    "First call localize_boundaries exactly once with ALL candidate IDs. It returns both shape "
    "proposals and raw samples. Then use the original plot and those "
    "observations to select a proposal or reject for every ID. Do not call further tools."
)
POINT_EXPLANATION_SYSTEM = SYSTEM + (
    " For each Point candidate, compare two explanations: a genuine isolated anomaly, "
    "and an occasional extreme observation within ordinary background variability. "
    "An isolated shape or exceeding a candidate-generation threshold establishes a proposal, "
    "not independent confirmation. Compare the candidate with nearby raw observations and "
    "the rest of the observed record. Use numerical strength when available together with "
    "visual context to assess which explanation is better supported. A zoomed spike alone "
    "is insufficient evidence. Apply the same assessment independently to every candidate. "
    "Do not invent an amplitude cutoff, expected event count or required number of rejections. "
    "Keep a genuine anomaly when the observations support it; reject a candidate when ordinary "
    "background variation better explains it. Preserve the same boolean JSON output contract."
)
LOCAL_POINT_ARMS = ("local_point_agent", "local_candidate_visual", "hypothesis_point_agent", "hypothesis_visual")


def candidates(point, visual_range, temporary, task):
    """Frozen support: N points; union of V ranges and N temporary intervals."""
    rows = [point] if task == "point" else [visual_range, temporary]
    if any(r is None or r.status != "success" for r in rows):
        raise ValueError("candidate producer failed")
    spans = {}
    for row, source in zip(rows, ["numerical"] if task == "point" else ["visual", "numerical"]):
        for axis in numerical.AXES:
            for value in getattr(row.predictions, axis):
                start, end = (value, value) if task == "point" else value
                spans.setdefault((axis, start, end), []).append(source)
    return [{"id": f"c{i:03d}", "axis": axis, "start": start, "end": end,
             "sources": spans[(axis, start, end)]}
            for i, (axis, start, end) in enumerate(sorted(spans), 1)]


def decisions(content, items, shrink=False):
    value = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {c["id"] for c in items}:
        raise ValueError("exact candidate IDs required")
    for c in items:
        selected = value[c["id"]]
        if shrink:
            if selected is not None and (
                not isinstance(selected, list) or len(selected) != 2
                or any(type(day) is not int for day in selected)
                or not c["start"] <= selected[0] <= selected[1] <= c["end"]
            ):
                raise ValueError("null or inclusive integer endpoints inside the candidate required")
        elif type(selected) is not bool:
            raise ValueError("boolean decisions required")
    return value


def result_from_decisions(cid, task, arm, items, choices):
    shrink = arm == "boundary_agent" and task == "range"
    if shrink:
        decisions(json.dumps(choices), items, shrink=True)
    prediction = {}
    for axis in numerical.AXES:
        days = set()
        for c in items:
            selected = choices[c["id"]]
            if c["axis"] == axis and (selected is not None if shrink else selected):
                start, end = selected if shrink else (c["start"], c["end"])
                days.update(range(start, end + 1))
        prediction[axis] = sorted(days) if task == "point" else intervals(days)
    model = PointResult if task == "point" else RangeResult
    return model(case_id=cid, method=arm, status="success", predictions=prediction)


class Evidence:
    """Tools have no dataset path, truth object or generation metadata."""

    def __init__(self, case: CaseInput, items, parameters):
        self.values = np.asarray(case.displacement_mm)
        self.items = {c["id"]: c for c in items}
        self.parameters = parameters

    def select(self, candidate_ids):
        if (not candidate_ids or len(candidate_ids) != len(set(candidate_ids))
                or not set(candidate_ids) <= self.items.keys()):
            raise ValueError("unknown or duplicate candidate ID")
        return [self.items[cid] for cid in candidate_ids]

    def candidate_statistics(self, candidate_ids):
        output = {}
        for c in self.select(candidate_ids):
            x = self.values[:, numerical.AXES.index(c["axis"])]
            s, e = c["start"], c["end"]
            t = self.parameters["thresholds"][c["axis"]]
            cfg = self.parameters["config"]
            score = numerical.spike_scores(x, cfg)[s:e+1]
            slope = numerical.trend_scores(x, cfg)[s:e+1]
            def median(values):
                return round(float(np.median(values)), 4) if len(values) else None
            output[c["id"]] = {
                "before_7d_median_mm": median(x[max(0, s-7):s]),
                "inside_median_mm": median(x[s:e+1]),
                "after_7d_median_mm": median(x[e+1:min(365, e+8)]),
                "maximum_spike_to_threshold": round(float(score.max()/t["spike"]), 4),
                "median_raw_31d_slope_mm_per_day": median(slope),
                "maximum_abs_raw_31d_slope_mm_per_day": round(float(abs(slope).max()), 4),
                "calibrated_trend_threshold_on_processed_series": t["trend_high"],
                "note": "Raw slopes are contextual; the detector threshold uses processed data.",
            }
        return output

    def local_samples(self, candidate_ids):
        output = {}
        for c in self.select(candidate_ids):
            x = self.values[:, numerical.AXES.index(c["axis"])]
            s, e = c["start"], c["end"]
            days = sorted({*range(max(0, s-3), min(365, s+4)),
                           *range(max(0, e-3), min(365, e+4)),
                           *np.linspace(s, e, 5, dtype=int).tolist()})
            output[c["id"]] = [[int(d), round(float(x[d]), 4)] for d in days]
        return output

    def tools(self):
        @tool
        def candidate_statistics(candidate_ids: list[str]) -> dict:
            """Get local level, isolated-spike strength and raw slope summaries for chosen IDs."""
            return self.candidate_statistics(candidate_ids)

        @tool
        def local_samples(candidate_ids: list[str]) -> dict:
            """Read observed [day, mm] pairs near proposed boundaries and inside chosen IDs."""
            return self.local_samples(candidate_ids)

        return [candidate_statistics, local_samples]


class StrictQwen(BaseChatModel):
    """LangChain model adapter with raw contract checks and no HTTP/SDK retries."""

    client: Any = Field(exclude=True, repr=False)
    endpoint: str = Field(exclude=True, repr=False)
    api_key: str = Field(exclude=True, repr=False)
    directory: Path
    items: list[dict]
    shrink: bool = False
    calls: int = 0
    corrections: int = 0
    boundary_packages: dict | None = None
    decision_system: str = SYSTEM

    @property
    def _llm_type(self):
        return "strict-qwen-observation-review"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools],
                         tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        wire = convert_to_openai_messages(messages)
        definitions = kwargs.get("tools", [])
        while True:
            if self.calls >= 3:
                raise RuntimeError("per-task request budget exhausted")
            self.calls += 1
            payload = {"model": "qwen3.8-flash", "temperature": 0, "max_tokens": 8192,
                       "enable_thinking": False, "vl_high_resolution_images": False,
                       "messages": wire}
            if definitions:
                payload.update(tools=definitions, tool_choice="required")
            else:
                payload["response_format"] = {"type": "json_object"}
            prefix = self.directory / str(self.calls)
            write_json(prefix.with_name(prefix.name + "-started.json"), {
                "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                "model": payload["model"], "tool_phase": bool(definitions),
                "correction": self.corrections > 0})
            start = time.perf_counter()
            record = {"status": "failed", "error": None, "usage": None}
            structure_error = False
            try:
                response = self.client.post(self.endpoint + "/chat/completions", json=payload,
                                           headers={"Authorization": "Bearer " + self.api_key})
                record["http_status"] = response.status_code
                response.raise_for_status()
                body = response.json()
                record.update(response=body, usage=body.get("usage"))
                choice = body["choices"][0]
                msg = choice["message"]
                if (body.get("model") != "qwen3.8-flash" or msg.get("reasoning_content")
                        or choice.get("finish_reason") not in ("stop", "tool_calls")):
                    raise RuntimeError("model/thinking/finish contract violation")
                try:
                    calls = msg.get("tool_calls") or []
                    if definitions:
                        if not 1 <= len(calls) <= (1 if self.boundary_packages is not None else 2):
                            raise ValueError("one or two tool calls required")
                        names, identifiers, parsed = set(), set(), []
                        for call in calls:
                            name = call["function"]["name"]
                            args = json.loads(call["function"]["arguments"],
                                              object_pairs_hook=_unique_object)
                            allowed = (("localize_boundaries",) if self.boundary_packages is not None
                                       else ("candidate_statistics", "local_samples"))
                            if (name not in allowed
                                    or name in names or call["id"] in identifiers
                                    or not isinstance(call["id"], str) or not call["id"]
                                    or set(args) != {"candidate_ids"}
                                    or not isinstance(args["candidate_ids"], list)
                                    or not args["candidate_ids"]
                                    or any(not isinstance(v, str) for v in args["candidate_ids"])
                                    or len(set(args["candidate_ids"])) != len(args["candidate_ids"])
                                    or not set(args["candidate_ids"]) <= {c["id"] for c in self.items}):
                                raise ValueError("invalid tool request")
                            if self.boundary_packages is not None and set(args["candidate_ids"]) != set(self.boundary_packages):
                                raise ValueError("localize_boundaries must cover ALL candidate IDs")
                            names.add(name)
                            identifiers.add(call["id"])
                            parsed.append({"name": name, "args": args, "id": call["id"]})
                        answer = AIMessage(content=msg.get("content") or "", tool_calls=parsed)
                    else:
                        if calls or choice["finish_reason"] != "stop":
                            raise ValueError("final response must contain decisions only")
                        if self.boundary_packages is not None:
                            localization.decisions(msg["content"], self.boundary_packages)
                        else:
                            decisions(msg["content"], self.items, self.shrink)
                        answer = AIMessage(content=msg["content"])
                    record["status"] = "success"
                except (ValueError, TypeError, KeyError) as exc:
                    structure_error = True
                    record["error"] = "structure:" + type(exc).__name__
            except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
                record["error"] = "transport_or_contract:" + type(exc).__name__
            record["seconds"] = time.perf_counter() - start
            write_json(prefix.with_name(prefix.name + "-completed.json"), record)
            if record["status"] == "success":
                return ChatResult(generations=[ChatGeneration(message=answer)])
            if not structure_error or self.corrections or self.calls >= 3:
                raise RuntimeError(record["error"])
            self.corrections += 1
            wire = [*wire, {"role": "user", "content": (
                "Your completed response violated the output structure. Correct it once. "
                + ((LOCALIZATION_GUIDANCE if definitions else LOCALIZATION_SYSTEM)
                   if self.boundary_packages is not None else
                   TOOL_GUIDANCE if definitions else self.decision_system))}]


def review(case, image, items, parameters, task, arm, client, base, key, directory,
           boundary_packages=None, extra_images=()):
    """Two semantic model turns plus at most one structural correction in total."""
    if arm not in (*ARMS, "localization_agent", "window_agent", *LOCAL_POINT_ARMS):
        raise ValueError("unknown review arm")
    if arm in LOCAL_POINT_ARMS and task != "point":
        raise ValueError("Local Point arms only review Point; Range must use frozen shared results")
    if extra_images and (task != "point" or arm not in LOCAL_POINT_ARMS):
        raise ValueError("Local context images are only allowed in registered Point arms")
    if items and arm in LOCAL_POINT_ARMS and not extra_images:
        raise ValueError("Local Point images required")
    directory.mkdir(parents=True, exist_ok=False)
    if not items:
        return result_from_decisions(case.case_id, task, arm, [], {}), {"calls": 0, "tools": []}
    evidence = Evidence(case, items, parameters)
    task_text = make_config()["prompts"][task].split("Return {")[0]
    text = task_text + "\nCandidates: " + json.dumps(items)
    adaptive = arm in ("langchain_agent", "boundary_agent", "localization_agent", "window_agent",
                       "local_point_agent", "hypothesis_point_agent")
    shrink = arm == "boundary_agent" and task == "range"
    localize = arm in ("localization_agent", "window_agent") and task == "range"
    system = (LOCALIZATION_SYSTEM if localize else BOUNDARY_SYSTEM if shrink else
              POINT_EXPLANATION_SYSTEM if arm in ("hypothesis_point_agent", "hypothesis_visual") else SYSTEM)
    if localize and (boundary_packages is None or set(boundary_packages) != set(evidence.items)):
        raise ValueError("Registered boundary evidence required for every candidate")
    text += "\n" + (LOCALIZATION_GUIDANCE if localize else
                    TOOL_GUIDANCE if adaptive else REVIEW_GUIDANCE)
    if arm == "fixed_fusion":
        ids = list(evidence.items)
        text += "\nNumerical evidence: " + json.dumps({
            "candidate_statistics": evidence.candidate_statistics(ids),
            "local_samples": evidence.local_samples(ids)})
    user = HumanMessage(content=[{"type": "text", "text": text}, *[{
        "type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                               base64.b64encode(value).decode("ascii")}}
        for value in (image, *extra_images)]])
    model = StrictQwen(client=client, endpoint=base, api_key=key, directory=directory,
                       items=items, shrink=shrink,
                       decision_system=system,
                       boundary_packages=boundary_packages if localize else None)
    tools = evidence.tools() if adaptive else []
    if localize:
        @tool
        def localize_boundaries(candidate_ids: list[str]) -> dict:
            """Get observed level/trend boundary proposals and raw samples for ALL candidates."""
            if set(candidate_ids) != set(boundary_packages):
                raise ValueError("Every candidate must be examined")
            return {cid: {key: value for key, value in package.items() if key != "rule_choice"}
                    for cid, package in boundary_packages.items()}
        tools = [localize_boundaries]
    turns = 0

    @wrap_model_call
    def limit_tools(request, handler):
        nonlocal turns
        turns += 1
        if turns > 2:
            raise RuntimeError("semantic turn budget exhausted")
        if turns == 2:
            request = request.override(tools=[])
        return handler(request)

    # Disable remote tracing: credentials and observation content stay out of tracing services.
    with tracing_context(enabled=False):
        agent = create_agent(model, tools=tools,
                             system_prompt=system,
                             middleware=[limit_tools])
        output = agent.invoke({"messages": [user]}, {"recursion_limit": 12})
        if not adaptive:
            output = agent.invoke({"messages": [*output["messages"], HumanMessage(FINAL_GUIDANCE)]},
                                  {"recursion_limit": 12})
    messages = output["messages"]
    write_json(directory / "trace.json", [m.model_dump(mode="json", exclude={"id"})
                                          for m in messages if m.type != "human"])
    selected = (localization.decisions(messages[-1].content, boundary_packages) if localize
                else decisions(messages[-1].content, items, shrink))
    tool_calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    row = (localization.result(case.case_id, arm, boundary_packages, selected) if localize
           else result_from_decisions(case.case_id, task, arm, items, selected))
    return row, {
        "calls": model.calls, "corrections": model.corrections, "tools": tool_calls}
