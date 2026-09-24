import json
from copy import deepcopy
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import Field

from .contracts import CHANNELS, Prediction, StrictModel, Window, prediction_schema
from .detectors import NAMES, NUMERICAL_NAMES, detect, quality, segments
from .fusion import EVIDENCE_PROMPT, SUSTAINED_PROMPT, merge_branches
from .models import ModelClient
from .plotting import render, review_bounds
from .storage import digest, read_json

METHODS = (
    *NUMERICAL_NAMES,
    "visual",
    "daily_union",
    "fixed",
    "split_fusion",
    "split_fusion_evidence",
    "rule",
    "generic",
    "lma",
    "lma_no_vision",
    "lma_no_review",
)


class Choice(StrictModel):
    tool: Literal["hampel", "cusum", "iforest", "visual", "stop"]
    reason: str = Field(max_length=1000)


def load_capabilities(path: str | None) -> dict | None:
    if path is None:
        return None
    card = read_json(Path(path))
    if card.get("source_split") not in ("development", "validation"):
        raise ValueError("capabilities must come from development/validation, never test")
    if set(card.get("best_by_missing", {})) != {"complete", "missing"}:
        raise ValueError("invalid capability card")
    if any(name not in (*NAMES, "visual") for name in card["best_by_missing"].values()):
        raise ValueError("invalid capability tool")
    return card


class Executor:
    def __init__(self, config: dict, model: ModelClient | None, card: dict | None = None):
        self.config, self.model, self.card = config, model, card
        self.tools: list[dict] = []
        self.numerical_cache: dict = {}

    def visual(
        self,
        window: Window,
        prior: Prediction | None = None,
        *,
        sustained: bool = False,
        evidence: bool = False,
    ) -> Prediction:
        if self.model is None:
            raise ValueError("visual model is not configured")
        explicit_grid = self.config.get("visual_protocol") == "global-mask-v2"
        if evidence and not sustained:
            raise ValueError("evidence prompt requires independent sustained analysis")
        if sustained and (prior is not None or not explicit_grid or window.sampling_hours != 24):
            raise ValueError(
                "independent sustained analysis requires daily global input without prior"
            )
        pngs = [render(window, None if explicit_grid else prior)]
        if prior and prior.events:
            pngs.append(render(window, prior, review_bounds(window, prior)))
        prompt = (
            f"检测长度为 {len(window.values)} 的 GNSS 窗口中的异常，"
            f"每个索引间隔{window.sampling_hours}小时。"
            "仅依据观测；可输出空事件。图的阴影如存在，仅为候选，可能有错。"
        )
        if sustained:
            prompt += EVIDENCE_PROMPT if evidence else SUSTAINED_PROMPT
        elif window.sampling_hours == 24:
            prompt += (
                "这是每日北京时间15点观测。检查孤立偏离、异常波动、台阶、"
                "持续速率变化和加速；不要将任何非零斜率一概视为异常。"
                "定位受影响区间而不仅是起点；持续偏移直到恢复或窗口结束。缺测不算异常。"
            )
        if prior:
            prompt += "\n数值/先前候选（请独立检查，可增删或调整）：" + prior.model_dump_json()
        schema = Prediction
        if explicit_grid:
            grid = {
                "samples": len(window.values),
                "legal_index": [0, len(window.values) - 1],
                "sampling_hours": window.sampling_hours,
                "unit": "mm",
                "index_to_beijing_date": [
                    t.astimezone(ZoneInfo("Asia/Shanghai")).isoformat() for t in window.timestamps
                ],
                "missing_inclusive_intervals": {
                    channel: [
                        [int(a), int(b - 1)]
                        for a, b in segments(~np.isfinite(window.array()[:, c]))
                    ]
                    for c, channel in enumerate(CHANNELS)
                },
            }
            prompt = (
                (
                    "独立检查唯一的无候选全局图，按下述持续事件职责分析。"
                    if sustained
                    else "先独立检查第一张无候选阴影的全局图，识别局部偏离与持续变化；"
                    "再核对数值候选及局部图（如有）。候选不是完备清单，可增加候选以外的事件。"
                )
                + "缺测位置以以下观测元数据为准，不从图中猜测；区间可跨缺口，但缺口不提供异常证据。\n"
                + json.dumps(grid, ensure_ascii=False)
                + "\n"
                + prompt
            )
            schema = prediction_schema(len(window.values))
        result = self.model.ask(prompt, schema, pngs)
        result.mask(len(window.values))
        return result

    def tool(self, window: Window, name: str, prior: Prediction | None = None) -> Prediction:
        if name not in (*NUMERICAL_NAMES, "visual"):
            raise ValueError("tool not allowed")
        trace = {"tool": name, "status": "started"}
        self.tools.append(trace)
        if name == "visual":
            result = self.visual(window, prior)
        else:
            trace["diagnostics"] = {}
            key = digest(
                [
                    window.model_dump(mode="json"),
                    name,
                    self.config["detectors"],
                    self.config["seed"],
                ]
            )
            cached = self.config.get("cache_numerical", False)
            if cached and key in self.numerical_cache:
                result, trace["diagnostics"] = deepcopy(self.numerical_cache[key])
                trace["cache_hit"] = True
            else:
                result = detect(
                    window,
                    name,
                    self.config["detectors"],
                    self.config["seed"],
                    trace["diagnostics"],
                )
                if cached:
                    self.numerical_cache[key] = deepcopy((result, trace["diagnostics"]))
                    trace["cache_hit"] = False
        trace.update(status=result.status, result=result.model_dump())
        return result

    def run(self, window: Window, method: str) -> Prediction:
        self.tools = []
        if method not in METHODS:
            raise ValueError("unknown method")
        q = quality(window)
        if max(q["observed_per_channel"]) < self.config["detectors"]["min_samples"]:
            return Prediction(status="insufficient", reason="too few observations")
        if method in NUMERICAL_NAMES:
            return self.tool(window, method)
        if method == "visual":
            return self.tool(window, "visual")
        if method == "daily_union":
            raise ValueError("daily_union derives from the paired run; use run_experiment")
        if method == "fixed":
            candidate = self.tool(window, self.config["fixed_detector"])
            return self.tool(window, "visual", candidate)
        if method in ("split_fusion", "split_fusion_evidence"):
            candidate = self.tool(window, "hampel")
            evidence = method == "split_fusion_evidence"
            trace = {
                "tool": "visual_sustained_evidence" if evidence else "visual_sustained",
                "status": "started",
            }
            self.tools.append(trace)
            result = self.visual(window, sustained=True, evidence=evidence)
            trace.update(status=result.status, result=result.model_dump())
            return merge_branches(candidate, result, len(window.values))
        if method == "rule":
            if not self.card:
                raise ValueError("rule policy requires measured capability card")
            bucket = "missing" if q["missing_fraction"] else "complete"
            name = self.card["best_by_missing"][bucket]
            candidate = self.tool(window, name)
            # A minimal inspectable rule. Freeze after development, not hidden hand tuning.
            if candidate.events and q["missing_fraction"] > 0 and name != "visual":
                return self.tool(window, "visual", candidate)
            return candidate
        if self.model is None:
            raise ValueError("agent model is not configured")
        if method.startswith("lma") and self.card is None:
            raise ValueError("LMA requires development capability card")
        allowed = list(NAMES) + ([] if method == "lma_no_vision" else ["visual"])
        context = {
            "quality": q,
            "tools": {
                "hampel": "局部稳健残差离群检测",
                "cusum": "增量累计变化检测",
                "iforest": "局部统计特征的孤立森林",
                "visual": "读取全局/局部曲线图",
            },
            "allowed_tools": allowed,
        }
        if method.startswith("lma"):
            context["development_evidence"] = self.card
            context["domain_guidance"] = "注意三轴噪声差异、缺口附近候选；无法判因时不要确认滑坡。"
        choice = self.model.ask(
            "为当前数据选择一个首轮工具，不允许 stop。\n" + json.dumps(context, ensure_ascii=False),
            Choice,
        )
        if choice.tool not in allowed:
            raise ValueError("agent chose an unavailable first tool")
        first = self.tool(window, choice.tool)
        if method == "lma_no_review":
            return first
        context["first_tool"] = choice.tool
        context["first_prediction"] = first.model_dump()
        decision = self.model.ask(
            "是否需要复核？无需复核选择 stop，否则选择不同的可用工具。\n"
            + json.dumps(context, ensure_ascii=False),
            Choice,
        )
        if decision.tool == "stop":
            return first
        if decision.tool not in allowed or decision.tool == choice.tool:
            raise ValueError("invalid or repeated review tool")
        second = self.tool(window, decision.tool, first)
        if decision.tool == "visual":
            return second
        # No hidden visual analysis: combine only evidence actually requested by the agent.
        final = self.model.ask(
            "根据两次工具证据给出最终候选，可增删区间，不得超出窗口。\n"
            + json.dumps(
                {
                    "samples": len(window.values),
                    "sampling_hours": window.sampling_hours,
                    "quality": q,
                    "first": first.model_dump(),
                    "review": second.model_dump(),
                },
                ensure_ascii=False,
            ),
            Prediction,
        )
        final.mask(len(window.values))
        return final
