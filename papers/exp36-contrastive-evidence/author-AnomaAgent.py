from __future__ import annotations

import logging
from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from AnomaMind.core.Actor import Actor
from AnomaMind.core.Detector import Detector
from AnomaMind.core.Evaluator import Evaluator
from AnomaMind.core.Locator import Locator
from AnomaMind.core.finalize import finalize_detector_output
from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.core.llm_config import LLMConfig, create_llm, default_locator_llm_config
from AnomaMind.core.workflowstate import WorkflowState

logger = logging.getLogger(__name__)

MAX_EVAL_REDO_ITERATIONS = 3


class AnomaAgent:
    """LangGraph: Locator -> Actor -> Detector -> [Evaluator loop] -> finalize."""

    def __init__(
        self,
        max_turns: int = 8,
        locator_config: LLMConfig | None = None,
        actor_config: LLMConfig | None = None,
        detector_config: LLMConfig | None = None,
        evaluator_config: LLMConfig | None = None,
        knowledgeBase: Optional[KnowledgeBase] = None,
        max_eval_redo_iterations: int = MAX_EVAL_REDO_ITERATIONS,
        enable_reflection: bool = True,
    ):
        self.max_turns = max_turns
        self.max_eval_redo_iterations = max(0, int(max_eval_redo_iterations))
        self.enable_reflection = bool(enable_reflection)
        self.knowledgeBase = knowledgeBase if knowledgeBase else KnowledgeBase()

        loc_cfg = locator_config or default_locator_llm_config()
        self.locator_llm = create_llm(loc_cfg)
        self.locator = Locator(llm=self.locator_llm, knowledge_base=self.knowledgeBase)

        act_cfg = actor_config or detector_config or LLMConfig()
        self.actor_llm = create_llm(act_cfg)
        self.actor = Actor(knowledge_base=self.knowledgeBase, llm=self.actor_llm)

        det_cfg = detector_config or actor_config or LLMConfig()
        self.detector_llm = create_llm(det_cfg)
        self.detector = Detector(knowledge_base=self.knowledgeBase, llm=self.detector_llm)

        self.evaluator: Evaluator | None = None
        if self.enable_reflection:
            ev_cfg = evaluator_config or actor_config or detector_config or LLMConfig()
            self.evaluator = Evaluator(llm=create_llm(ev_cfg))

    def locator_node(self, state: WorkflowState) -> dict:
        return self.locator.locate(state)

    def actor_node(self, state: WorkflowState) -> dict:
        return self.actor.act(state, max_turns=self.max_turns)

    def detector_node(self, state: WorkflowState) -> dict:
        return self.detector.detect(state)

    def evaluator_node(self, state: WorkflowState) -> dict:
        if self.evaluator is None:
            return {"redo": False, "suggestion": ""}
        return self.evaluator.evaluate(state)

    def prepare_redo_node(self, state: WorkflowState) -> dict:
        return {
            "actor_messages": [],
            "detector_messages": [],
            "final_output": {},
            "iteration_count": int(state.get("iteration_count") or 0) + 1,
        }

    def finalize(self, state: WorkflowState) -> WorkflowState:
        return self._finalize_output(state)

    def _route_after_evaluator(
        self, state: WorkflowState
    ) -> Literal["prepare_redo", "finalize"]:
        if state.get("has_error"):
            return "finalize"
        if not state.get("redo"):
            return "finalize"
        if int(state.get("iteration_count") or 0) > self.max_eval_redo_iterations:
            return "finalize"
        return "prepare_redo"

    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node("locator", self.locator_node)  # type: ignore
        workflow.add_node("actor", self.actor_node)  # type: ignore
        workflow.add_node("detector", self.detector_node)  # type: ignore
        workflow.add_node("finalize", self.finalize)  # type: ignore
        workflow.set_entry_point("locator")
        workflow.add_edge("locator", "actor")
        workflow.add_edge("actor", "detector")
        if not self.enable_reflection:
            workflow.add_edge("detector", "finalize")
            workflow.add_edge("finalize", END)
            return workflow.compile()

        workflow.add_node("evaluator", self.evaluator_node)  # type: ignore
        workflow.add_node("prepare_redo", self.prepare_redo_node)  # type: ignore
        workflow.add_edge("detector", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            self._route_after_evaluator,
            {"prepare_redo": "prepare_redo", "finalize": "finalize"},
        )
        workflow.add_edge("prepare_redo", "actor")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _finalize_output(self, state: WorkflowState) -> WorkflowState:
        return finalize_detector_output(state)
