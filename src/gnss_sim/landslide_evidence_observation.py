"""Match a local display to the evidence that first made its target reviewable."""
from __future__ import annotations

from gnss_sim.landslide_adaptive_review import review_priorities


def evidence_display_choices(initial, features, base42_evidence, selected84_targets, policy):
    """Use block summaries only for targets below the existing42 trigger threshold.

    This decision uses raw evidence and the initial answer, never local answers,
    labels or scores. Established targets keep raw views to avoid the observed
    suppression of weak tails under universal block display.
    """
    windows = [(target["start"], target["stop"]) for target in selected84_targets]
    priorities, _ = review_priorities(initial, features, base42_evidence, windows, policy)
    choices = []
    for target, priority in zip(selected84_targets, priorities):
        if target["conflict_days"] < policy.minimum_conflict_days:
            raise ValueError("Only already selected84 targets may receive a display choice")
        choices.append({"index": target["index"], "start": target["start"], "stop": target["stop"],
                        "base42_conflict_days": priority["conflict_days"],
                        "selected84_conflict_days": target["conflict_days"],
                        "arm": "narrow_blocks" if priority["conflict_days"] < policy.minimum_conflict_days
                               else "narrow_raw"})
    return choices
