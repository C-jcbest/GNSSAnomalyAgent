"""Task-specific evidence ownership, using the same saved visual responses as controls."""
from __future__ import annotations

import numpy as np

from gnss_research.schema import Prediction, event_masks, observed_values

DERIVED_METHODS = {
    "anchored_fusion": "guided_fusion",
    "anchored_no_local": "guided_no_local",
    "anchored_no_scores": "guided_no_scores",
}


def anchored_review(observation: dict, visual: dict, candidates: dict) -> dict:
    if visual["status"] != "success":
        return {"status": "failed", "error_type": "VisualDependencyFailed", "request_count": 0}
    numerical = candidates["numerical_prediction"]
    confirmed = [event for event in visual["prediction"]["events"]
                 if event["status"] == "confirmed"]
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    events = []
    for proposal in numerical["prediction"]["events"]:
        kind = proposal["kind"]
        if kind == "transient":
            continue
        supported = False
        for decision in confirmed:
            if decision["kind"] != kind:
                continue
            left = max(proposal["start"], decision["start"])
            right = min(proposal["end"], decision["end"])
            if left <= right and observed[left:right + 1].any():
                supported = True
                break
        if supported:
            events.append({**proposal,
                           "evidence": "Numerical localization retained; raw-curve visual "
                           "review confirms this morphology on overlapping observed days"})
    transient_windows = [window for window in candidates["windows"]
                         if window["inspection_kind"] == "transient"]
    for decision in confirmed:
        if decision["kind"] != "transient":
            continue
        if any(window["start"] <= decision["start"] <= decision["end"] <= window["end"]
               for window in transient_windows):
            events.append({**decision,
                           "evidence": "Visual shift-and-return boundaries within "
                           "numerically selected transient inspection window: " + decision["evidence"]})
    prediction = Prediction.model_validate({"events": events})
    masks = event_masks(prediction, observed)
    generic = np.any(np.column_stack(list(masks.values())), axis=1)
    return {"status": "success", "prediction": prediction.model_dump(),
            "generic_mask": generic.tolist(), "generic_score": None, "request_count": 0,
            "derivation": "same visual response; numeric point/activity anchoring, "
            "visual transient boundaries; no new model request"}
