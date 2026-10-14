"""Task-specific evidence ownership, using the same saved visual responses as controls."""
from __future__ import annotations

import numpy as np

from gnss_research.schema import Prediction, event_masks, observed_values

DERIVED_METHODS = {
    "anchored_fusion": "guided_fusion",
    "anchored_no_local": "guided_no_local",
    "anchored_no_scores": "guided_no_scores",
    "boundary_fusion": "guided_fusion",
    "boundary_no_local": "guided_no_local",
    "boundary_no_scores": "guided_no_scores",
    "boundary_numeric_points": "guided_fusion",
    "ranked_boundary_fusion": "guided_fusion",
    "ranked_no_local": "guided_no_local",
    "ranked_no_scores": "guided_no_scores",
    "morphology_fusion": "guided_fusion",
    "morphology_no_local": "guided_no_local",
    "morphology_no_scores": "guided_no_scores",
    "morphology_fixed_activity": "guided_fusion",
    "morphology_global_fixed": "guided_no_local",
    "morphology_step_local": "guided_step_local",
    "morphology_step_fixed": "guided_step_local",
    "localized_fusion": "guided_fusion",
    "localized_global": "guided_no_local",
    "localized_step_local": "guided_step_local",
    "localized_fixed": "guided_fusion",
    "localized_global_fixed": "guided_no_local",
    "localized_step_fixed": "guided_step_local",
}


def localized_review(observation: dict, visual: dict, candidates: dict,
                     revise_activity: bool = True) -> dict:
    """Let visual step semantics confirm a window, then retain its numerical onset.

    This explicitly replaces EXP04's exact-day intersection. A visual date must fall
    in a step inspection window; if windows overlap, the nearest RF center owns it.
    The date estimator uses only observations and the normal reference, never truth.
    """
    result = morphology_review(observation, visual, candidates, revise_activity)
    if result["status"] != "success":
        return result
    events = [event for event in result["prediction"]["events"] if event["kind"] != "step"]
    windows = [window for window in candidates["windows"]
               if window["inspection_kind"] == "step"]
    localizations = {item["window_id"]: item for item in candidates["step_localizations"]}
    emitted = set()
    for decision in visual["prediction"]["events"]:
        if decision["kind"] != "step" or decision["status"] != "confirmed":
            continue
        supporting = [window for window in windows
                      if window["start"] <= decision["start"] <= window["end"]]
        if not supporting:
            continue
        window = min(supporting, key=lambda item: abs(item["peak_index"] - decision["start"]))
        index = localizations[window["id"]]["index"]
        if index is None or index in emitted:
            continue
        events.append({**decision, "start": index, "end": index,
                       "evidence": "Visual persistent-step confirmation within " + window["id"]
                       + "; onset from station-standardized common-trend level-jump fit: "
                       + decision["evidence"]})
        emitted.add(index)
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    result.update(prediction=prediction.model_dump(),
                  generic_mask=np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
                  derivation="same response; numerical spikes, visual window-confirmed fitted "
                  "step dates, " + ("visual activity boundaries" if revise_activity
                                     else "numeric activity boundaries"))
    return result


def morphology_review(observation: dict, visual: dict, candidates: dict,
                      revise_activity: bool = True) -> dict:
    """Use RF votes for spikes, ranked visual confirmation for steps, and visual ranges.

    Sparse RF step votes may miss persistent offsets, while visual spike confirmation
    can promote ordinary noise peaks. Assign these two shapes different decision rules.
    The fixed-activity ablation changes only the ownership of deformation boundaries.
    """
    result = ranked_boundary_review(observation, visual, candidates)
    if result["status"] != "success":
        return result
    events = [event for event in result["prediction"]["events"] if event["kind"] != "spike"]
    proposals = candidates["numerical_prediction"]["prediction"]["events"]
    events.extend({**event, "evidence": "RF spike decision; visual review owns other morphologies"}
                  for event in proposals if event["kind"] == "spike")
    if not revise_activity:
        anchored = anchored_review(observation, visual, candidates)
        events = [event for event in events if event["kind"] != "deformation"]
        events.extend(event for event in anchored["prediction"]["events"]
                      if event["kind"] == "deformation")
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    result.update(prediction=prediction.model_dump(),
                  generic_mask=np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
                  derivation="same response; RF spikes, exact ranked visual steps, "
                  + ("visual activity boundaries" if revise_activity else "numeric activity boundaries"))
    return result


def ranked_boundary_review(observation: dict, visual: dict, candidates: dict) -> dict:
    """Rank locates sparse points; exact raw-curve visual confirmation decides presence."""
    result = boundary_review(observation, visual, candidates)
    if result["status"] != "success":
        return result
    events = [event for event in result["prediction"]["events"]
              if event["kind"] not in ("spike", "step")]
    confirmed_points = {(event["kind"], event["start"]): event
                        for event in visual["prediction"]["events"]
                        if event["status"] == "confirmed" and event["kind"] in ("spike", "step")}
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    emitted = set()
    for window in candidates["windows"]:
        kind, peak = window["inspection_kind"], window["peak_index"]
        key = (kind, peak)
        if key in confirmed_points and observed[peak] and key not in emitted:
            decision = confirmed_points[key]
            events.append({**decision, "evidence":
                "Ranked classifier peak and exact-day visual morphology agree; no RF vote cutoff: "
                + decision["evidence"]})
            emitted.add(key)
    prediction = Prediction.model_validate({"events": events})
    masks = event_masks(prediction, observed)
    result.update(prediction=prediction.model_dump(),
                  generic_mask=np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
                  derivation="same response; ranked peaks with exact-day visual point confirmation, "
                  "visually revised numeric-supported activity boundaries")
    return result


def boundary_review(observation: dict, visual: dict, candidates: dict,
                    numerical_points: bool = False) -> dict:
    """Visual displacement boundaries may revise a numerically supported activity.

    Overlap gates morphology, never proves the boundaries. The visual model must inspect
    raw displacement and arrest; this rule cannot guarantee its judgment is correct.
    The point ablation removes visual vetoes without changing RF votes or their dates.
    """
    result = anchored_review(observation, visual, candidates)
    if result["status"] != "success":
        return result
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    proposals = candidates["numerical_prediction"]["prediction"]["events"]
    decisions = [event for event in visual["prediction"]["events"]
                 if event["status"] == "confirmed"]
    events = [event for event in result["prediction"]["events"]
              if event["kind"] != "deformation"]
    if numerical_points:
        events = [event for event in events if event["kind"] not in ("spike", "step")]
        events.extend({**event, "evidence": "Numerical point decision without visual veto"}
                      for event in proposals if event["kind"] in ("spike", "step"))
    for decision in decisions:
        if decision["kind"] != "deformation":
            continue
        for proposal in proposals:
            if proposal["kind"] != "deformation":
                continue
            left = max(proposal["start"], decision["start"])
            right = min(proposal["end"], decision["end"])
            if left <= right and observed[left:right + 1].any():
                events.append({**decision, "evidence":
                    "Visual raw-displacement boundaries revise an overlapping numeric activity: "
                    + decision["evidence"]})
                break
    prediction = Prediction.model_validate({"events": events})
    masks = event_masks(prediction, observed)
    return {"status": "success", "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0,
            "derivation": "same visual response; revised activity boundaries, "
            + ("numeric points without veto" if numerical_points else "exact-day point consensus")}


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
