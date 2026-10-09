"""Observed image inventory and evidence-locator diagnostics; no stage decisions."""

from __future__ import annotations

import json
import re

GROUNDING_NOTE = """

ACTUAL IMAGE INVENTORY AND EVIDENCE LOCATORS
The inventory lists the seven images attached to THIS request in exact order.
Ranges describe horizontal axes, not continuous valid observations. Missing gaps
and unsupported derivative estimates remain absent, never zero or interpolated.
R1 is the full raw record; R2/R3/R4 are overlapping views of that SAME record.
Do not use the start of the last local view as the start of every raw image.
M31/M61/M91 are full-record centered motion plots, NOT extra activity evidence.
For each frozen activity, inspect a raw view that covers the relevant interval
before interpreting rate evolution. Image coverage alone does not prove motion,
nonzero steadiness, a rate trend or a stage. Activity is already frozen and cannot
be overturned by an uncertain stage, an absent fit or a noisy auxiliary curve.

For EACH stage, keep the original evidence string but identify the raw image(s)
you actually used with exact tags [R1], [R2], [R3], [R4], and auxiliary images
with [M31], [M61], [M91]. Describe what those plots show within the stage dates.
At least one cited raw view should cover the complete proposed stage interval.
Cite only images actually inspected; these tags are locators, NOT confidence.
Do not label a stage merely to supply a citation. If evidence is unreadable,
weak, conflicting or unavailable, omit the stage and explain the unresolved
activity in stage_notes. Do not claim an interval is outside all raw images
unless its dates are outside ALL listed raw ranges. Do not introduce extra JSON
keys: stages and stage_notes and the existing stage schema remain unchanged.
No turning proposals, classified stages or reference answers are provided here.
The original definitions, sparse spans, half-open boundaries and masks still apply.
"""


def image_inventory(record, images, calendar_days):
    """Resolve explicit ranges from the frozen render metadata, never from labels."""
    if type(calendar_days) is not int or calendar_days < 1:
        raise ValueError("Expected a positive complete calendar length")
    if len(record["views"]) != 4:
        raise ValueError("Expected the four frozen raw views")
    inventory = []
    for index, view in enumerate(record["views"], 1):
        start, end = view["view"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end < calendar_days):
            raise ValueError("Raw view must have valid inclusive day endpoints")
        if any(panel["xlim"] != [start, end] for panel in view["audit"]["panels"]):
            raise ValueError("Raw range differs from actual frozen axes")
        inventory.append({"id": f"R{index}", "order": index, "image": view["image"],
                          "kind": "raw_NEU_displacement",
                          "axis_days_inclusive": [start, end]})
    if inventory[0]["axis_days_inclusive"] != [0, calendar_days - 1]:
        raise ValueError("First raw image must cover the full calendar")
    for index, window in enumerate((31, 61, 91), 5):
        inventory.append({"id": f"M{window}", "order": index,
                          "image": record["diagnostics"][str(window)]["image"],
                          "kind": "centered_motion_estimates", "window_days": window,
                          "axis_days_inclusive": [0, calendar_days - 1]})
    if [item["image"] for item in inventory] != images:
        raise ValueError("Inventory differs from actual seven-image order")
    return {"case_id": record["case_id"], "calendar_days": calendar_days,
            "images": inventory,
            "meaning": "image locators and axes only; not observed activity or stage evidence"}


def grounding_prompt(original, inventory):
    return original + GROUNDING_NOTE + json.dumps(inventory, separators=(",", ":"), allow_nan=False)


def citation_audit(answer, inventory):
    """Check locator syntax and temporal coverage, not scientific validity."""
    images = {item["id"]: item for item in inventory["images"]}
    stages = []
    for stage in answer["stages"]:
        citations = list(dict.fromkeys(re.findall(r"\[(R\d+|M\d+)\]", stage["evidence"])))
        unknown_ids = [name for name in citations if name not in images]
        raw_ids = [name for name in citations if name in images and name.startswith("R")]
        motion_ids = [name for name in citations if name in images and name.startswith("M")]
        covering_raw_ids = []
        for name in raw_ids:
            start, end = images[name]["axis_days_inclusive"]
            if start <= stage["start"] < stage["stop"] <= end + 1:
                covering_raw_ids.append(name)
        stages.append({"start": stage["start"], "stop": stage["stop"], "feature": stage["feature"],
                       "citations": citations, "unknown_ids": unknown_ids,
                       "raw_ids": raw_ids, "motion_ids": motion_ids,
                       "covering_raw_ids": covering_raw_ids,
                       "locator_and_coverage_pass": bool(covering_raw_ids and motion_ids and not unknown_ids)})
    return {"meaning": "mechanical locator/coverage check only; not accuracy or visual comprehension",
            "stages": stages, "proposed_stages": len(stages),
            "locator_and_coverage_pass": sum(item["locator_and_coverage_pass"] for item in stages),
            "unknown_image_id_stages": sum(bool(item["unknown_ids"]) for item in stages),
            "missing_raw_citation_stages": sum(not item["raw_ids"] for item in stages),
            "no_covering_raw_citation_stages": sum(not item["covering_raw_ids"] for item in stages),
            "missing_motion_citation_stages": sum(not item["motion_ids"] for item in stages)}
