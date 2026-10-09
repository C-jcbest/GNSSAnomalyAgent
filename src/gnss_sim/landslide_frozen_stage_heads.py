"""Stage drafts on immutable author activity, with explicit evidence support."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from gnss_sim.landslide_reference_interface import partition_sha256
from gnss_sim.schemas import StrictModel

FEATURE_CODES = {-1: "unknown", 0: "none", 1: "acceleration",
                 2: "steady_motion", 3: "deceleration"}
MOTION_FEATURES = {"acceleration", "steady_motion", "deceleration"}
METHODS = ("rule", "xgboost", "visual")


class StageSpan(StrictModel):
    start: int = Field(strict=True, ge=0)
    stop: int = Field(strict=True, ge=1)
    feature: Literal["acceleration", "steady_motion", "deceleration"]
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def meaningful_span(self):
        if self.start >= self.stop or not self.evidence.strip():
            raise ValueError("A stage needs a positive half-open span and evidence")
        return self


class StageAnswer(StrictModel):
    stages: list[StageSpan]
    stage_notes: str


def validate_activity(case, review, rows, expected_hash):
    if partition_sha256(review) != expected_hash:
        raise ValueError("The frozen activity content changed")
    canonical = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
    if review["input_sha256"] != canonical or review["case_id"] != case.case_id:
        raise ValueError("The activity belongs to different observations")
    expected = np.full(len(case.dates), -1, dtype=int)
    state_codes = {"activity": 1, "stationary": 0, "unknown": -1}
    next_start = 0
    for span in review["spans"]:
        if span["start"] != next_start or not span["start"] < span["stop"] <= len(expected):
            raise ValueError("Invalid frozen activity partition")
        expected[span["start"]:span["stop"]] = state_codes[span["state"]]
        next_start = span["stop"]
    if next_start != len(expected) or len(rows) != len(expected):
        raise ValueError("Frozen activity must cover the calendar")
    expected[[value is None for value in case.displacement_mm]] = -1
    for day, row in enumerate(rows):
        if (row["case_id"] != case.case_id or row["day_index"] != day
                or row["date"] != str(case.dates[day]) or row["activity_label"] != expected[day]):
            raise ValueError("Frozen daily activity differs from the partition/calendar")
    return expected


def stage_prompt(case_id, review, activity_hash, images, days):
    active = [[span["start"], span["stop"]] for span in review["spans"] if span["state"] == "activity"]
    return f"""Retrospective GNSS movement-stage review for {case_id}.
The RAW displacement activity review has already been frozen. Do not revise activity.
Activity content SHA-256: {activity_hash}
Allowed nominal activity interiors, HALF-OPEN [start, stop): {json.dumps(active)}
Calendar: integer observation day indices 0..{days - 1}; {days} is an exclusive stop.
Image order: {json.dumps(images)}
First four images: frozen raw full record and three overlapping local views.
Last three images: observation-only centered 31/61/91-day motion estimates.
Overlapping views are context, not separate episodes. Read N/E/U mm scales.

Label stages ONLY within an allowed activity interior. A span cannot bridge distinct
activity interiors, a platform or an uncertain activity margin. Interior observation
gaps may lie inside a span describing surrounding movement, but missing days always
remain unknown in daily export. Unsupported estimates are also masked to unknown.
Do not extend stages because centered derivatives respond before displacement starts.
This is retrospective description, not prediction, failure timing or causal inference.

acceleration: increasing NONZERO motion rate, supported by raw slope/curvature and
multi-window rate evolution; positive speed alone is not acceleration.
deceleration: decreasing motion rate; a negative displacement slope alone is not
deceleration. Motion direction and speed evolution are different quantities.
steady_motion: approximately constant NONZERO rate supported over time. Small or
low-speed movement alone does not prove steadiness. Noise-biased 3D speed around zero
must not create a steady or accelerating stage. Weak activity may remain unclassified.
Use raw shape with the auxiliary evidence; an isolated peak or spike is insufficient.
Leave transitions, weak rate evidence, conflicting windows and unsupported edges
unlabelled. Do not force all active dates into A/S/D. Do not assign low-speed as a
fourth stage class in this task, and do not infer landslide cause.

Return exactly JSON keys stages and stage_notes, no Markdown:
{{"stages":[{{"start":100,"stop":150,"feature":"acceleration",
"evidence":"Describe actual raw and multi-window evidence within frozen activity."}}],
"stage_notes":"Describe unresolved stage evidence and uncertainty."}}
Numbers illustrate the schema, not expected intervals. Empty stages is valid.
Allowed features: acceleration, steady_motion, deceleration.
All starts/stops must be integer HALF-OPEN boundaries, start < stop. Sort by start;
adjacent stages may share a cut point, never a day. Omitted dates remain unknown.
"""


def unclassified_rows(activity_rows):
    return [{**row, "feature": "none" if row["activity_label"] == 0 else "unknown"}
            for row in activity_rows]


def numeric_stage_rows(activity_rows, prediction):
    activity = np.array([row["activity_label"] for row in activity_rows])
    if not np.array_equal(prediction.activity, activity):
        raise ValueError("A stage head changed the supplied activity")
    prediction.validate(len(activity))
    return [{**row, "feature": FEATURE_CODES[int(stage)]}
            for row, stage in zip(activity_rows, prediction.stage, strict=True)]


def apply_stage_support(native_rows, activity_rows, case, motion):
    if len(native_rows) != len(activity_rows) or len(native_rows) != len(case.dates):
        raise ValueError("Stage and observation calendars differ")
    velocity_support = np.array([row is not None and np.isfinite(row).all() for row in motion.velocity_mm_day])
    tangential_support = np.array([value is not None and np.isfinite(value)
                                   for value in motion.tangential_acceleration_mm_day2])
    compiled = []
    suppressed = Counter()
    for day, (native, activity) in enumerate(zip(native_rows, activity_rows, strict=True)):
        if any(native[key] != activity[key] for key in ("case_id", "day_index", "date", "activity_label")):
            raise ValueError("Stage row changed the frozen activity/calendar")
        feature = native["feature"]
        if feature not in {"unknown", "none", *MOTION_FEATURES}:
            raise ValueError("Unsupported stage feature")
        if feature in MOTION_FEATURES and activity["activity_label"] != 1:
            raise ValueError("Definite stage outside frozen activity; no clipping")
        if feature == "none" and activity["activity_label"] != 0:
            raise ValueError("Active/unknown dates cannot default to none")
        if activity["activity_label"] == 0:
            feature = "none"
        elif feature in MOTION_FEATURES:
            if not velocity_support[day]:
                suppressed["no_velocity"] += 1
                feature = "unknown"
            elif feature in {"acceleration", "deceleration"} and not tangential_support[day]:
                suppressed["no_tangential_acceleration"] += 1
                feature = "unknown"
        compiled.append({**activity, "feature": feature})
    return compiled, dict(suppressed)


def compile_stage_answer(payload, case, review, activity_rows, expected_hash, motion):
    validate_activity(case, review, activity_rows, expected_hash)
    answer = StageAnswer.model_validate(payload)
    rows = unclassified_rows(activity_rows)
    active_spans = [span for span in review["spans"] if span["state"] == "activity"]
    previous_stop = 0
    for stage in answer.stages:
        if stage.start < previous_stop:
            raise ValueError("Stage spans overlap or are unsorted")
        if not any(span["start"] <= stage.start < stage.stop <= span["stop"] for span in active_spans):
            raise ValueError("Stage extends outside one frozen activity interior; no clipping")
        for day in range(stage.start, stage.stop):
            if activity_rows[day]["activity_label"] == 1:
                rows[day]["feature"] = stage.feature
        previous_stop = stage.stop
    compiled, suppressed = apply_stage_support(rows, activity_rows, case, motion)
    stage_review = {"schema_version": "landslide-frozen-stage-draft-v1", "case_id": case.case_id,
                    "activity_sha256": expected_hash, **answer.model_dump(mode="json"),
                    "provenance": "AI model development prediction; not independent stage truth"}
    return stage_review, rows, compiled, suppressed


def stage_pair_counts(first_rows, second_rows):
    if len(first_rows) != len(second_rows):
        raise ValueError("Paired calendars differ")
    transitions = Counter()
    for first, second in zip(first_rows, second_rows, strict=True):
        if any(first[key] != second[key] for key in ("case_id", "day_index", "date", "activity_label")):
            raise ValueError("Paired stages do not share the frozen activity")
        if first["activity_label"] == 1:
            transitions[f"{first['feature']}->{second['feature']}"] += 1
    common = sum(count for key, count in transitions.items()
                 if all(feature in MOTION_FEATURES for feature in key.split("->")))
    equal = sum(count for key, count in transitions.items()
                if key.split("->")[0] == key.split("->")[1] and key.split("->")[0] in MOTION_FEATURES)
    return {"active_days": sum(transitions.values()), "common_classified_days": common,
            "common_same_class_days": equal, "common_class_disagreement_days": common - equal,
            "transitions": dict(transitions), "meaning": "Prediction agreement, NOT accuracy"}


def render_stage_heads(case, view, rows_by_method, path, plot):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    observed = np.array([row is not None for row in case.displacement_mm])
    activity = np.array([row["activity_label"] for row in rows_by_method["rule"]])
    activity_colors = activity + 1
    activity_colors[~observed] = 5
    activity_colors[observed & (activity == 1)] = 2
    color_codes = {"unknown": 0, "none": 1, "acceleration": 2, "steady_motion": 3, "deceleration": 4}
    label_colors = [activity_colors.tolist()]
    for method in METHODS:
        labels = np.array([color_codes[row["feature"]] for row in rows_by_method[method]])
        labels[~observed] = 5
        label_colors.append(labels.tolist())
    figure = Figure(figsize=(plot["figure_inches"][0], 10), dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 2]})
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.91, hspace=0.18)
    geometry = view["audit"]
    for axis, name in enumerate(("N", "E", "U")):
        panels[axis].plot(np.arange(len(values)), values[:, axis], linewidth=0.8,
                          color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panels[axis].set_xlim(geometry["panels"][axis]["xlim"])
        panels[axis].set_ylim(geometry["panels"][axis]["ylim"])
        panels[axis].set_ylabel(name + " (mm)")
        panels[axis].grid(alpha=0.18)
    panels[-1].imshow(label_colors, aspect="auto", interpolation="nearest", vmin=0, vmax=5,
                      cmap=ListedColormap(["#e5b65b", "#dce5de", "#d27545", "#557ab6", "#a98630", "#737373"]),
                      extent=(-0.5, len(values) - 0.5, 3.5, -0.5))
    panels[-1].set_yticks(range(4), ["activity", "rule", "XGBoost", "visual"])
    panels[-1].set_xlim(geometry["panels"][0]["xlim"])
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index")
    figure.suptitle(f"{case.case_id} | frozen activity and stage-head drafts | NOT stage truth")
    figure.text(0.10, 0.035,
                "Activity row: orange=active; pale green=stationary; amber=unknown; gray=missing\n"
                "Stage rows: orange=A; blue=S; gold=D; amber=unknown; pale green=none; gray=missing. Same raw axes.",
                fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": label_colors, "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [panel["ylim"] for panel in geometry["panels"]]}
