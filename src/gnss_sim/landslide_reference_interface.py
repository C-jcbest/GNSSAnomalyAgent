"""A prospective, exclusive activity interface; no repairs of frozen v1 answers."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from gnss_sim.landslide_reference_draft import activity_prompt
from gnss_sim.schemas import StrictModel


class ActivityPartitionSpan(StrictModel):
    start: int = Field(strict=True, ge=0)
    stop: int = Field(strict=True, ge=1)
    state: Literal["activity", "stationary", "unknown"]
    raw_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def positive_length(self):
        if self.start >= self.stop:
            raise ValueError("A half-open activity span must have start < stop")
        if not self.raw_evidence.strip():
            raise ValueError("Describe raw evidence or the reason for uncertainty")
        return self


class ActivityPartitionAnswer(StrictModel):
    spans: list[ActivityPartitionSpan]
    activity_notes: str


def partition_prompt(case_id: str, last_day: int, images: list[str]) -> str:
    original = activity_prompt(case_id, last_day, images)
    # The observational task is identical; only the time/output contract changes.
    task = original.split("For each episode state", 1)[0]
    task = task.replace(
        f"All endpoints are inclusive INTEGER original day indices in 0..{last_day}.",
        f"Observation day indices are INTEGER original indices in 0..{last_day}.",
    )
    return task + f"""Use ONE chronological partition of the calendar, rather than separate episode and
stationary lists. Every span is HALF-OPEN [start, stop): start is included, stop is
EXCLUDED. Adjacent spans share a cut point, never an observation day. The first span
starts at 0, every next start equals the previous stop, and the final stop equals
{last_day + 1}. Each span must have start < stop. Do not snap every cut to plot ticks.

States: activity=conservative interior supported by continuing RAW displacement;
stationary=explicitly reviewed raw evidence of no continuing displacement;
unknown=possible activity margins, uncertain transitions, inconclusive or unreviewed
dates. UNKNOWN is valid anywhere, including the entire calendar. Full calendar
coverage is an accounting requirement, NOT a requirement to decide uncertain days.
Missing observations always remain unknown in the exported daily mask; do not infer
motion through gaps. Do not use auxiliary images or assign any stages now.
Use concise raw evidence or a reason for uncertainty for each span (at most two
sentences). Mention component disagreement and physical-attribution limits.

Return JSON with EXACTLY these keys, without Markdown:
{{"spans":[{{"start":0,"stop":100,"state":"stationary",
"raw_evidence":"No continuing raw change."}},
{{"start":100,"stop":{last_day + 1},"state":"unknown",
"raw_evidence":"Remaining dates have not been resolved."}}],
"activity_notes":"Mention ambiguous ranges and physical-attribution limits."}}
Numbers illustrate the schema, NOT expected answers. Allowed state strings are
activity, stationary, unknown. Use only this partition schema.
"""


def parse_partition_draft(payload, case, record, model: str):
    answer = ActivityPartitionAnswer.model_validate(payload)
    expected_input = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
    if expected_input != record["input_sha256"]:
        raise ValueError("Partition does not identify these observations")
    days = len(case.dates)
    activity = np.full(days, -1, dtype=int)
    next_start = 0
    states = {"activity": 1, "stationary": 0, "unknown": -1}
    for span in answer.spans:
        if span.start != next_start:
            raise ValueError("Partition has a gap, overlap or unsorted spans")
        if span.stop > days:
            raise ValueError("Partition extends beyond the observation calendar")
        activity[span.start:span.stop] = states[span.state]
        next_start = span.stop
    if next_start != days:
        raise ValueError("Partition must explicitly cover the full calendar")
    observed = np.array([row is not None for row in case.displacement_mm])
    activity[~observed] = -1
    review = {
        "schema_version": "landslide-activity-partition-v2",
        "case_id": case.case_id,
        "input_sha256": expected_input,
        "raw_image_sha256": [view["image_sha256"] for view in record["views"]],
        "reviewer": f"{model} / single-pass raw-image review",
        "provenance": "AI-assisted development draft; no independent expert verification",
        **answer.model_dump(mode="json"),
    }
    rows = [{"case_id": case.case_id, "day_index": day, "date": str(date),
             "activity_label": int(activity[day]),
             "feature": "none" if activity[day] == 0 else "unknown"}
            for day, date in enumerate(case.dates)]
    return review, rows


def partition_sha256(review: dict) -> str:
    encoded = json.dumps(review, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def paired_activity_counts(first_rows, second_rows, observed) -> dict:
    if len(first_rows) != len(second_rows) or len(first_rows) != len(observed):
        raise ValueError("Paired daily calendars differ")
    first = np.array([row["activity_label"] for row in first_rows])
    second = np.array([row["activity_label"] for row in second_rows])
    observed = np.asarray(observed, dtype=bool)
    if not np.isin(first, [-1, 0, 1]).all() or not np.isin(second, [-1, 0, 1]).all():
        raise ValueError("Unexpected activity state")
    if (first[~observed] != -1).any() or (second[~observed] != -1).any():
        raise ValueError("Missing days must remain unknown in both arms")
    transitions = {
        f"{old}->{new}": int(np.count_nonzero(observed & (first == old) & (second == new)))
        for old in (-1, 0, 1) for new in (-1, 0, 1)
    }
    return {
        "observed_days": int(observed.sum()),
        "state_disagreement_days": int(np.count_nonzero(observed & (first != second))),
        "transitions_episode_to_partition": transitions,
        "meaning": "Paired draft disagreement, NOT accuracy or error reduction",
    }


def render_activity_pair(case, record, rows_by_arm, path, plot):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    figure = Figure(figsize=(plot["figure_inches"][0], 9), dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 1.5]})
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.91, hspace=0.18)
    geometry = record["views"][0]["audit"]
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        panel.plot(np.arange(len(case.dates)), values[:, axis], linewidth=0.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panel.set_xlim(geometry["panels"][axis]["xlim"])
        panel.set_ylim(geometry["panels"][axis]["ylim"])
        panel.set_ylabel(name + " (mm)")
        panel.grid(alpha=0.18)
    label_image = []
    observed = np.array([row is not None for row in case.displacement_mm])
    for arm in ("episode_v1", "partition_v2"):
        labels = np.array([row["activity_label"] for row in rows_by_arm[arm]])
        colors = labels + 1
        colors[~observed] = 3
        label_image.append(colors.tolist())
    panels[-1].imshow(label_image, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                      cmap=ListedColormap(["#e5b65b", "#dce5de", "#427b9b", "#737373"]),
                      extent=(-0.5, len(case.dates) - 0.5, 1.5, -0.5))
    panels[-1].set_yticks([0, 1], ["episode v1", "partition v2"])
    panels[-1].set_xlim(geometry["panels"][0]["xlim"])
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index")
    figure.suptitle(case.case_id + " | prospective interface comparison; AI drafts, NOT truth")
    figure.text(0.12, 0.035, "Blue: activity | pale green: stationary | amber: observed unknown | gray: missing\n"
                "Same frozen raw axes; no stages assigned; this comparison image was NOT sent to the model.", fontsize=9)
    figure.savefig(path, format="png")
    return {"label_image": label_image, "not_sent_to_model": True}
