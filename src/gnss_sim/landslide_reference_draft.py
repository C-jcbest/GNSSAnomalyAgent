"""Raw-first AI reference drafts; these are not independent performance references."""
from __future__ import annotations

import json

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_observation_review import (
    ActivityEpisode,
    ObservationReview,
    ObservedStage,
    StableRange,
    activity_review_sha256,
    compile_observation_review,
)

PROVENANCE = "AI-assisted development draft; no independent expert verification"
FEATURE_COLORS = {
    "low_speed_deformation": "#447d70", "acceleration": "#d27545",
    "steady_motion": "#557ab6", "deceleration": "#a98630",
}


def activity_prompt(case_id: str, last_day: int, images: list[str]) -> str:
    return f"""Read this GNSS record retrospectively, from RAW displacement images only.
Record {case_id}. Image order: {json.dumps(images)}.
The first image shows the full record, the next three are fixed overlapping views of
the SAME observations. Their overlap is context, not separate events. N/E/U axes have
different physical scales: read mm units, not just pixel slopes. Missing observations
remain gaps. All endpoints are inclusive INTEGER original day indices in 0..{last_day}.

First identify sustained continuing change in raw displacement. Activity may be weak,
start/stop mid-record, recur, reverse direction, or be truncated. An elevated plateau
after movement is stationary. A single step, isolated spike, or noisy fluctuation does
not by itself establish continuing displacement. Small changes require a sustained
raw trend relative to the visible correlated variability; one component can be stronger
than others, but mention any disagreement. Do not use derivatives or hidden mechanisms.
Do not infer landslide cause, event counts, failure, or movement not visible in these plots.
Smooth instrument drift and slope motion can be indistinguishable in one record;
this task describes OBSERVABLE sustained displacement, with physical attribution unknown.

For each episode state possible_start <= confirmed_start <= confirmed_end <= possible_end.
Confirmed limits must be conservative interior limits supported by raw observations.
Do not snap every boundary to plot ticks. Possible margins remain UNKNOWN.
Separate episodes' possible ranges must not overlap. Explicitly reviewed stationary
ranges are separate; they must not overlap any episode's entire possible range.
Unreviewed/inconclusive days stay unknown, NOT automatically stationary. It is valid
to return no episodes and to leave ambiguous changes unlabelled. Never fill stages now.
Use concise raw evidence for every episode and stationary range (at most two sentences).

Return JSON with EXACTLY these keys, without Markdown:
{{"episodes":[{{"possible_start":100,"confirmed_start":130,"confirmed_end":220,
"possible_end":250,"raw_evidence":"Continuing N/E change, followed by a plateau."}}],
"stable_ranges":[{{"start":0,"end":70,"raw_evidence":"No continuing raw change."}}],
"activity_notes":"Mention ambiguous ranges and physical-attribution limits."}}
The numbers above illustrate the schema, NOT expected answers. Empty lists are valid.
Sort each interval list by its start; no overlapping closed endpoints.
"""


def stage_prompt(review: ObservationReview, last_day: int, images: list[str]) -> str:
    confirmed = [[episode.confirmed_start, episode.confirmed_end] for episode in review.episodes]
    return f"""Retrospective second-layer stage review for {review.case_id}.
The RAW displacement activity review is already frozen. DO NOT change its boundaries.
Frozen activity SHA-256: {activity_review_sha256(review)}
Confirmed inclusive activity intervals: {json.dumps(confirmed)}
Image order: {json.dumps(images)}. First four images are the same raw full/fixed-local
views; last three contain observation-only centered fits with 31, 61 and 91 day windows.
All dates retain original day index 0..{last_day}; missing observations stay gaps.

Only WITHIN a confirmed interval infer observable motion stages. Use raw curve shape
and cross-window velocity/acceleration agreement. Centered estimates can respond before
motion begins; this is not online prediction. Auxiliary peaks, isolated jumps, high speed,
or 3D speed's positive bias around zero do NOT establish acceleration.
Features: acceleration=increasing nonzero motion rate; deceleration=decreasing rate;
steady_motion=approximately constant NONZERO rate; low_speed_deformation=visible small
continuing cumulative displacement where a more definite A/S/D classification is not
supported. Low speed alone is NOT proof of steadiness. Physical cause remains unknown.
Leave weak evidence, unsupported edges, transitions, gaps, direction reversal and
inconsistent windows unlabelled as needed. Do not force every active day into a class.
No definite feature may extend into a platform, uncertain margin, or outside activity.
Use inclusive integer endpoints. Adjacent stages cannot share a day; sort by start.

Return JSON with EXACTLY these keys, without Markdown:
{{"stages":[{{"start":130,"end":160,"feature":"acceleration",
"evidence":"Raw curvature and multi-window rate rise within frozen activity."}}],
"stage_notes":"Mention transition gaps and unresolved stage evidence."}}
These numbers only illustrate the schema. Empty stages are valid. Allowed feature
strings are low_speed_deformation, acceleration, steady_motion, deceleration.
"""


def parse_activity_draft(payload: dict, case, record: dict, diagnostics, model: str):
    if not isinstance(payload, dict) or set(payload) != {"episodes", "stable_ranges", "activity_notes"}:
        raise ValueError("Activity draft requires exactly episodes, stable_ranges, activity_notes")
    if not isinstance(payload["episodes"], list) or not isinstance(payload["stable_ranges"], list):
        raise ValueError("Activity interval fields must be JSON lists")
    review = ObservationReview(
        case_id=case.case_id, input_sha256=record["input_sha256"],
        raw_image_sha256=record["views"][0]["image_sha256"],
        reviewer=f"{model} / single-pass raw-image review", provenance=PROVENANCE,
        activity_status="reviewed", episodes=[ActivityEpisode.model_validate(row) for row in payload["episodes"]],
        stable_ranges=[StableRange.model_validate(row) for row in payload["stable_ranges"]],
        activity_notes=payload["activity_notes"],
    )
    for bounds in ([(row.possible_start, row.possible_end) for row in review.episodes],
                   [(row.start, row.end) for row in review.stable_ranges]):
        if bounds != sorted(bounds):
            raise ValueError("Draft intervals must be sorted")
    rows = compile_observation_review(case, review, diagnostics, review.raw_image_sha256)
    return review, rows


def parse_stage_draft(payload: dict, review, case, diagnostics):
    if not isinstance(payload, dict) or set(payload) != {"stages", "stage_notes"}:
        raise ValueError("Stage draft requires exactly stages and stage_notes")
    if not isinstance(payload["stage_notes"], str):
        raise ValueError("Stage notes must be text")
    if not isinstance(payload["stages"], list):
        raise ValueError("Stage intervals must be a JSON list")
    staged = review.model_copy(deep=True)
    staged.stage_activity_sha256 = activity_review_sha256(review)
    staged.stages = [ObservedStage.model_validate(row) for row in payload["stages"]]
    bounds = [(row.start, row.end) for row in staged.stages]
    if bounds != sorted(bounds):
        raise ValueError("Stages must be sorted")
    rows = compile_observation_review(case, staged, diagnostics, review.raw_image_sha256)
    return staged, rows


def unknown_rows(case):
    return [{"case_id": case.case_id, "day_index": day, "date": str(date),
             "activity_label": -1, "feature": "unknown"} for day, date in enumerate(case.dates)]


def render_draft_overlay(case, record: dict, rows: list[dict], path, plot: dict):
    """Draw exactly the compiled daily masks; this chart is NEVER an inference input."""
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    activity = np.asarray([row["activity_label"] for row in rows])
    features = np.asarray([row["feature"] for row in rows])
    figure = Figure(figsize=plot["figure_inches"], dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(3, 1, sharex=True)
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.89, hspace=0.17)
    geometry = record["views"][0]["audit"]
    mask_spans = {"activity": [[start, stop - 1] for start, stop in runs(activity == 1)]}
    for feature in FEATURE_COLORS:
        mask_spans[feature] = [[start, stop - 1] for start, stop in runs(features == feature)]
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        for start, end in mask_spans["activity"]:
            panel.axvspan(start - 0.5, end + 0.5, color="#c2d8df", alpha=0.45, linewidth=0)
        for feature, color in FEATURE_COLORS.items():
            for start, end in mask_spans[feature]:
                panel.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.23, linewidth=0)
        panel.plot(np.arange(len(case.dates)), values[:, axis], linewidth=0.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panel.set_xlim(geometry["panels"][axis]["xlim"])
        panel.set_ylim(geometry["panels"][axis]["ylim"])
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=0.18)
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index; missing/unsupported stages remain unlabelled")
    figure.suptitle(f"{case.case_id} | AI draft overlay; NOT expert truth; NOT sent to model")
    figure.text(0.10, 0.02, "Blue background: raw-confirmed activity | "
                "L: green; A: orange; S: blue; D: gold\n"
                "No colored stage outside compiled activity; white can mean stable, missing or unknown.",
                fontsize=9)
    figure.savefig(path, format="png")
    return mask_spans
