"""Compile observed references only after raw-displacement activity review is frozen."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.schemas import StrictModel


class ActivityEpisode(StrictModel):
    possible_start: int = Field(strict=True, ge=0)
    confirmed_start: int = Field(strict=True, ge=0)
    confirmed_end: int = Field(strict=True, ge=0)
    possible_end: int = Field(strict=True, ge=0)
    raw_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_endpoints(self):
        if not self.possible_start <= self.confirmed_start <= self.confirmed_end <= self.possible_end:
            raise ValueError("Activity endpoints are not ordered")
        return self


class StableRange(StrictModel):
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, ge=0)
    raw_evidence: str = Field(min_length=1)


class ObservedStage(StrictModel):
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, ge=0)
    feature: Literal["low_speed_deformation", "acceleration", "steady_motion", "deceleration"]
    evidence: str = Field(min_length=1)


class ObservationReview(StrictModel):
    schema_version: Literal["landslide-observation-review-v1"] = "landslide-observation-review-v1"
    case_id: str
    input_sha256: str
    raw_image_sha256: str
    reviewer: str = ""
    provenance: str = "pending; no observation reference exists"
    activity_status: Literal["pending", "reviewed"] = "pending"
    episodes: list[ActivityEpisode] = Field(default_factory=list)
    stable_ranges: list[StableRange] = Field(default_factory=list)
    activity_notes: str = ""
    stage_activity_sha256: str | None = None
    stages: list[ObservedStage] = Field(default_factory=list)


def activity_review_sha256(review: ObservationReview) -> str:
    payload = review.model_dump(exclude={"stage_activity_sha256", "stages"})
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_observation_review(case: LandslideInput, review: ObservationReview,
                               diagnostics: LandslideDiagnostics, raw_image_sha256: str) -> list[dict]:
    """Unreviewed days and uncertain boundaries remain unknown, never implicit negatives."""
    input_hash = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
    if review.case_id != case.case_id or review.input_sha256 != input_hash:
        raise ValueError("Review does not identify these observations")
    if review.raw_image_sha256 != raw_image_sha256:
        raise ValueError("Raw review image changed")
    if diagnostics.case_id != case.case_id or diagnostics.input_sha256 != input_hash:
        raise ValueError("Auxiliary estimates do not identify these observations")
    if diagnostics.window_days != 61:
        raise ValueError("Stage support uses the fixed 61-day diagnostics")
    if review.activity_status != "reviewed" or not review.reviewer.strip():
        raise ValueError("Pending reviews cannot produce scored labels")
    if not review.provenance.strip() or review.provenance.startswith("pending"):
        raise ValueError("Describe the actual reference provenance before exporting labels")
    if review.stages and review.stage_activity_sha256 != activity_review_sha256(review):
        raise ValueError("Freeze the raw activity review before assigning stages")
    days = len(case.dates)
    activity = np.full(days, -1, dtype=int)
    features = np.full(days, "unknown", dtype=object)
    occupied = np.zeros(days, dtype=bool)

    def reserve(start: int, end: int):
        if not 0 <= start <= end < days:
            raise ValueError("Review interval is outside the observation calendar")
        if occupied[start:end + 1].any():
            raise ValueError("Activity, stable or uncertainty intervals overlap")
        occupied[start:end + 1] = True

    for episode in review.episodes:
        reserve(episode.possible_start, episode.possible_end)
        activity[episode.confirmed_start:episode.confirmed_end + 1] = 1
    for stable in review.stable_ranges:
        reserve(stable.start, stable.end)
        activity[stable.start:stable.end + 1] = 0
        features[stable.start:stable.end + 1] = "none"
    assigned_stage = np.zeros(days, dtype=bool)
    supported = np.array([row is not None for row in diagnostics.velocity_mm_day])
    for stage in review.stages:
        if not 0 <= stage.start <= stage.end < days:
            raise ValueError("Stage interval is outside the observation calendar")
        if not (activity[stage.start:stage.end + 1] == 1).all():
            raise ValueError("A stage must be inside confirmed raw-displacement activity")
        if assigned_stage[stage.start:stage.end + 1].any():
            raise ValueError("Stage intervals overlap")
        assigned_stage[stage.start:stage.end + 1] = True
        features[stage.start:stage.end + 1] = stage.feature
    features[(activity == 1) & ~supported] = "unknown"
    tangential_supported = np.array([
        value is not None for value in diagnostics.tangential_acceleration_mm_day2
    ])
    features[np.isin(features, ["acceleration", "deceleration"]) & ~tangential_supported] = "unknown"
    observed = np.array([row is not None for row in case.displacement_mm])
    activity[~observed] = -1
    features[~observed] = "unknown"
    return [{"case_id": case.case_id, "day_index": day, "date": str(case.dates[day]),
             "activity_label": int(activity[day]), "feature": str(features[day])}
            for day in range(days)]
