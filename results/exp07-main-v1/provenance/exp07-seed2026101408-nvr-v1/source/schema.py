"""Observed morphology, uncertainty and displacement-first phase constraints."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVENT_TYPES = ("spike", "step", "transient", "deformation")


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("interval end precedes start")
        return self


class Phase(Interval):
    kind: Literal["acceleration", "steady", "deceleration"]


class Event(Interval):
    kind: Literal["spike", "step", "transient", "deformation"]
    status: Literal["confirmed", "uncertain"]
    evidence: str = Field(min_length=1)
    phases: list[Phase]

    @model_validator(mode="after")
    def displacement_before_phases(self):
        if self.kind != "deformation" or self.status != "confirmed":
            if self.phases:
                raise ValueError("only confirmed displacement activity may contain phases")
        previous_end = self.start - 1
        for phase in self.phases:
            if not self.start <= phase.start <= phase.end <= self.end:
                raise ValueError("phase lies outside confirmed displacement activity")
            if phase.start <= previous_end:
                raise ValueError("phases must be ordered and nonoverlapping")
            previous_end = phase.end
        if self.kind in ("spike", "step") and self.start != self.end:
            raise ValueError("spikes and step onsets are point events")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    events: list[Event]

    def validate_length(self, length: int) -> Prediction:
        if any(event.end >= length for event in self.events):
            raise ValueError("prediction exceeds observation indices")
        return self


class VisualEvent(Interval):
    kind: Literal["spike", "step", "transient", "deformation"]
    status: Literal["confirmed", "uncertain"]
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def point_event(self):
        if self.kind in ("spike", "step") and self.start != self.end:
            raise ValueError("spikes and step onsets are point events")
        return self


class VisualDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    events: list[VisualEvent]

    def as_prediction(self, length: int) -> Prediction:
        return Prediction.model_validate({"events": [
            {**event.model_dump(), "phases": []} for event in self.events
        ]}).validate_length(length)


def observed_values(observation: dict) -> np.ndarray:
    values = np.array([
        row if row is not None else [np.nan, np.nan, np.nan]
        for row in observation["displacement_mm"]
    ], dtype=float)
    if values.shape != (len(observation["dates"]), 3):
        raise ValueError("daily N/E/U observations required")
    if len(values) < 30 or not np.isfinite(values).any():
        raise ValueError("insufficient observations")
    return values


def event_masks(prediction: Prediction, observed: np.ndarray) -> dict[str, np.ndarray]:
    masks = {kind: np.zeros(len(observed), dtype=bool) for kind in EVENT_TYPES}
    for event in prediction.events:
        if event.status == "confirmed":
            masks[event.kind][event.start:event.end + 1] = True
    for mask in masks.values():
        mask &= observed
    return masks


def phase_labels(prediction: Prediction, observed: np.ndarray) -> list[str | None]:
    labels: list[str | None] = [None] * len(observed)
    for event in prediction.events:
        for phase in event.phases:
            for day in range(phase.start, phase.end + 1):
                if observed[day]:
                    labels[day] = phase.kind
    return labels
