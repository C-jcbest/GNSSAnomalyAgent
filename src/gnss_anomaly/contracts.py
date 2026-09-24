"""Only Window reaches detectors/models; truth and experiment strata live elsewhere."""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

Channel = Literal["N", "E", "U"]
DetectorName = Literal["hampel", "cusum", "iforest"]
CHANNELS = ("N", "E", "U")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Window(StrictModel):
    case_id: str
    sampling_hours: Literal[1, 24] = 1
    timestamps: list[datetime]
    values: list[tuple[float | None, float | None, float | None]]

    @model_validator(mode="after")
    def validate_grid(self):
        if len(self.timestamps) != len(self.values) or not self.values:
            raise ValueError("timestamps and values must have the same nonzero length")
        if any(t.utcoffset() is None for t in self.timestamps):
            raise ValueError("timestamps must include timezone")
        self.timestamps = [t.astimezone(timezone.utc) for t in self.timestamps]
        if any(t.minute or t.second or t.microsecond for t in self.timestamps):
            raise ValueError("timestamps must align to whole UTC hours")
        if any(
            (b - a).total_seconds() != self.sampling_hours * 3600
            for a, b in zip(self.timestamps, self.timestamps[1:])
        ):
            raise ValueError(
                "expected regular sampling grid; use null values for missing positions"
            )
        return self

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)


class Event(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    channels: list[Channel] = Field(min_length=1, max_length=3)
    kind: str = Field(default="unspecified", max_length=80)
    reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end < self.start or len(set(self.channels)) != len(self.channels):
            raise ValueError("invalid inclusive interval or duplicate channels")
        return self


class Prediction(StrictModel):
    status: Literal["ok", "insufficient", "error"] = "ok"
    events: list[Event] = Field(default_factory=list, max_length=200)
    reason: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_status(self):
        if self.status != "ok" and self.events:
            raise ValueError("non-ok prediction cannot contain events")
        return self

    def mask(self, length: int) -> np.ndarray:
        result = np.zeros((length, 3), dtype=bool)
        for event in self.events:
            if event.end >= length:
                raise ValueError("prediction lies outside input window")
            for channel in event.channels:
                result[event.start : event.end + 1, CHANNELS.index(channel)] = True
        return result


@lru_cache(maxsize=64)
def prediction_schema(length: int):
    if length < 1:
        raise ValueError("prediction length must be positive")
    bounded_event = create_model(
        f"Event{length}",
        __base__=Event,
        start=(int, Field(ge=0, le=length - 1, strict=True)),
        end=(int, Field(ge=0, le=length - 1, strict=True)),
    )
    return create_model(
        f"Prediction{length}",
        __base__=Prediction,
        events=(list[bounded_event], Field(default_factory=list, max_length=200)),
    )


def mask_events(mask: np.ndarray, kind: str = "candidate") -> list[Event]:
    events = []
    for c, channel in enumerate(CHANNELS):
        padded = np.r_[False, mask[:, c], False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        events.extend(
            Event(start=int(a), end=int(b), channels=[channel], kind=kind)
            for a, b in zip(starts, ends)
        )
    return sorted(events, key=lambda e: (e.start, e.channels))
