from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AxisVector = tuple[float, float, float]
Axis = Literal["N", "E", "U"]
CaseType = Literal[
    "normal", "spike", "step", "slow_trend", "acceleration", "transient_shift"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(StrictModel):
    seed: int = Field(ge=0, le=4294967295)
    count: int = Field(ge=1, le=5000)
    case_type: CaseType


class CaseInput(StrictModel):
    schema_version: Literal["event-input-v2"] = "event-input-v2"
    case_id: str
    dates: list[date]
    reference_coordinate_mm: AxisVector
    observed_coordinate_mm: list[AxisVector]
    displacement_mm: list[AxisVector]
    horizontal_offset_mm: list[float]
    spatial_offset_mm: list[float]


class ComponentSeeds(StrictModel):
    annual_phase: int
    semiannual_phase: int
    white_noise: int


class EventSeeds(StrictModel):
    position: int
    shape: int
    sign: int


class EventBase(StrictModel):
    event_id: Literal["event_001"]
    axis: Axis
    start_index: int = Field(ge=60, le=304)
    end_index: int = Field(ge=60, le=304)
    start_date: date
    end_date: date
    persistent: bool

    @model_validator(mode="after")
    def validate_interval(self):
        first = date(2025, 1, 1)
        if self.end_index < self.start_index:
            raise ValueError("end_index precedes start_index")
        if self.start_date != first + timedelta(days=self.start_index):
            raise ValueError("start_date does not match start_index")
        if self.end_date != first + timedelta(days=self.end_index):
            raise ValueError("end_date does not match end_index")
        return self


class SpikeParameters(StrictModel):
    duration_days: Literal[1]
    amplitude_mm: float
    sigma_multiplier: Literal[6]


class StepParameters(StrictModel):
    duration_days: Literal[1]
    amplitude_mm: float


class TrendParameters(StrictModel):
    duration_days: Literal[90]
    final_offset_mm: float
    slope_mm_per_day: float


class AccelerationParameters(StrictModel):
    duration_days: Literal[90]
    final_offset_mm: float


class TransientParameters(StrictModel):
    duration_days: Literal[14]
    amplitude_mm: float


class SpikeEvent(EventBase):
    type: Literal["spike"]
    source: Literal["observation_artifact"]
    persistent: Literal[False]
    parameters: SpikeParameters


class StepEvent(EventBase):
    type: Literal["step"]
    source: Literal["injected_deformation"]
    persistent: Literal[True]
    parameters: StepParameters


class SlowTrendEvent(EventBase):
    type: Literal["slow_trend"]
    source: Literal["injected_deformation"]
    persistent: Literal[True]
    parameters: TrendParameters


class AccelerationEvent(EventBase):
    type: Literal["acceleration"]
    source: Literal["injected_deformation"]
    persistent: Literal[True]
    parameters: AccelerationParameters


class TransientShiftEvent(EventBase):
    type: Literal["transient_shift"]
    source: Literal["observation_artifact"]
    persistent: Literal[False]
    parameters: TransientParameters


Event = Annotated[
    SpikeEvent | StepEvent | SlowTrendEvent | AccelerationEvent | TransientShiftEvent,
    Field(discriminator="type"),
]


class CaseTruth(StrictModel):
    schema_version: Literal["event-truth-v2"] = "event-truth-v2"
    case_id: str
    normal_background_mm: list[AxisVector]
    measurement_noise_mm: list[AxisVector]
    injected_deformation_mm: list[AxisVector]
    observation_artifact_mm: list[AxisVector]
    annual_phase_rad: AxisVector
    semiannual_phase_rad: AxisVector
    component_seeds: ComponentSeeds
    event_seeds: EventSeeds | None = None
    events: list[Event] = Field(default_factory=list, max_length=1)


class CaseSummary(StrictModel):
    case_id: str
    case_seed: int
    event_count: int = Field(ge=0, le=1)


class DatasetManifest(StrictModel):
    schema_version: Literal["event-dataset-v2"] = "event-dataset-v2"
    generator_version: Literal["event-v2"] = "event-v2"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: GenerationRequest
    generated_cases: int = 0
    cases: list[CaseSummary] = Field(default_factory=list)
    error: str | None = None
