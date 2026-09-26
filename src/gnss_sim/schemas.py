from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AxisVector = tuple[float, float, float]
Axis = Literal["N", "E", "U"]
CaseType = Literal[
    "normal", "spike", "step", "slow_trend", "acceleration", "transient_shift"
]
ScenarioType = Literal[
    "multi_spike", "change_with_local", "temporary_with_local",
    "longterm_with_local", "longterm_with_change", "complex_multiaxis",
]
SCENARIO_TYPES: tuple[ScenarioType, ...] = (
    "multi_spike", "change_with_local", "temporary_with_local",
    "longterm_with_local", "longterm_with_change", "complex_multiaxis",
)
GenerationType = CaseType | ScenarioType
CASE_TYPES: tuple[CaseType, ...] = (
    "normal", "spike", "step", "slow_trend", "acceleration", "transient_shift"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(StrictModel):
    seed: int = Field(ge=0, le=4294967295)
    count: int = Field(ge=1, le=5000)
    case_type: GenerationType | Literal["all", "all_scenarios"]

    @model_validator(mode="after")
    def validate_mixed_count(self):
        if self.case_type in ("all", "all_scenarios") and self.count < 6:
            raise ValueError("mixed generation requires at least six cases")
        return self


class CaseInput(StrictModel):
    schema_version: Literal["event-input-v6"] = "event-input-v6"
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
    event_id: str = Field(pattern=r"^event_\d{3,}$")
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


class EventContribution(StrictModel):
    event_id: str = Field(pattern=r"^event_\d{3,}$")
    component: Literal["injected_deformation", "observation_artifact"]
    values_mm: list[AxisVector]


class CaseTruth(StrictModel):
    schema_version: Literal["event-truth-v6"] = "event-truth-v6"
    case_id: str
    scenario_type: ScenarioType | None = None
    normal_background_mm: list[AxisVector]
    measurement_noise_mm: list[AxisVector]
    injected_deformation_mm: list[AxisVector]
    observation_artifact_mm: list[AxisVector]
    annual_phase_rad: AxisVector
    semiannual_phase_rad: AxisVector
    component_seeds: ComponentSeeds
    event_seeds: EventSeeds | None = None
    events: list[Event] = Field(default_factory=list)
    event_contributions: list[EventContribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_event_links(self):
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")
        if event_ids != [part.event_id for part in self.event_contributions]:
            raise ValueError("each event requires one ordered contribution")
        for event, part in zip(self.events, self.event_contributions):
            if event.source != part.component:
                raise ValueError("event contribution source mismatch")
        return self


class CaseSummary(StrictModel):
    case_id: str
    case_seed: int
    case_type: GenerationType
    event_count: int = Field(ge=0)


class DatasetManifest(StrictModel):
    schema_version: Literal["event-dataset-v6"] = "event-dataset-v6"
    generator_version: Literal["event-v6"] = "event-v6"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: GenerationRequest
    type_counts: dict[GenerationType, int]
    generated_cases: int = 0
    cases: list[CaseSummary] = Field(default_factory=list)
    error: str | None = None


DayIndex = Annotated[int, Field(strict=True, ge=0, le=364)]
DayRange = tuple[DayIndex, DayIndex]


class PointPredictions(StrictModel):
    N: list[DayIndex]
    E: list[DayIndex]
    U: list[DayIndex]


class RangePredictions(StrictModel):
    N: list[DayRange]
    E: list[DayRange]
    U: list[DayRange]

    @model_validator(mode="after")
    def validate_ranges(self):
        for ranges in (self.N, self.E, self.U):
            if any(start > end for start, end in ranges):
                raise ValueError("range start must not exceed end")
        return self


class PointResult(StrictModel):
    case_id: str
    method: str
    status: Literal["success", "failed"]
    predictions: PointPredictions


class RangeResult(StrictModel):
    case_id: str
    method: str
    status: Literal["success", "failed"]
    predictions: RangePredictions
