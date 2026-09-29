"""Current synthetic observations, ground truth and detection outputs."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CaseType = Literal["normal", "global_extremum", "trend", "mean_shift"]
CASE_TYPES: tuple[CaseType, ...] = ("normal", "global_extremum", "trend", "mean_shift")
Axis = Literal["N", "E", "U"]
Vector = tuple[float, float, float]
Series = Annotated[list[Vector], Field(min_length=365, max_length=365)]
Labels = Annotated[list[Literal[0, 1]], Field(min_length=365, max_length=365)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GenerationRequest(StrictModel):
    seed: int = Field(strict=True, ge=0, le=4294967295)
    count: int = Field(strict=True, ge=1, le=5000)
    case_type: CaseType | Literal["all"]

    @model_validator(mode="after")
    def validate_count(self):
        if self.case_type == "all" and self.count % 4:
            raise ValueError("配对批次数量必须是 4 的倍数；推荐 24 例覆盖三轴正负方向")
        return self


class CaseInput(StrictModel):
    schema_version: Literal["gnss-input-v1"] = "gnss-input-v1"
    case_id: str = Field(pattern=r"^case_\d{4}$")
    dates: list[date] = Field(min_length=365, max_length=365)
    reference_coordinate_mm: Vector
    observed_coordinate_mm: Series
    displacement_mm: Series
    horizontal_offset_mm: list[float] = Field(min_length=365, max_length=365)
    spatial_offset_mm: list[float] = Field(min_length=365, max_length=365)


class Event(StrictModel):
    event_id: Literal["event_001"] = "event_001"
    type: Literal["global_extremum", "trend", "mean_shift"]
    task: Literal["point", "range"]
    axis: Axis
    start_index: int = Field(ge=60, le=304)
    end_index: int = Field(ge=60, le=304)
    start_date: date
    end_date: date
    persistent: bool
    operation: Literal["native_extremum", "add"]
    target_offset_mm: float  # Actual latent displacement at the active interval end.
    observed_end_mm: float
    sigma_mm: float = Field(gt=0)
    source: str
    native_parameters: dict[str, float | int | bool | str]

    @model_validator(mode="after")
    def validate_profile(self):
        duration = {"global_extremum": 1, "trend": 90, "mean_shift": 14}[self.type]
        if self.end_index - self.start_index + 1 != duration:
            raise ValueError("native interval does not match profile")
        if self.persistent != (self.type == "trend"):
            raise ValueError("invalid residual policy")
        point = self.type == "global_extremum"
        if self.task != ("point" if point else "range"):
            raise ValueError("invalid task")
        if self.operation != ("native_extremum" if point else "add"):
            raise ValueError("invalid operation")
        for index, day in ((self.start_index, self.start_date), (self.end_index, self.end_date)):
            if day != date(2025, 1, 1) + timedelta(days=index):
                raise ValueError("event date/index mismatch")
        return self


class CaseTruth(StrictModel):
    schema_version: Literal["gnss-truth-v1"] = "gnss-truth-v1"
    case_id: str
    background_group: str
    measurement_noise_mm: Series
    anomaly_delta_mm: Series
    noise_seed: int
    position_seed: int
    tods_seed: int
    native_labels: Labels
    axis_labels: list[tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]]] = Field(
        min_length=365, max_length=365)
    source_lock_sha256: str
    events: list[Event] = Field(max_length=1)


class CaseSummary(StrictModel):
    case_id: str = Field(pattern=r"^case_\d{4}$")
    case_seed: int
    case_type: CaseType
    event_count: int = Field(ge=0, le=1)
    background_group: str
    group_axis: Axis
    group_sign: Literal[-1, 1]


class DatasetManifest(StrictModel):
    schema_version: Literal["gnss-dataset-v1"] = "gnss-dataset-v1"
    generator_version: Literal["synthetic-v1"] = "synthetic-v1"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: GenerationRequest
    type_counts: dict[CaseType, int]
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
