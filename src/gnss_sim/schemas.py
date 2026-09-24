from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AxisVector = tuple[float, float, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(StrictModel):
    seed: int = Field(ge=0, le=4294967295)
    count: int = Field(ge=1, le=75)


class CaseInput(StrictModel):
    schema_version: Literal["normal-input-v1"] = "normal-input-v1"
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


class CaseTruth(StrictModel):
    schema_version: Literal["normal-truth-v1"] = "normal-truth-v1"
    case_id: str
    normal_background_mm: list[AxisVector]
    measurement_noise_mm: list[AxisVector]
    annual_phase_rad: AxisVector
    semiannual_phase_rad: AxisVector
    component_seeds: ComponentSeeds
    events: list[dict] = Field(default_factory=list)


class CaseSummary(StrictModel):
    case_id: str
    case_seed: int
    event_count: Literal[0] = 0


class DatasetManifest(StrictModel):
    schema_version: Literal["normal-dataset-v1"] = "normal-dataset-v1"
    generator_version: Literal["normal-v1"] = "normal-v1"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: GenerationRequest
    generated_cases: int = 0
    cases: list[CaseSummary] = Field(default_factory=list)
    error: str | None = None
