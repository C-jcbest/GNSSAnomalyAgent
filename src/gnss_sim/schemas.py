from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AxisVector = tuple[float, float, float]


class SimulationConfig(BaseModel):
    start_date: date = date(2025, 1, 1)
    days: Literal[180] = 180
    reference_coordinate_mm: AxisVector = (0.0, 0.0, 0.0)
    white_sigma_mm: AxisVector = (1.5, 1.5, 3.0)
    ar_innovation_sigma_mm: AxisVector = (0.75, 0.75, 1.5)
    ar_rho: float = 0.7
    seasonal_amplitude_mm: AxisVector = (2.0, 2.0, 4.0)
    seasonal_phase_rad: AxisVector = (0.0, 2.0943951023931953, 4.1887902047863905)
    seasonal_period_days: float = 365.0

    @model_validator(mode="after")
    def validate_parameters(self) -> SimulationConfig:
        vectors = (
            self.reference_coordinate_mm,
            self.white_sigma_mm,
            self.ar_innovation_sigma_mm,
            self.seasonal_amplitude_mm,
            self.seasonal_phase_rad,
        )
        if any(not isfinite(value) for vector in vectors for value in vector):
            raise ValueError("all simulation parameters must be finite")
        for vector in (
            self.white_sigma_mm,
            self.ar_innovation_sigma_mm,
            self.seasonal_amplitude_mm,
        ):
            if any(value < 0 for value in vector):
                raise ValueError("noise scales and seasonal amplitudes must be nonnegative")
        if not isfinite(self.ar_rho) or not -1 < self.ar_rho < 1:
            raise ValueError("ar_rho must be finite and between -1 and 1")
        if not isfinite(self.seasonal_period_days) or self.seasonal_period_days <= 0:
            raise ValueError("seasonal_period_days must be finite and positive")
        return self


class GenerationRequest(BaseModel):
    preset: Literal["normal-p1"] = "normal-p1"
    seed: int = Field(default=20260923, ge=0, le=4294967295)
    count: int = Field(default=10, ge=1, le=75)
    config: SimulationConfig = Field(default_factory=SimulationConfig)


class CaseInput(BaseModel):
    schema_version: Literal["p1-input-v1"] = "p1-input-v1"
    case_id: str
    dates: list[date]
    reference_coordinate_mm: AxisVector
    observed_coordinate_mm: list[AxisVector]
    displacement_mm: list[AxisVector]
    horizontal_offset_mm: list[float]
    spatial_offset_mm: list[float]


class CaseTruth(BaseModel):
    schema_version: Literal["p1-truth-v1"] = "p1-truth-v1"
    case_id: str
    events: list[dict] = Field(default_factory=list)
    background_displacement_mm: list[AxisVector]
    white_noise_mm: list[AxisVector]
    ar_noise_mm: list[AxisVector]
    observation_noise_mm: list[AxisVector]
    true_coordinate_mm: list[AxisVector]
    injected_displacement_mm: list[AxisVector]
    observation_artifact_mm: list[AxisVector]
    active_motion: list[bool]
    persistent_offset: list[bool]


class CaseSummary(BaseModel):
    case_id: str
    case_seed: int
    days: int = 180
    event_count: int = 0


class DatasetManifest(BaseModel):
    schema_version: Literal["p1-dataset-v1"] = "p1-dataset-v1"
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: GenerationRequest
    config_sha256: str
    generated_cases: int = 0
    cases: list[CaseSummary] = Field(default_factory=list)
    error: str | None = None
