"""Mechanism-inspired, multi-year landslide simulations, not field observations.

The daily motion is generated as a nonnegative speed and then integrated in a
slowly varying slope direction. Observation errors never become physical motion.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from gnss_sim.schemas import StrictModel, Vector

Scenario = Literal[
    "stable", "slow_creep", "seasonal_steps", "progressive_acceleration",
    "acceleration_arrest", "reactivation",
    "mixed_stages",
]
SCENARIOS: tuple[Scenario, ...] = (
    "stable", "slow_creep", "seasonal_steps", "progressive_acceleration",
    "acceleration_arrest", "reactivation",
    "mixed_stages",
)


class LandslideRequest(StrictModel):
    seed: int = Field(strict=True, ge=0, le=4294967295)
    count: int = Field(default=24, strict=True, ge=1, le=120)
    days: int = Field(default=1095, strict=True, ge=730, le=2192)
    start_date: date = date(2022, 1, 1)

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.start_date.toordinal() + self.days - 1 > date.max.toordinal():
            raise ValueError("起始日期加观测天数超出日期范围")
        return self


class LandslideInput(StrictModel):
    schema_version: Literal["landslide-input-v1"] = "landslide-input-v1"
    case_id: str = Field(pattern=r"^case_\d{4}$")
    dates: list[date] = Field(min_length=730, max_length=2192)
    reference_coordinate_mm: Vector = (0, 0, 0)
    displacement_mm: list[Vector | None]

    @model_validator(mode="after")
    def validate_daily_observations(self):
        if len(self.dates) != len(self.displacement_mm):
            raise ValueError("日期与观测长度不一致")
        if any(b - a != timedelta(days=1) for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("日期必须逐日连续，缺测用 null 表示")
        return self


class MotionPhase(StrictModel):
    kind: str
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    description: str

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_index < self.start_index:
            raise ValueError("阶段结束早于开始")
        return self


class LandslideTruth(StrictModel):
    schema_version: Literal["landslide-truth-v1"] = "landslide-truth-v1"
    case_id: str
    scenario: Scenario
    seed: int
    latent_displacement_mm: list[Vector]
    measurement_noise_mm: list[Vector]
    observation_artifact_mm: list[Vector]
    daily_rate_mm: list[float]
    rainfall_mm: list[float]
    missing_indices: list[int]
    phases: list[MotionPhase]
    parameters: dict[str, float | int | str | list[float]]


class LandslideSummary(StrictModel):
    case_id: str
    scenario: Scenario
    observed_days: int
    missing_days: int


class LandslideManifest(StrictModel):
    schema_version: Literal["landslide-dataset-v1"] = "landslide-dataset-v1"
    generator_version: Literal["landslide-v1"] = "landslide-v1"
    data_origin: Literal["mechanism-inspired simulation"] = "mechanism-inspired simulation"
    generator_sha256: str
    numpy_version: str = np.__version__
    dataset_id: str
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    request: LandslideRequest
    generated_cases: int = 0
    type_counts: dict[Scenario, int]
    observed_extent_mm: tuple[float, float] = (0, 0)
    cases: list[LandslideSummary] = Field(default_factory=list)
    error: str | None = None


def allocate_scenarios(request: LandslideRequest) -> list[Scenario]:
    """Most records contain multiple stages; retain quiet controls in every batch."""
    rng = np.random.default_rng(np.random.SeedSequence([request.seed, 0xBA7C]))
    scenarios: list[Scenario] = ["mixed_stages"] * request.count
    for index in range(5, request.count, 6):
        scenarios[index] = "stable"
    rng.shuffle(scenarios)
    return scenarios


def _correlated_variation(rng: np.random.Generator, days: int, memory: float) -> np.ndarray:
    values = np.empty(days)
    values[0] = rng.normal()
    innovations = rng.normal(0, np.sqrt(1 - memory ** 2), days)
    for index in range(1, days):
        values[index] = memory * values[index - 1] + innovations[index]
    return values


def _smooth_transition(day: np.ndarray, start: int, end: int) -> np.ndarray:
    fraction = np.clip((day - start) / max(end - start, 1), 0, 1)
    return fraction ** 2 * (3 - 2 * fraction)


def _hydrology(rng: np.random.Generator, dates: list[date]):
    """A synthetic wet-season driver, with independent timing/severity each year."""
    rainfall = np.zeros(len(dates))
    wetness = np.zeros(len(dates))
    day_of_year = np.array([day.timetuple().tm_yday for day in dates])
    years = np.array([day.year for day in dates])
    for year in np.unique(years):
        indices = np.flatnonzero(years == year)
        center = rng.uniform(160, 235)
        width = rng.uniform(32, 68)
        season = np.exp(-0.5 * ((day_of_year[indices] - center) / width) ** 2)
        rainy = rng.random(len(indices)) < 0.07 + rng.uniform(0.28, 0.60) * season
        rainfall[indices] = rainy * rng.gamma(1.4, rng.uniform(7, 19), len(indices))
    memory = rng.uniform(0.92, 0.98)
    for index in range(1, len(dates)):
        wetness[index] = memory * wetness[index - 1] + (1 - memory) * rainfall[index]
    return rainfall, wetness, memory


def _motion_rate(rng: np.random.Generator, scenario: Scenario, wetness: np.ndarray):
    if scenario == "mixed_stages":
        return _mixed_motion_rate(rng, wetness)
    days = len(wetness)
    day = np.arange(days)
    phases = []
    base_rate = rng.uniform(0.012, 0.065)
    rate = np.full(days, base_rate)
    parameters: dict[str, float | int | str | list[float]] = {"base_rate_mm_day": base_rate}
    wet_response = np.maximum(wetness - rng.uniform(2, 5), 0) / 10

    if scenario == "stable":
        rate[:] = 0
        parameters["base_rate_mm_day"] = 0.0
        phases.append(MotionPhase(kind="stable", start_index=0, end_index=days - 1,
                                  description="全程静稳对照；测量波动不属于坡体形变"))
    elif scenario == "slow_creep":
        rate *= rng.uniform(0.65, 1.8)
        rate += rng.uniform(0.005, 0.025) * wet_response
        phases.append(MotionPhase(kind="creep", start_index=0, end_index=days - 1,
                                  description="持续低速累积，速率随时间缓变"))
    elif scenario == "seasonal_steps":
        rate *= 0.25
        sensitivity = rng.uniform(0.18, 0.65)
        rate += sensitivity * wet_response ** rng.uniform(1.2, 1.9)
        parameters["wet_response_gain"] = sensitivity
        phases.append(MotionPhase(kind="seasonal_activity", start_index=0, end_index=days - 1,
                                  description="不同年份湿季强度与时间不同，间歇加速后回到低速"))
    elif scenario == "progressive_acceleration":
        start = int(rng.uniform(0.40, 0.67) * days)
        exponent = rng.uniform(1.2, 2.8)
        gain = rng.uniform(0.6, 1.8)
        progress = np.maximum((day - start) / (days - start), 0)
        rate += gain * progress ** exponent
        rate *= 1 + 0.15 * wet_response
        parameters.update(acceleration_gain_mm_day=gain, acceleration_exponent=exponent)
        phases.append(MotionPhase(kind="acceleration", start_index=start, end_index=days - 1,
                                  description="总体持续增速，叠加短暂回落；不设定失稳时刻"))
    elif scenario == "acceleration_arrest":
        start = int(rng.uniform(0.20, 0.43) * days)
        peak = start + int(rng.uniform(0.09, 0.17) * days)
        end = min(days - 60, peak + int(rng.uniform(0.10, 0.24) * days))
        gain = rng.uniform(0.35, 1.1)
        rate += gain * _smooth_transition(day, start, peak) * (
            1 - _smooth_transition(day, peak, end))
        parameters["transient_gain_mm_day"] = gain
        phases.extend([
            MotionPhase(kind="acceleration", start_index=start, end_index=peak,
                        description="逐步加速，局部速率允许波动"),
            MotionPhase(kind="deceleration", start_index=peak + 1, end_index=end,
                        description="活动减弱并恢复低速，已累积位移保留"),
        ])
    else:
        rate *= 0.18
        start = int(rng.uniform(0.10, 0.24) * days)
        while start < days - 80:
            duration = int(rng.uniform(55, 165))
            end = min(start + duration, days - 1)
            peak = start + int((end - start) * rng.uniform(0.2, 0.5))
            gain = rng.uniform(0.18, 0.85)
            rate += gain * _smooth_transition(day, start, peak) * (
                1 - _smooth_transition(day, peak, end))
            phases.append(MotionPhase(kind="reactivation", start_index=start, end_index=end,
                                      description="低速间歇后的再活动，时长和幅度不固定"))
            start = end + int(rng.uniform(80, 260))

    # Two time scales perturb the speed, not the displacement: no rigid straight ramps.
    modulation = (0.24 * _correlated_variation(rng, days, 0.985)
                  + 0.09 * _correlated_variation(rng, days, 0.65))
    rate *= np.exp(modulation)
    return rate, phases, parameters


def _mixed_motion_rate(rng: np.random.Generator, wetness: np.ndarray):
    """Assemble episodes on a calendar, then crop to the observation window.

    Quiet intervals have zero latent speed, not zero cumulative displacement.
    Phase labels describe the designed mechanism, not observation-based decisions.
    """
    days = len(wetness)
    rate = np.zeros(days)
    phases: list[MotionPhase] = []
    boundary_mode = str(rng.choice(["early", "middle", "late", "left_censored"]))
    if boundary_mode == "left_censored":
        cursor = -int(rng.integers(25, 140))
    elif boundary_mode == "early":
        cursor = int(rng.uniform(0.02, 0.12) * days)
    elif boundary_mode == "middle":
        cursor = int(rng.uniform(0.18, 0.35) * days)
    else:
        cursor = int(rng.uniform(0.45, 0.62) * days)

    descriptions = {
        "stable": "首次活动前静稳，潜在累计位移保持不变",
        "creep": "低速形变累积，含不规则速率波动",
        "acceleration": "本次活动总体增速，允许局部短暂减缓",
        "steady_slip": "相对本次活动背景的较高速持续滑移",
        "deceleration": "速度总体下降，累计位移仍可继续增加",
        "dormant": "活动后停滞，保留已累积位移；观测仍含噪声",
    }

    def add_stage(kind: str, start: int, duration: int, initial: float, final: float):
        stop = start + duration
        visible_start = max(0, start)
        visible_stop = min(days, stop)
        if visible_start >= visible_stop:
            return stop
        progress = (np.arange(visible_start, visible_stop) - start) / max(duration - 1, 1)
        if initial == final:
            stage_rate = np.full(len(progress), initial)
        else:
            shape = progress ** rng.uniform(0.85, 2.4)
            stage_rate = initial + (final - initial) * shape
        rate[visible_start:visible_stop] = stage_rate
        description = descriptions[kind]
        if start < 0:
            description += "；记录开始前已经进入此阶段（左截断）"
        if stop > days:
            description += "；记录结束时阶段尚未结束（右截断）"
        phases.append(MotionPhase(kind=kind, start_index=visible_start,
                                  end_index=visible_stop - 1, description=description))
        return stop

    if cursor > 0:
        add_stage("stable", 0, cursor, 0, 0)
    episode_count = 0
    while cursor < days:
        episode_count += 1
        low_rate = rng.uniform(0.025, 0.12)
        peak_rate = rng.uniform(0.20, 1.15)
        episode_kind = str(rng.choice(["creep_only", "surge", "multiwave", "gradual"],
                                      p=[0.15, 0.20, 0.30, 0.35]))
        current_rate = 0.0
        if episode_kind != "surge":
            cursor = add_stage("creep", cursor, int(rng.integers(35, 150)), 0, low_rate)
            current_rate = low_rate
        if episode_kind != "creep_only":
            cursor = add_stage("acceleration", cursor, int(rng.integers(22, 115)),
                               current_rate, peak_rate)
            current_rate = peak_rate
            if episode_kind == "multiwave" or rng.random() < 0.55:
                cursor = add_stage("steady_slip", cursor, int(rng.integers(20, 100)),
                                   current_rate, current_rate)
            if episode_kind == "multiwave":
                next_peak = current_rate * rng.uniform(1.3, 2.4)
                cursor = add_stage("acceleration", cursor, int(rng.integers(18, 75)),
                                   current_rate, next_peak)
                current_rate = next_peak
        cursor = add_stage("deceleration", cursor, int(rng.integers(25, 120)), current_rate, 0)
        cursor = add_stage("dormant", cursor, int(rng.integers(55, 220)), 0, 0)

    # A shared continuous modulation avoids extra jumps at designed phase boundaries.
    fluctuation = (0.17 * _correlated_variation(rng, days, 0.987)
                   + 0.055 * _correlated_variation(rng, days, 0.72))
    wet_response = np.maximum(wetness - rng.uniform(2, 5), 0) / 10
    rate *= np.exp(fluctuation) * (1 + rng.uniform(0.06, 0.30) * wet_response)
    parameters = {
        "composition": "mixed-stages-v1",
        "boundary_mode": boundary_mode,
        "episode_count": episode_count,
        "phase_semantics": "完整互斥的生成分段；非观测识别结论，无通用速度分级阈值",
    }
    return rate, phases, parameters


def _measurement_noise(rng: np.random.Generator, days: int, rainfall: np.ndarray):
    sigma = np.array([rng.uniform(0.5, 1.3), rng.uniform(0.5, 1.3), rng.uniform(1.1, 2.7)])
    noise = rng.normal(size=(days, 3))
    frequency = np.fft.rfftfreq(2 * days)
    spectral_scale = np.zeros_like(frequency)
    spectral_scale[1:] = 1 / np.sqrt(frequency[1:])
    for axis in range(3):
        # Generate twice the record length and crop, avoiding a forced end-to-start join.
        flicker = np.fft.irfft(
            np.fft.rfft(rng.normal(size=2 * days)) * spectral_scale, n=2 * days,
        )[:days]
        flicker /= np.sqrt(2 * np.sum(spectral_scale[1:-1] ** 2) / (2 * days))
        noise[:, axis] += rng.uniform(0.45, 0.85) * flicker
        noise[:, axis] += 0.3 * _correlated_variation(rng, days, 0.7)
    quality_scale = 1 + np.minimum(rainfall / 90, 0.8)
    noise *= sigma * quality_scale[:, None]
    return noise, sigma


def simulate_case(request: LandslideRequest, index: int, scenario: Scenario):
    seed = int(np.random.SeedSequence([request.seed, index, 0x51DE]).generate_state(1)[0])
    # Independent streams keep the motion independent of observation quality/missingness.
    streams = np.random.SeedSequence(seed).spawn(3)
    motion_rng, noise_rng, quality_rng = [np.random.default_rng(stream) for stream in streams]
    days = request.days
    dates = [request.start_date + timedelta(days=index) for index in range(days)]
    rainfall, wetness, water_memory = _hydrology(motion_rng, dates)
    rate, phases, parameters = _motion_rate(motion_rng, scenario, wetness)

    azimuth = motion_rng.uniform(0, 2 * np.pi)
    dip = motion_rng.uniform(np.deg2rad(8), np.deg2rad(38))
    direction_drift = 0.025 * _correlated_variation(motion_rng, days, 0.997)
    direction = np.column_stack((np.cos(azimuth + direction_drift) * np.cos(dip),
                                 np.sin(azimuth + direction_drift) * np.cos(dip),
                                 np.full(days, -np.sin(dip))))
    latent = np.cumsum(rate[:, None] * direction, axis=0)
    latent -= latent[0]
    noise, sigma = _measurement_noise(noise_rng, days, rainfall)
    artifacts = np.zeros((days, 3))
    outlier_indices = np.flatnonzero(quality_rng.random(days) < quality_rng.uniform(0.001, 0.004))
    artifacts[outlier_indices] += quality_rng.normal(0, 5, (len(outlier_indices), 3)) * sigma
    if quality_rng.random() < 0.25:
        offset_day = int(quality_rng.integers(days // 5, days * 4 // 5))
        artifacts[offset_day:] += quality_rng.normal(0, 3, 3) * sigma
        parameters["instrument_offset_index"] = offset_day
    missing = quality_rng.random(days) < quality_rng.uniform(0.005, 0.025)
    for _ in range(int(quality_rng.integers(1, 4))):
        start = int(quality_rng.integers(20, days - 25))
        missing[start:start + int(quality_rng.integers(4, 18))] = True

    observed = latent + noise + artifacts
    case_id = f"case_{index + 1:04d}"
    parameters.update(
        azimuth_degrees=float(np.rad2deg(azimuth)), dip_degrees=float(np.rad2deg(dip)),
        white_noise_sigma_mm=sigma.tolist(), hydrology_memory=float(water_memory),
        interpretation="设计参数，未用现场数据标定；阶段为生成机制区间，不是诊断真值",
    )
    case = LandslideInput(case_id=case_id, dates=dates,
                          displacement_mm=[None if missing[i] else row.tolist()
                                           for i, row in enumerate(observed)])
    truth = LandslideTruth(
        case_id=case_id, scenario=scenario, seed=seed, latent_displacement_mm=latent.tolist(),
        measurement_noise_mm=noise.tolist(), observation_artifact_mm=artifacts.tolist(),
        daily_rate_mm=rate.tolist(), rainfall_mm=rainfall.tolist(),
        missing_indices=np.flatnonzero(missing).tolist(), phases=phases, parameters=parameters,
    )
    return case, truth
