from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from gnss_sim import events
from gnss_sim.schemas import CaseInput, CaseTruth, CaseType, ComponentSeeds, EventSeeds

GENERATOR_VERSION = "event-v5"
START_DATE = date(2025, 1, 1)
DAYS = 365
PERIOD_DAYS = 365.25
REFERENCE_COORDINATE_MM = (0.0, 0.0, 0.0)
ANNUAL_AMPLITUDE_MM = (1.0, 1.0, 1.5)
SEMIANNUAL_AMPLITUDE_MM = (0.25, 0.25, 0.5)
WHITE_NOISE_SIGMA_MM = (0.5, 0.5, 1.0)
EVENT_MAGNITUDE_MM = {
    "spike": (4.5, 4.5, 9.0),
    "step": (3.75, 3.75, 7.5),
    "slow_trend": (4.5, 4.5, 9.0),
    "acceleration": (4.5, 4.5, 9.0),
    "transient_shift": (3.75, 3.75, 7.5),
}


def derive_component_seeds(case_seed: int) -> ComponentSeeds:
    children = np.random.SeedSequence(case_seed).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return ComponentSeeds(
        annual_phase=values[0], semiannual_phase=values[1], white_noise=values[2]
    )


def derive_event_seeds(case_seed: int) -> EventSeeds:
    # A distinct namespace leaves the frozen P1 phase/noise streams untouched.
    children = np.random.SeedSequence([case_seed, 0x45564E54]).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return EventSeeds(position=values[0], shape=values[1], sign=values[2])


def generate_case(
    case_id: str, case_seed: int, case_type: CaseType
) -> tuple[CaseInput, CaseTruth]:
    seeds = derive_component_seeds(case_seed)
    annual_phase = np.random.default_rng(seeds.annual_phase).uniform(0, 2 * np.pi, size=3)
    semiannual_phase = np.random.default_rng(seeds.semiannual_phase).uniform(
        0, 2 * np.pi, size=3
    )
    t = np.arange(DAYS, dtype=float)[:, None]
    annual = np.asarray(ANNUAL_AMPLITUDE_MM) * np.sin(
        2 * np.pi * t / PERIOD_DAYS + annual_phase
    )
    semiannual = np.asarray(SEMIANNUAL_AMPLITUDE_MM) * np.sin(
        4 * np.pi * t / PERIOD_DAYS + semiannual_phase
    )
    normal_background_mm = annual + semiannual
    measurement_noise_mm = np.random.default_rng(seeds.white_noise).normal(
        size=(DAYS, 3)
    ) * np.asarray(WHITE_NOISE_SIGMA_MM)
    reference_coordinate_mm = np.asarray(REFERENCE_COORDINATE_MM)
    dates = [START_DATE + timedelta(days=index) for index in range(DAYS)]
    injected_deformation_mm = np.zeros((DAYS, 3), dtype=float)
    observation_artifact_mm = np.zeros((DAYS, 3), dtype=float)
    event_seeds = None
    event_truth = []
    if case_type != "normal":
        event_seeds = derive_event_seeds(case_seed)
        shape_rng = np.random.default_rng(event_seeds.shape)
        axis = ("N", "E", "U")[int(shape_rng.integers(0, 3))]
        magnitude = EVENT_MAGNITUDE_MM[case_type][("N", "E", "U").index(axis)]
        duration = (
            events.TREND_DURATION
            if case_type in ("slow_trend", "acceleration")
            else events.TRANSIENT_DURATION if case_type == "transient_shift" else 1
        )
        last_start = events.SAFE_END - duration + 1
        start = int(np.random.default_rng(event_seeds.position).integers(events.SAFE_START, last_start + 1))
        sign = 1 if np.random.default_rng(event_seeds.sign).integers(0, 2) else -1
        if case_type == "spike":
            contribution, event = events.make_spike(dates, axis, start, sign * magnitude)
        elif case_type == "step":
            contribution, event = events.make_step(dates, axis, start, sign * magnitude)
        elif case_type == "slow_trend":
            contribution, event = events.make_slow_trend(dates, axis, start, sign * magnitude)
        elif case_type == "acceleration":
            contribution, event = events.make_acceleration(dates, axis, start, sign * magnitude)
        else:
            contribution, event = events.make_transient_shift(dates, axis, start, sign * magnitude)
        event_truth = [event]
        if event.source == "injected_deformation":
            injected_deformation_mm = contribution
        else:
            observation_artifact_mm = contribution
    observed_coordinate_mm = (
        reference_coordinate_mm
        + normal_background_mm
        + injected_deformation_mm
        + measurement_noise_mm
        + observation_artifact_mm
    )
    displacement_mm = observed_coordinate_mm - reference_coordinate_mm

    case_input = CaseInput(
        case_id=case_id,
        dates=dates,
        reference_coordinate_mm=REFERENCE_COORDINATE_MM,
        observed_coordinate_mm=observed_coordinate_mm.tolist(),
        displacement_mm=displacement_mm.tolist(),
        horizontal_offset_mm=np.linalg.norm(displacement_mm[:, :2], axis=1).tolist(),
        spatial_offset_mm=np.linalg.norm(displacement_mm, axis=1).tolist(),
    )
    truth = CaseTruth(
        case_id=case_id,
        normal_background_mm=normal_background_mm.tolist(),
        measurement_noise_mm=measurement_noise_mm.tolist(),
        injected_deformation_mm=injected_deformation_mm.tolist(),
        observation_artifact_mm=observation_artifact_mm.tolist(),
        annual_phase_rad=tuple(annual_phase),
        semiannual_phase_rad=tuple(semiannual_phase),
        component_seeds=seeds,
        event_seeds=event_seeds,
        events=event_truth,
    )
    return case_input, truth


def generate_normal_case(case_id: str, case_seed: int) -> tuple[CaseInput, CaseTruth]:
    return generate_case(case_id, case_seed, "normal")
