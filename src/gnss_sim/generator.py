from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from gnss_sim.schemas import CaseInput, CaseTruth, ComponentSeeds

GENERATOR_VERSION = "normal-v1"
START_DATE = date(2025, 1, 1)
DAYS = 365
PERIOD_DAYS = 365.25
REFERENCE_COORDINATE_MM = (0.0, 0.0, 0.0)
ANNUAL_AMPLITUDE_MM = (2.0, 2.0, 3.0)
SEMIANNUAL_AMPLITUDE_MM = (1.0, 1.0, 2.0)
WHITE_NOISE_SIGMA_MM = (1.5, 1.5, 3.0)


def derive_component_seeds(case_seed: int) -> ComponentSeeds:
    children = np.random.SeedSequence(case_seed).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return ComponentSeeds(
        annual_phase=values[0], semiannual_phase=values[1], white_noise=values[2]
    )


def generate_normal_case(case_id: str, case_seed: int) -> tuple[CaseInput, CaseTruth]:
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
    injected_deformation_mm = np.zeros((DAYS, 3), dtype=float)
    observation_artifact_mm = np.zeros((DAYS, 3), dtype=float)
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
        dates=[START_DATE + timedelta(days=index) for index in range(DAYS)],
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
        annual_phase_rad=tuple(annual_phase),
        semiannual_phase_rad=tuple(semiannual_phase),
        component_seeds=seeds,
    )
    return case_input, truth
