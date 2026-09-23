from __future__ import annotations

from datetime import timedelta

import numpy as np

from gnss_sim.schemas import CaseInput, CaseTruth, SimulationConfig


def generate_normal_case(
    case_id: str, case_seed: int, config: SimulationConfig
) -> tuple[CaseInput, CaseTruth]:
    rng = np.random.default_rng(case_seed)
    days = config.days
    t = np.arange(days, dtype=float)[:, None]
    reference = np.asarray(config.reference_coordinate_mm, dtype=float)
    amplitude = np.asarray(config.seasonal_amplitude_mm, dtype=float)
    phase = np.asarray(config.seasonal_phase_rad, dtype=float)
    background = amplitude * (
        np.sin(2 * np.pi * t / config.seasonal_period_days + phase) - np.sin(phase)
    )

    white = rng.normal(size=(days, 3)) * np.asarray(config.white_sigma_mm)
    ar_sigma = np.asarray(config.ar_innovation_sigma_mm)
    innovations = rng.normal(size=(days, 3)) * ar_sigma
    ar = np.empty((days, 3), dtype=float)
    ar[0] = rng.normal(size=3) * ar_sigma / np.sqrt(1 - config.ar_rho**2)
    for index in range(1, days):
        ar[index] = config.ar_rho * ar[index - 1] + innovations[index]

    noise = white + ar
    true_coordinate = reference + background
    observed = true_coordinate + noise
    displacement = observed - reference
    horizontal = np.linalg.norm(displacement[:, :2], axis=1)
    spatial = np.linalg.norm(displacement, axis=1)
    dates = [config.start_date + timedelta(days=index) for index in range(days)]
    zeros = np.zeros((days, 3), dtype=float)

    case_input = CaseInput(
        case_id=case_id,
        dates=dates,
        reference_coordinate_mm=tuple(reference),
        observed_coordinate_mm=observed.tolist(),
        displacement_mm=displacement.tolist(),
        horizontal_offset_mm=horizontal.tolist(),
        spatial_offset_mm=spatial.tolist(),
    )
    truth = CaseTruth(
        case_id=case_id,
        background_displacement_mm=background.tolist(),
        white_noise_mm=white.tolist(),
        ar_noise_mm=ar.tolist(),
        observation_noise_mm=noise.tolist(),
        true_coordinate_mm=true_coordinate.tolist(),
        injected_displacement_mm=zeros.tolist(),
        observation_artifact_mm=zeros.tolist(),
        active_motion=[False] * days,
        persistent_offset=[False] * days,
    )
    return case_input, truth
