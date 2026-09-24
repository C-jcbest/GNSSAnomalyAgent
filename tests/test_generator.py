from datetime import date, timedelta

import numpy as np
import pytest

from gnss_sim.generator import (
    ANNUAL_AMPLITUDE_MM,
    DAYS,
    PERIOD_DAYS,
    SEMIANNUAL_AMPLITUDE_MM,
    START_DATE,
    WHITE_NOISE_SIGMA_MM,
    derive_component_seeds,
    generate_normal_case,
)
from gnss_sim.schemas import GenerationRequest


def test_same_seed_reproduces_365_daily_observations():
    first_input, first_truth = generate_normal_case("case_0001", 42)
    second_input, second_truth = generate_normal_case("case_0001", 42)
    assert first_input == second_input
    assert first_truth == second_truth
    assert len(first_input.dates) == DAYS == 365
    assert first_input.dates == [START_DATE + timedelta(days=i) for i in range(DAYS)]
    assert first_input.dates[-1] == date(2025, 12, 31)
    assert first_truth.events == []
    assert first_input.reference_coordinate_mm == (0.0, 0.0, 0.0)


def test_frozen_controlled_background_parameters():
    assert ANNUAL_AMPLITUDE_MM == (1.0, 1.0, 1.5)
    assert SEMIANNUAL_AMPLITUDE_MM == (0.25, 0.25, 0.5)
    assert WHITE_NOISE_SIGMA_MM == (0.75, 0.75, 1.5)


def test_fixed_components_and_reference_offsets():
    case_input, truth = generate_normal_case("case_0001", 5)
    reference = np.asarray(case_input.reference_coordinate_mm)
    observed = np.asarray(case_input.observed_coordinate_mm)
    background = np.asarray(truth.normal_background_mm)
    noise = np.asarray(truth.measurement_noise_mm)
    t = np.arange(DAYS)[:, None]
    expected_background = np.asarray(ANNUAL_AMPLITUDE_MM) * np.sin(
        2 * np.pi * t / PERIOD_DAYS + truth.annual_phase_rad
    ) + np.asarray(SEMIANNUAL_AMPLITUDE_MM) * np.sin(
        4 * np.pi * t / PERIOD_DAYS + truth.semiannual_phase_rad
    )
    expected_noise = np.random.default_rng(truth.component_seeds.white_noise).normal(
        size=(DAYS, 3)
    ) * np.asarray(WHITE_NOISE_SIGMA_MM)
    assert np.allclose(background, expected_background)
    assert np.allclose(noise, expected_noise)
    assert np.allclose(observed, reference + background + noise)
    assert np.allclose(case_input.displacement_mm, observed - reference)
    assert np.allclose(case_input.horizontal_offset_mm, np.linalg.norm((observed - reference)[:, :2], axis=1))
    assert np.allclose(case_input.spatial_offset_mm, np.linalg.norm(observed - reference, axis=1))


def test_phase_noise_seeds_are_separate_and_recorded():
    seeds = derive_component_seeds(19)
    _, truth = generate_normal_case("case_0001", 19)
    assert truth.component_seeds == seeds
    assert len({seeds.annual_phase, seeds.semiannual_phase, seeds.white_noise}) == 3
    assert all(0 <= phase < 2 * np.pi for phase in truth.annual_phase_rad)
    assert all(0 <= phase < 2 * np.pi for phase in truth.semiannual_phase_rad)
    assert len(set(truth.annual_phase_rad)) == 3
    assert len(set(truth.semiannual_phase_rad)) == 3
    _, other_truth = generate_normal_case("case_0001", 20)
    assert other_truth.normal_background_mm != truth.normal_background_mm


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 42, "count": 1, "preset": "normal-p1"},
        {"seed": 42, "count": 1, "config": {"days": 180}},
        {"seed": 42, "count": 1, "days": 90},
        {"seed": 42, "count": 1, "ar_rho": 0.7},
    ],
)
def test_old_generation_options_are_rejected(payload):
    with pytest.raises(ValueError):
        GenerationRequest.model_validate(payload)


def test_generation_count_is_engineering_cap_not_pilot_size():
    assert GenerationRequest(seed=42, count=76, case_type="normal").count == 76
    with pytest.raises(ValueError):
        GenerationRequest(seed=42, count=5001, case_type="normal")
