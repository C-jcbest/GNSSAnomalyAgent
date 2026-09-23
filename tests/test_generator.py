from datetime import timedelta

import numpy as np
import pytest

from gnss_sim.generator import generate_normal_case
from gnss_sim.schemas import SimulationConfig


def test_same_seed_reproduces_case_and_daily_dates():
    config = SimulationConfig()
    first_input, first_truth = generate_normal_case("case_0001", 42, config)
    second_input, second_truth = generate_normal_case("case_0001", 42, config)
    assert first_input == second_input
    assert first_truth == second_truth
    assert len(first_input.dates) == 180
    assert first_input.dates == [config.start_date + timedelta(days=i) for i in range(180)]
    assert first_truth.events == []
    assert not any(first_truth.active_motion)
    assert not any(first_truth.persistent_offset)


def test_components_and_fixed_reference_offsets():
    config = SimulationConfig(reference_coordinate_mm=(1000.0, -500.0, 20.0))
    case_input, truth = generate_normal_case("case_0001", 5, config)
    reference = np.asarray(config.reference_coordinate_mm)
    observed = np.asarray(case_input.observed_coordinate_mm)
    background = np.asarray(truth.background_displacement_mm)
    white = np.asarray(truth.white_noise_mm)
    ar = np.asarray(truth.ar_noise_mm)
    assert np.allclose(truth.observation_noise_mm, white + ar)
    assert np.allclose(truth.true_coordinate_mm, reference + background)
    assert np.allclose(observed, reference + background + white + ar)
    assert np.allclose(case_input.displacement_mm, observed - reference)
    assert np.allclose(case_input.horizontal_offset_mm, np.linalg.norm((observed - reference)[:, :2], axis=1))
    assert np.allclose(case_input.spatial_offset_mm, np.linalg.norm(observed - reference, axis=1))


def test_noise_free_case_still_has_configurable_seasonal_background():
    config = SimulationConfig(white_sigma_mm=(0, 0, 0), ar_innovation_sigma_mm=(0, 0, 0))
    case_input, truth = generate_normal_case("case_0001", 7, config)
    assert np.allclose(case_input.observed_coordinate_mm, truth.true_coordinate_mm)
    assert np.any(np.abs(truth.background_displacement_mm) > 0)
    assert np.allclose(truth.injected_displacement_mm, 0)


@pytest.mark.parametrize("updates", [{"ar_rho": 1}, {"white_sigma_mm": (-1, 1, 1)}, {"seasonal_period_days": 0}])
def test_invalid_config_rejected(updates):
    with pytest.raises(ValueError):
        SimulationConfig(**updates)
