from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from gnss_sim.events import (
    make_acceleration,
    make_slow_trend,
    make_spike,
    make_step,
    make_transient_shift,
)
from gnss_sim.generator import (
    DAYS,
    START_DATE,
    WHITE_NOISE_SIGMA_MM,
    derive_event_seeds,
    generate_case,
    generate_normal_case,
)
from gnss_sim.schemas import CaseTruth

DATES = [START_DATE + timedelta(days=i) for i in range(DAYS)]
TYPES = ("spike", "step", "slow_trend", "acceleration", "transient_shift")


@pytest.mark.parametrize("axis", ("N", "E", "U"))
def test_pure_spike_and_step(axis):
    column = ("N", "E", "U").index(axis)
    spike, point = make_spike(DATES, axis, 100, -7.5)
    expected = np.zeros((DAYS, 3))
    expected[100, column] = -7.5
    np.testing.assert_array_equal(spike, expected)
    assert (point.start_index, point.end_index, point.persistent) == (100, 100, False)
    assert spike[101, column] == 0

    step, onset = make_step(DATES, axis, 100, 2.5)
    expected[:] = 0
    expected[100:, column] = 2.5
    np.testing.assert_array_equal(step, expected)
    assert (onset.start_index, onset.end_index, onset.persistent) == (100, 100, True)
    assert onset.parameters.amplitude_mm == 2.5


@pytest.mark.parametrize("axis", ("N", "E", "U"))
def test_pure_slow_and_acceleration(axis):
    column = ("N", "E", "U").index(axis)
    slow, slow_event = make_slow_trend(DATES, axis, 100, -9.0)
    expected = np.zeros((DAYS, 3))
    expected[100:, column] = -9.0 * np.minimum(np.arange(DAYS - 100), 89) / 89
    np.testing.assert_allclose(slow, expected, rtol=0, atol=1e-12)
    assert slow_event.end_index == 189
    assert slow_event.parameters.duration_days == 90
    assert slow_event.parameters.slope_mm_per_day == pytest.approx(-9 / 89)
    assert slow[189, column] == pytest.approx(-9)
    assert slow[190, column] == pytest.approx(-9)

    acceleration, convex = make_acceleration(DATES, axis, 100, 9.0)
    expected[:] = 0
    expected[100:, column] = 9 * (np.minimum(np.arange(DAYS - 100), 89) / 89) ** 2
    np.testing.assert_allclose(acceleration, expected, rtol=0, atol=1e-12)
    increments = np.diff(acceleration[100:190, column])
    assert np.all(increments > 0)
    assert np.all(np.diff(increments) > 0)
    assert convex.end_index == 189
    assert acceleration[189, column] == pytest.approx(9)
    assert acceleration[190, column] == pytest.approx(9)


@pytest.mark.parametrize("axis", ("N", "E", "U"))
def test_pure_transient_shift(axis):
    column = ("N", "E", "U").index(axis)
    transient, event = make_transient_shift(DATES, axis, 100, -6.0)
    expected = np.zeros((DAYS, 3))
    expected[100:114, column] = -6.0
    np.testing.assert_array_equal(transient, expected)
    assert event.end_index == 113
    assert event.parameters.duration_days == 14
    assert not event.persistent
    assert transient[114, column] == 0


@pytest.mark.parametrize("case_type", TYPES)
def test_generated_event_contract_and_pointwise_composition(case_type):
    case_input, truth = generate_case("case_0001", 729, case_type)
    assert len(truth.events) == 1
    event = truth.events[0]
    assert event.type == case_type
    assert event.start_date == case_input.dates[event.start_index]
    assert event.end_date == case_input.dates[event.end_index]
    assert 60 <= event.start_index <= event.end_index <= 304
    assert truth.event_seeds == derive_event_seeds(729)
    deformation = np.asarray(truth.injected_deformation_mm)
    artifact = np.asarray(truth.observation_artifact_mm)
    if case_type in ("spike", "transient_shift"):
        assert event.source == "observation_artifact"
        assert not deformation.any()
        assert artifact.any()
    else:
        assert event.source == "injected_deformation"
        assert not artifact.any()
        assert deformation.any()
    axis = ("N", "E", "U").index(event.axis)
    assert not np.delete(deformation + artifact, axis, axis=1).any()
    observed = np.asarray(case_input.observed_coordinate_mm)
    expected = (
        np.asarray(case_input.reference_coordinate_mm)
        + np.asarray(truth.normal_background_mm)
        + np.asarray(truth.measurement_noise_mm)
        + deformation
        + artifact
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-15)
    assert case_input.dates[-1] == START_DATE + timedelta(days=364)


@pytest.mark.parametrize("case_type", TYPES)
def test_paired_background_is_bitwise_identical(case_type):
    normal_input, normal_truth = generate_normal_case("case_0001", 42)
    variant_input, variant_truth = generate_case("case_0001", 42, case_type)
    assert variant_truth.normal_background_mm == normal_truth.normal_background_mm
    assert variant_truth.measurement_noise_mm == normal_truth.measurement_noise_mm
    assert variant_truth.component_seeds == normal_truth.component_seeds
    delta = np.asarray(variant_input.observed_coordinate_mm) - np.asarray(
        normal_input.observed_coordinate_mm
    )
    contribution = np.asarray(variant_truth.injected_deformation_mm) + np.asarray(
        variant_truth.observation_artifact_mm
    )
    np.testing.assert_allclose(delta, contribution, rtol=0, atol=5e-15)


def test_event_truth_discriminator_and_date_validation():
    _, truth = generate_case("case_0001", 20, "spike")
    payload = truth.model_dump(mode="json")
    payload["events"][0]["start_date"] = "2025-01-01"
    with pytest.raises(ValidationError):
        CaseTruth.model_validate(payload)
    payload = truth.model_dump(mode="json")
    payload["events"][0]["parameters"]["made_up"] = 1
    with pytest.raises(ValidationError):
        CaseTruth.model_validate(payload)
    payload = truth.model_dump(mode="json")
    payload["events"].append(payload["events"][0])
    with pytest.raises(ValidationError):
        CaseTruth.model_validate(payload)


def test_event_safe_zone_guard():
    for function, args in ((make_spike, ("N", 59, 7.5)), (make_slow_trend, ("E", 216, 9.0))):
        with pytest.raises(ValueError):
            function(DATES, *args)


@pytest.mark.parametrize("case_type", TYPES)
@pytest.mark.parametrize("axis", ("N", "E", "U"))
def test_all_15_generated_type_axis_combinations(case_type, axis):
    matches = (
        generate_case("case_0001", case_seed, case_type)
        for case_seed in range(30)
    )
    case_input, truth = next(
        (item for item in matches if item[1].events[0].axis == axis), None
    )
    assert case_input is not None
    event = truth.events[0]
    column = ("N", "E", "U").index(axis)
    contribution = np.asarray(truth.injected_deformation_mm) + np.asarray(
        truth.observation_artifact_mm
    )
    profile = contribution[:, column]
    sigma = WHITE_NOISE_SIGMA_MM[column]
    assert np.count_nonzero(profile) > 0
    if case_type == "spike":
        assert np.count_nonzero(profile) == 1
        assert abs(profile[event.start_index]) == 6 * sigma
        assert event.parameters.duration_days == 1
    elif case_type == "step":
        assert abs(profile[-1]) == 5 * sigma
        assert event.parameters.duration_days == 1
        assert np.all(profile[event.start_index:] == profile[-1])
    elif case_type in ("slow_trend", "acceleration"):
        assert abs(profile[-1]) == 6 * sigma
        assert event.end_index - event.start_index + 1 == 90
    else:
        assert abs(profile[event.start_index]) == 5 * sigma
        assert event.parameters.duration_days == 14
        assert np.count_nonzero(profile) == 14
        assert profile[event.end_index + 1] == 0
