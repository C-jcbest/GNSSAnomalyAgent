"""Business contracts for the coordinate consistency mechanism."""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_coordinate_review import agree_or_abstain, coordinate_values


def test_reflection_preserves_calendar_amplitude_and_missingness():
    original = [[2., -3., 4.], None, [-5., 0., 1.]]
    case = SimpleNamespace(displacement_mm=original)
    positive = coordinate_values(case, 1)
    reflected = coordinate_values(case, -1)
    np.testing.assert_allclose(reflected, -positive, equal_nan=True)
    assert np.isnan(reflected[1]).all()
    assert case.displacement_mm == original


def test_disagreement_and_failed_reading_abstain_without_extending_authority():
    initial = np.array([1, 1, 0, 0, -1, 1])
    first = np.array([0, 1, 0, 1, -1, 0])
    second = np.array([0, 0, -1, 1, -1, 1])
    observed = np.array([True, True, True, True, False, True])
    scope = np.array([True, True, True, True, False, False])
    np.testing.assert_array_equal(
        agree_or_abstain(initial, first, second, scope, observed),
        [0, -1, -1, 1, -1, 1],
    )
    failed = np.full(6, -1)
    np.testing.assert_array_equal(
        agree_or_abstain(initial, first, failed, scope, observed),
        [-1, -1, -1, -1, -1, 1],
    )


def test_missing_day_cannot_receive_revision_authority():
    observed = np.array([True, False])
    prediction = np.array([0, -1])
    with pytest.raises(ValueError, match="observed days only"):
        agree_or_abstain(prediction, prediction, prediction, np.array([True, True]), observed)
