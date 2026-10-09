from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_baselines import observation_features, robust_activity


def observations_with_auxiliary(values):
    """An intentionally early derivative response must not override the raw activity gate."""
    days = len(values)
    case = SimpleNamespace(displacement_mm=values.tolist())
    auxiliary = SimpleNamespace(speed_mm_day=np.full(days, .2),
                                tangential_acceleration_mm_day2=np.full(days, .01),
                                fitted_displacement_mm=values.tolist())
    return observation_features(case, {window: auxiliary for window in (31, 61, 91)})


def test_derivative_response_on_flat_raw_displacement_does_not_confirm_activity():
    feature = observations_with_auxiliary(np.zeros((180, 3)))
    activity = robust_activity(feature, minimum_rate=.03, minimum_snr=2)
    assert np.count_nonzero(activity == 1) == 0


def test_single_step_or_spike_is_not_sustained_displacement():
    for shape in ("step", "spike"):
        values = np.zeros((180, 3))
        if shape == "step":
            values[90:, 0] = 100
        else:
            values[90, 0] = 100
        feature = observations_with_auxiliary(values)
        assert np.count_nonzero(robust_activity(feature, .03, 2) == 1) == 0


def test_sustained_raw_motion_is_detected_and_translation_does_not_change_gate():
    values = np.zeros((180, 3))
    values[:, 0] = np.arange(180) * .2
    prediction = robust_activity(observations_with_auxiliary(values), .03, 2)
    translated = robust_activity(observations_with_auxiliary(values + [500, -10, 35]), .03, 2)
    assert np.all(prediction[40:140] == 1)
    np.testing.assert_array_equal(prediction, translated)
