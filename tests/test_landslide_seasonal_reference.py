from datetime import date, timedelta

import numpy as np

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_seasonal_reference import (
    apply_annual_reference,
    fit_annual_reference,
)


def reference_case(name, start, coefficients, missing=()):
    dates = [start + timedelta(days=day) for day in range(730)]
    observed = np.ones(730, dtype=bool)
    observed[list(missing)] = False
    time = np.array([(day - date(2000, 1, 1)).days for day in dates], dtype=float)
    annual = np.column_stack((np.sin(2 * np.pi * time / 365.25), np.cos(2 * np.pi * time / 365.25)))
    values = annual @ coefficients
    trend = np.column_stack((np.ones(730), np.arange(730)))
    values -= trend @ np.linalg.lstsq(trend[observed], values[observed], rcond=None)[0]
    case = LandslideInput(case_id=name, dates=dates,
        displacement_mm=[row.tolist() if available else None for row, available in zip(values, observed)])
    return case, values, observed


def test_training_reference_predicts_a_different_calendar_without_refitting():
    coefficients = np.array([[3., -2., 8.], [-1., 4., 2.]])
    first, _, _ = reference_case("case_9001", date(2006, 2, 17), coefficients)
    second, _, _ = reference_case("case_9002", date(2018, 2, 14), coefficients, [100, 101])
    model = fit_annual_reference([first, second])
    unseen, expected, observed = reference_case("case_9003", date(2024, 2, 13), coefficients, [500, 501])
    corrected, predicted = apply_annual_reference(unseen, model)
    np.testing.assert_allclose(predicted, expected, atol=1e-9)
    np.testing.assert_allclose(np.array([row for row in corrected.displacement_mm if row is not None]), 0, atol=1e-9)
    assert model["training_case_ids"] == ["case_9001", "case_9002"]
    assert [row is not None for row in corrected.displacement_mm] == observed.tolist()


def test_frozen_reference_does_not_fit_away_current_motion_or_fill_gaps():
    coefficients = np.array([[3., -2., 8.], [-1., 4., 2.]])
    first, _, _ = reference_case("case_9001", date(2006, 2, 17), coefficients)
    second, _, _ = reference_case("case_9002", date(2018, 2, 14), coefficients)
    model = fit_annual_reference([first, second])
    current, background, observed = reference_case("case_9003", date(2024, 2, 13), coefficients, [250, 251])
    increment = np.zeros((730, 3))
    increment[200:420] = np.arange(220)[:, None] * np.array([.04, -.03, -.02])
    increment[420:] = increment[419]
    current = current.model_copy(update={"displacement_mm": [row.tolist() if available else None
        for row, available in zip(background + increment, observed)]})
    corrected, _ = apply_annual_reference(current, model)
    values = np.array([row for row in corrected.displacement_mm if row is not None])
    np.testing.assert_allclose(values, increment[observed], atol=1e-9)
    assert corrected.displacement_mm[250:252] == [None, None]
