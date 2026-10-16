from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_endpoint_fit import apply_endpoint_fits, fit_activity_endpoints
from gnss_sim.landslide_evaluation import event_masks


def moving_case():
    days = np.arange(420)
    displacement = np.clip(days - 120, 0, 140)
    values = displacement[:, None] * np.array([0.2, -0.1, 0.05])
    activity = np.zeros(len(days), dtype=int)
    activity[100:280] = 1
    return values, activity


def repair(case, noise, activity):
    fits = fit_activity_endpoints(case, noise, activity)
    observed = np.asarray([row is not None for row in case.displacement_mm])
    return apply_endpoint_fits(activity, observed, fits, "selective")


def test_raw_transition_refines_both_visual_endpoints_without_deleting_motion():
    values, initial = moving_case()
    case = SimpleNamespace(displacement_mm=values.tolist())
    repaired, evidence = repair(case, np.ones(3), initial)
    active = np.flatnonzero(repaired == 1)
    assert abs(active[0] - 120) <= 2
    assert abs(active[-1] + 1 - 260) <= 2
    assert np.all(repaired[122:258] == 1)
    assert all(row["accepted"] for row in evidence)
    assert len(event_masks(repaired)) == 1


def test_stationary_drift_and_single_step_do_not_justify_endpoint_repair():
    days = np.arange(420)
    _, activity = moving_case()
    for displacement in (
        np.zeros(len(days)),
        days * 0.1,
        (days >= 120).astype(float) * 30,
    ):
        case = SimpleNamespace(
            displacement_mm=(displacement[:, None] * np.array([1, -1, 0.5])).tolist()
        )
        repaired, evidence = repair(case, np.ones(3), activity)
        np.testing.assert_array_equal(repaired, activity)
        assert not any(row["accepted"] for row in evidence)


def test_gaps_unknown_dates_and_record_censoring_are_preserved():
    values, activity = moving_case()
    rows = values.tolist()
    rows[170:182] = [None] * 12
    activity[170:182] = -1
    activity[:4] = -1
    case = SimpleNamespace(displacement_mm=rows)
    repaired, evidence = repair(case, np.ones(3), activity)
    np.testing.assert_array_equal(repaired[170:182], activity[170:182])
    np.testing.assert_array_equal(repaired[:4], activity[:4])
    assert len(event_masks(repaired)) == 2
    assert evidence[1]["status"] == "unavailable"
    assert not evidence[1]["accepted"]
    assert not evidence[2]["accepted"]
    full = np.ones(len(rows), dtype=int)
    full[170:182] = -1
    retained, diagnostics = repair(case, np.ones(3), full)
    np.testing.assert_array_equal(retained, full)
    assert diagnostics[0]["reason"] == "Record-censored endpoint"
    assert diagnostics[-1]["reason"] == "Record-censored endpoint"


def test_axis_reflection_and_translation_do_not_change_repaired_dates():
    values, activity = moving_case()
    original, _ = repair(SimpleNamespace(displacement_mm=values.tolist()), np.ones(3), activity)
    transformed, _ = repair(
        SimpleNamespace(displacement_mm=(-values + 1000).tolist()), np.ones(3), activity
    )
    np.testing.assert_array_equal(transformed, original)


def test_adjacent_repairs_keep_a_known_separator_and_do_not_cross_long_abstention():
    days = np.arange(600)
    displacement = np.clip(days - 120, 0, 140) + np.clip(days - 320, 0, 140)
    case = SimpleNamespace(
        displacement_mm=(displacement[:, None] * np.array([1, -1, 0.5])).tolist()
    )
    activity = np.zeros(len(days), dtype=int)
    activity[100:280] = 1
    activity[300:480] = 1
    fits = fit_activity_endpoints(case, np.ones(3), activity)
    observed = np.ones(len(days), dtype=bool)
    repaired, _ = apply_endpoint_fits(activity, observed, fits, "unconditional_fit")
    assert repaired[290] == 0
    assert len(event_masks(repaired)) == 2

    activity[70:82] = -1
    fits = fit_activity_endpoints(case, np.ones(3), activity)
    repaired, _ = apply_endpoint_fits(activity, observed, fits, "unconditional_fit")
    np.testing.assert_array_equal(repaired[70:82], activity[70:82])
    assert np.all(repaired[:70] == 0)
    assert len(event_masks(repaired)) == 2
