"""Longer evidence never combines incompatible scales or expands all old conflicts."""
from types import SimpleNamespace

import numpy as np
from probe_landslide_longer_motion_evidence import (
    longer_conflict_scope,
    merged_long_evidence,
    scale_positive,
)

from gnss_sim.landslide_adaptive_review import ReviewPolicy


def evidence(rate, noise, direction):
    return {"rate": np.asarray(rate, dtype=float), "signal_to_noise": np.asarray(noise, dtype=float),
            "direction": np.asarray(direction, dtype=float)}


def test_rate_and_noise_from_different_scales_cannot_create_movement():
    base = evidence([.05], [1], [1])
    extra = evidence([.02], [3], [1])
    merged = merged_long_evidence(base, extra, ReviewPolicy())
    assert not scale_positive(merged, ReviewPolicy()).any()
    assert merged["rate"].tolist() == [.05]
    assert merged["signal_to_noise"].tolist() == [1]


def test_existing_positive_retains_its_evidence_and_missing_support_can_be_added():
    base = evidence([.05, np.nan], [3, np.nan], [.5, np.nan])
    extra = evidence([.07, .01], [4, 1], [.9, .6])
    merged = merged_long_evidence(base, extra, ReviewPolicy())
    assert merged["rate"].tolist() == [.05, .01]
    assert merged["signal_to_noise"].tolist() == [3, 1]
    assert merged["direction"].tolist() == [.5, .6]


def test_only_new_conflicts_use_longer_physical_radius_and_missing_dates_stay_outside():
    observed = np.ones(400, dtype=bool)
    observed[200] = False
    rates = np.zeros(400)
    rates[40] = .1
    features = SimpleNamespace(observed=observed, progressive_rate=rates,
                               signal_to_noise=np.full(400, 3))
    base = evidence(np.zeros(400), np.ones(400), np.ones(400))
    extra = evidence(np.zeros(400), np.ones(400), np.ones(400))
    extra["rate"][220] = .05
    extra["signal_to_noise"][220] = 3
    _, conflicts, scope, additional = longer_conflict_scope(
        np.zeros(400, dtype=int), features, base, extra, ReviewPolicy())
    assert np.flatnonzero(conflicts).tolist() == [40, 220]
    assert np.flatnonzero(additional).tolist() == [220]
    assert np.all(scope[:90])
    assert not scope[90:129].any()
    assert np.all(scope[129:200]) and np.all(scope[201:312])
    assert not scope[200] and not scope[312:].any()


def test_new_support_can_remove_an_unsupported_activity_conflict_without_declaring_a_label():
    observed = np.ones(200, dtype=bool)
    features = SimpleNamespace(observed=observed, progressive_rate=np.zeros(200),
                               signal_to_noise=np.ones(200))
    base = evidence(np.zeros(200), np.ones(200), np.ones(200))
    extra = evidence(np.zeros(200), np.ones(200), np.ones(200))
    extra["rate"][100] = .05
    extra["signal_to_noise"][100] = 3
    initial = np.zeros(200, dtype=int)
    initial[100] = 1
    _, conflicts, scope, additional = longer_conflict_scope(initial, features, base, extra, ReviewPolicy())
    assert not conflicts.any() and not scope.any() and not additional.any()
    assert initial[100] == 1
