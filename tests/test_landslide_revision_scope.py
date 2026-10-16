"""Revision authority preserves decisions and treats missing/failed evidence honestly."""
from types import SimpleNamespace

import numpy as np
import pytest
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes

from gnss_sim.landslide_adaptive_review import ReviewPolicy


def test_supported_agreement_stays_outside_revision_authority():
    observed = np.ones(200, dtype=bool)
    features = SimpleNamespace(observed=observed, progressive_rate=np.full(200, .1),
                               signal_to_noise=np.full(200, 3))
    long = {"rate": np.full(200, .1), "signal_to_noise": np.full(200, 3)}
    initial = np.ones(200, dtype=int)
    initial[90:105] = 0
    conflict, neighbourhood = conflict_neighbourhood(initial, features, long, ReviewPolicy())
    assert np.flatnonzero(conflict).tolist() == list(range(90, 105))
    assert np.flatnonzero(neighbourhood).tolist() == list(range(41, 154))
    edited = scoped_activity(initial, np.zeros(200, dtype=int), neighbourhood, observed)
    assert np.all(edited[:41] == 1)
    assert np.all(edited[154:] == 1)
    assert np.all(edited[41:154] == 0)


def test_matched_central_scope_has_equal_observed_budget_and_never_edits_gaps():
    observed = np.ones(120, dtype=bool)
    observed[55:65] = False
    neighbourhood = np.zeros(120, dtype=bool)
    neighbourhood[10:30] = True
    neighbourhood[80:100] = True
    conflict, central, detail = target_scopes(neighbourhood, observed, [{"start": 0, "stop": 120}])
    assert conflict.sum() == central.sum() == 40
    assert not np.any(central & ~observed)
    assert detail[0]["editable_observed_days"] == 40
    assert np.all(central[35:55]) and np.all(central[65:85])


def test_failed_local_output_and_initial_unknown_are_not_restored_by_scope():
    observed = np.ones(40, dtype=bool)
    observed[20] = False
    initial = np.ones(40, dtype=int)
    initial[~observed] = -1
    reviewed = np.full(40, -1)
    scope = np.zeros(40, dtype=bool)
    scope[10:30] = True
    scope &= observed
    result = scoped_activity(initial, reviewed, scope, observed)
    assert np.all(result[10:30] == -1)
    assert np.all(result[:10] == 1)
    failed_initial = np.full(40, -1)
    successful_local = np.where(observed, 1, -1)
    result = scoped_activity(failed_initial, successful_local, scope, observed)
    assert np.all(result[:10] == -1)
    assert np.all(result[10:20] == 1)


def test_empty_or_overlapping_targets_do_not_create_revision_authority():
    observed = np.ones(60, dtype=bool)
    mask = np.zeros(60, dtype=bool)
    conflict, central, details = target_scopes(mask, observed, [])
    assert not conflict.any() and not central.any() and details == []
    with pytest.raises(ValueError, match="disjoint"):
        target_scopes(mask, observed, [{"start": 0, "stop": 40}, {"start": 20, "stop": 60}])
    with pytest.raises(ValueError, match="Missing"):
        scoped_activity(np.ones(60, dtype=int), np.ones(60, dtype=int), mask, np.zeros(60, dtype=bool))
