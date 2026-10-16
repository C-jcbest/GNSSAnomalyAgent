"""Auxiliary quality supervision retains original event/calendar semantics."""

import numpy as np

from gnss_sim.landslide_deformation_quality import query_quality_targets


def test_quality_is_query_local_and_does_not_extend_through_the_platform():
    rows = [{"quality_event_label": 0} for _ in range(80)]
    rows[30]["quality_event_label"] = 1
    targets = query_quality_targets(rows)
    assert np.all(targets[24:38] == 1)
    assert np.all(targets[38:72] == 0)


def test_hidden_quality_event_cannot_become_an_observed_positive():
    rows = [{"quality_event_label": 0} for _ in range(80)]
    for day in range(20, 40):
        rows[day]["quality_event_label"] = -1
    targets = query_quality_targets(rows)
    assert targets[30] == -1
    assert targets[10] == targets[60] == 0
