"""Original date permissions and unknowns in finite exemplar reviews."""

import numpy as np
import pytest
from probe_landslide_contrastive_exemplars import apply_labels, parse_labels

from gnss_sim.landslide_local_experiment import learned_stages


@pytest.mark.parametrize("payload", [
    {"labels": [0] * 13},
    {"labels": [0] * 15},
    {"labels": [False] * 14},
    {"labels": [0.] * 14},
    {"labels": [2] * 14},
    {"labels": [0] * 14, "reason": "extra"},
])
def test_malformed_day_vectors_cannot_change_a_prediction(payload):
    with pytest.raises(ValueError):
        parse_labels(payload)


def test_review_never_fills_initial_unknowns_or_changes_dates_outside_query():
    activity = np.zeros(30, dtype=int)
    activity[12] = -1
    observed = np.ones(30, dtype=bool)
    observed[12] = False
    support = observed.copy()
    stage_support = observed.copy()
    stage_support[13] = False
    codes = np.ones(30, dtype=int)
    baseline = learned_stages(codes, activity, stage_support).to_dict()
    packet = {"query": [7, 21], "support": support.tolist(), "observed": observed.tolist(),
              "stage_support": stage_support.tolist(), "stage_codes": codes.tolist()}
    result = apply_labels(packet, baseline, np.ones(14, dtype=int))
    assert result["activity"][:7] == baseline["activity"][:7]
    assert result["activity"][21:] == baseline["activity"][21:]
    assert result["activity"][12] == result["stage"][12] == -1
    assert result["activity"][13] == 1 and result["stage"][13] == -1
    assert result["stage"][7:12] == [1] * 5
    assert baseline["activity"][7:12] == [0] * 5
