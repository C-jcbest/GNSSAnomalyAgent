from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_proposal_utility import apply_proposal, proposal_features


def test_proposal_abstention_loses_expected_correctness_and_missing_has_no_authority():
    features = SimpleNamespace(
        observed=np.array([True, True, False]),
        progressive_rate=np.array([0., .1, np.nan]),
        signal_to_noise=np.array([0., 3., np.nan]),
    )
    evidence = {"rate": features.progressive_rate, "signal_to_noise": features.signal_to_noise}
    initial = np.array([0, 0, -1])
    reviewed = np.array([-1, 1, -1])
    permission = features.observed.copy()
    row, changed = proposal_features(
        initial, reviewed, permission, features, evidence, np.array([.1, .8, .5]), ReviewPolicy()
    )
    np.testing.assert_array_equal(changed, [True, True, False])
    np.testing.assert_allclose(row, [1., 0., .5, 0., -.15])


def test_rejected_proposal_preserves_initial_and_accepted_proposal_respects_scope():
    initial = np.array([0, 1, -1, 0])
    reviewed = np.array([1, 0, -1, 1])
    permission = np.array([True, False, False, True])
    np.testing.assert_array_equal(apply_proposal(initial, reviewed, permission, 0), initial)
    np.testing.assert_array_equal(apply_proposal(initial, reviewed, permission, .1), [1, 1, -1, 1])
    np.testing.assert_array_equal(initial, [0, 1, -1, 0])
