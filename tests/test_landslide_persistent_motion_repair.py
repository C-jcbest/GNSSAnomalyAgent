import numpy as np

from gnss_sim.landslide_persistent_motion_repair import repair_persistent_motion


def test_isolated_agreement_cannot_extend_existing_activity():
    initial = np.r_[np.ones(20, dtype=int), np.zeros(13, dtype=int)]
    result = repair_persistent_motion(initial, np.ones(33, dtype=int),
                                     np.ones(33, dtype=bool), np.ones(33, dtype=bool))
    assert np.all(result[:20] == 1)
    assert np.all(result[20:] == -1)


def test_fourteen_observed_agreements_form_a_new_activity_interval():
    initial = np.zeros(17, dtype=int)
    reviewed = np.r_[0, np.ones(14, dtype=int), 0, 0]
    result = repair_persistent_motion(initial, reviewed, np.ones(17, dtype=bool),
                                     np.ones(17, dtype=bool))
    assert result.tolist() == [-1] + [1] * 14 + [-1, -1]


def test_missing_observations_split_new_support_and_remain_unknown():
    observed = np.ones(20, dtype=bool)
    observed[10] = False
    initial = np.zeros(20, dtype=int)
    initial[10] = -1
    reviewed = np.ones(20, dtype=int)
    reviewed[10] = -1
    result = repair_persistent_motion(initial, reviewed, observed.copy(), observed)
    assert np.all(result == -1)
