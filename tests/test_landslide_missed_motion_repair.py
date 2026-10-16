import numpy as np
import pytest

from gnss_sim.landslide_missed_motion_repair import repair_missed_motion


def test_local_visual_confirmation_adds_only_permitted_motion():
    initial = np.array([0, 0, 1, -1, 0])
    reviewed = np.array([1, 1, 0, 1, 1])
    permission = np.array([True, False, True, True, False])
    result = repair_missed_motion(initial, reviewed, permission, np.ones(5, dtype=bool))
    assert result.tolist() == [1, 0, 1, 1, 0]


def test_evidence_without_visual_agreement_retains_unresolved_state():
    initial = np.array([0, -1, 1])
    reviewed = np.array([0, -1, -1])
    result = repair_missed_motion(initial, reviewed, np.ones(3, dtype=bool),
                                  np.ones(3, dtype=bool))
    assert result.tolist() == [-1, -1, 1]


def test_missing_day_cannot_become_activity_or_receive_permission():
    observed = np.array([True, False, True])
    initial = np.array([0, -1, 1])
    reviewed = np.array([1, -1, 0])
    result = repair_missed_motion(initial, reviewed, np.array([True, False, True]), observed)
    assert result.tolist() == [1, -1, 1]
    with pytest.raises(ValueError, match="Missing dates"):
        repair_missed_motion(initial, reviewed, np.ones(3, dtype=bool), observed)
