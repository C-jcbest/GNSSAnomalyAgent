from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_scale_review import compressed_review_targets, historical_scale_gains


def test_large_history_can_hide_local_motion_but_offsets_cannot_create_gain():
    values = np.column_stack([np.r_[np.linspace(0, 100, 30), np.linspace(100, 103, 30)],
                              np.zeros(60), np.zeros(60)])
    case = SimpleNamespace(displacement_mm=values.tolist())
    gain = historical_scale_gains(case, np.ones(3), [(30, 60)])[0]
    shifted = SimpleNamespace(displacement_mm=(values + np.array([800, -90, 700])).tolist())
    shifted_gain = historical_scale_gains(shifted, np.ones(3), [(30, 60)])[0]
    assert gain["gain"] == pytest.approx(10.3)
    assert shifted_gain["axis_gains"] == pytest.approx(gain["axis_gains"])


def test_quiet_signal_does_not_qualify_due_to_the_global_display_floor():
    values = np.column_stack([np.sin(np.arange(60)) * 2, np.zeros(60), np.zeros(60)])
    gain = historical_scale_gains(SimpleNamespace(displacement_mm=values.tolist()),
                                  np.ones(3), [(30, 60)])[0]
    assert gain["gain"] < 1


def test_scale_gain_and_missed_motion_are_both_required():
    targets = [{"index": 0, "missed_motion_days": 14},
               {"index": 1, "missed_motion_days": 13},
               {"index": 2, "missed_motion_days": 30}]
    gains = [{"gain": 4.0}, {"gain": 10.0}, {"gain": 3.99}]
    assert compressed_review_targets(targets, gains) == [targets[0]]


def test_missing_observations_do_not_create_scale_or_move_the_calendar():
    case = SimpleNamespace(displacement_mm=[[1, 2, 3], None, None, [2, 3, 4]])
    gains = historical_scale_gains(case, np.ones(3), [(0, 2), (1, 3)])
    assert [row["observed_days"] for row in gains] == [1, 0]
    assert gains[1]["gain"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="calendar"):
        historical_scale_gains(case, np.ones(3), [(1, 5)])
