"""Check controlled-reference meaning and matched-event boundary reporting."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_landslide_adaptive_review import boundary_errors  # noqa: E402
from prepare_landslide_motion_probe import controlled_case  # noqa: E402

from gnss_sim.landslide import LandslideRequest, simulate_case  # noqa: E402


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5])
def test_probe_labels_follow_known_motion_and_preserve_original_gaps(index):
    request = LandslideRequest(seed=2026100937, count=6, days=730)
    case, rows, reference = controlled_case(request, index)
    original, _ = simulate_case(request, index, "stable")
    speed = np.array(reference["speed_mm_day"])
    latent = np.array(reference["latent_displacement_mm"])
    np.testing.assert_allclose(np.linalg.norm(np.diff(latent, axis=0), axis=1), speed[1:],
                               atol=1e-12)
    for day, row in enumerate(rows):
        assert (case.displacement_mm[day] is None) == (original.displacement_mm[day] is None)
        if case.displacement_mm[day] is None:
            assert row["activity_label"] == -1 and row["feature"] == ""
        else:
            assert row["activity_label"] == int(speed[day] > 0)
            assert bool(row["feature"]) == bool(speed[day] > 0)
    if index == 1:
        assert reference["instrument_step_day"] is not None
        assert all(row["activity_label"] != 1 for row in rows)
    if index == 2:
        active_rows = [row for row in rows if row["activity_label"] == 1]
        assert active_rows and all(row["weak_signal"] == 1 for row in active_rows)
        assert all(row["feature"] == "steady_motion" for row in active_rows)


def test_boundary_errors_do_not_hide_missed_events_and_report_signed_offsets():
    rows = [{"activity_label": 0, "episode_id": ""} for _ in range(50)]
    for day in range(10, 20):
        rows[day] = {"activity_label": 1, "episode_id": "first"}
    for day in range(30, 40):
        rows[day] = {"activity_label": 1, "episode_id": "second"}
    prediction = np.zeros(50, dtype=int)
    prediction[8:22] = 1
    prediction[45:49] = 1
    result = boundary_errors(rows, prediction, .3)
    assert result["missed_events"] == 1
    assert result["unmatched_prediction_events"] == 1
    pair, = result["matched_pairs"]
    assert pair["onset_error_days"] == -2
    assert pair["offset_error_days"] == 2
