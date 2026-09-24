import numpy as np
import pytest

from gnss_anomaly.contracts import Event, Prediction
from gnss_anomaly.detectors import detect, fill_short_gaps
from gnss_anomaly.evaluation import aggregate, counts


def test_only_bounded_short_gaps_interpolate():
    x = np.array([np.nan, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, 8, np.nan])
    filled = fill_short_gaps(x)
    np.testing.assert_allclose(filled[1:5], [1, 2, 3, 4])
    assert np.isnan(filled[[0, 5, 6, 7, 9]]).all()


@pytest.mark.parametrize("name", ["hampel", "cusum", "iforest"])
def test_long_gap_does_not_create_jump_and_missing_never_flagged(name, window, config):
    values = np.zeros((60, 3))
    values[20:30] = np.nan
    values[30:] = 1000
    result = detect(window(values), name, config["detectors"])
    assert result.status == "ok" and not result.events


def test_spike_survives_random_short_gaps(window, config):
    values = np.zeros((80, 3))
    values[::4] = np.nan
    values[41, 0] = 20
    result = detect(window(values), "hampel", config["detectors"])
    assert result.status == "ok" and result.mask(80)[41, 0]
    assert not result.mask(80)[::4].any()


def test_no_point_adjustment_or_missing_credit(window):
    x = np.zeros((48, 3))
    x[11:14] = np.nan
    truth = Prediction(events=[Event(start=10, end=20, channels=["N"])])
    guess = Prediction(events=[Event(start=10, end=13, channels=["N"])])
    metric = counts(window(x), truth, guess)
    assert (metric["tp"], metric["fp"], metric["fn"]) == (1, 0, 7)
    assert metric["unobserved_positive_cells"] == 3
    hidden = Prediction(events=[Event(start=11, end=13, channels=["N"])])
    assert counts(window(x), hidden, Prediction())["hidden_events"] == 1


def test_failed_normal_is_not_a_correct_negative(window):
    metric = counts(window(), Prediction(), Prediction(status="error"))
    assert metric["tn"] == 0
    rows = [{"metrics": metric, "seconds": 1, "model_requests": 1, "tools": [], "cost": None}]
    result = aggregate(rows)
    assert result["normal_failures"] == 1
    assert result["normal_false_alarm_rate_completed"] is None
    assert result["cost"] is None
