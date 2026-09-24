from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.contracts import Event, Prediction, Window
from gnss_anomaly.detectors import detect
from gnss_anomaly.evaluation import counts
from gnss_anomaly.experiments import run_experiment, validate_config
from gnss_anomaly.policies import Choice, Executor
from gnss_anomaly.storage import read_json


@pytest.fixture
def trend_config():
    return read_json(Path(__file__).parents[1] / "configs/daily-trend-pilot.json")


def daily(x):
    x = np.asarray(x)
    return Window(
        case_id="anonymous",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=len(x), freq="D")),
        values=[[None if not np.isfinite(v) else float(v) for v in row] for row in x],
    )


def detect_x(x, config):
    diagnostic = {}
    w = daily(x)
    before = w.model_dump_json()
    p = detect(w, "reference_trend", config["detectors"], diagnostics=diagnostic)
    assert before == w.model_dump_json()
    return p, diagnostic


def test_uniform_motion_and_reference_outlier_are_not_new_change(trend_config):
    x = np.tile(np.arange(180)[:, None] * 0.3, (1, 3))
    x[10, 0] += 100
    x[::5, 1] = np.nan
    p, d = detect_x(x, trend_config)
    assert p.status == "ok" and not p.events
    assert all(f["slope_mm_per_day"] == pytest.approx(0.3) for f in d["reference_fits"])
    assert not np.array(d["scorable_mask"])[:45].any()


def test_sustained_departure_persists_but_recovery_stops(trend_config):
    x = np.zeros((180, 3))
    x[70:130, 0] = 10
    x[70, 1] = 100  # An isolated excursion must not become a long event.
    p, d = detect_x(x, trend_config)
    mask = p.mask(180)
    assert mask[70:130, 0].all()
    assert not mask[:70].any() and not mask[130:].any() and not mask[:, 1:].any()
    assert np.array(d["scorable_mask"])[45:].all()


def test_rate_change_and_acceleration_not_backfilled_to_onset(trend_config):
    t = np.arange(180)
    x = np.zeros((180, 3))
    x[:, 0] = 0.5 * np.maximum(t - 70, 0)
    x[:, 1] = -0.02 * np.maximum(t - 70, 0) ** 2
    p, _ = detect_x(x, trend_config)
    mask = p.mask(180)
    assert not mask[:75].any()  # Evidence must exceed the frozen residual threshold.
    assert mask[100:, :2].all() and not mask[:, 2].any()


def test_long_gap_requires_new_confirmation_and_no_invented_observations(trend_config):
    x = np.zeros((180, 3))
    x[80:100] = np.nan
    x[100:, 0] = 10
    x[100:108, 1] = 10  # Too short even if a previous episode occurred.
    x[65:80, 1] = 10
    p, d = detect_x(x, trend_config)
    mask = p.mask(180)
    assert not mask[80:100].any() and not np.array(d["scorable_mask"])[80:100].any()
    assert mask[100:, 0].all() and not mask[100:, 1].any()


def test_missing_reference_does_not_search_for_favourable_later_baseline(trend_config):
    x = np.zeros((180, 3))
    x[:25] = np.nan
    p, d = detect_x(x, trend_config)
    assert p.status == "insufficient" and d["scorable_cells"] == 0
    truth = Prediction(events=[Event(start=80, end=100, channels=["N"])])
    assert counts(daily(x), truth, p)["fn"] == 21


def test_reference_phase_positives_still_count_as_misses(trend_config):
    x = np.zeros((180, 3))
    x[10, 0] = 20
    p, _ = detect_x(x, trend_config)
    truth = Prediction(events=[Event(start=10, end=10, channels=["N"])])
    assert counts(daily(x), truth, p)["fn"] == 1


def test_daily_only_configuration_and_agent_tool_isolation(trend_config):
    validate_config(trend_config)
    w = daily(np.zeros((180, 3)))
    executor = Executor(trend_config, None)
    assert executor.run(w, "reference_trend").status == "ok"
    assert executor.tools[0]["diagnostics"]["implementation"] == "reference-trend-v1"
    with pytest.raises(ValueError):
        Choice(tool="reference_trend", reason="not yet a routing tool")
    trend_config["detectors"]["reference_trend"]["threshold"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_config(trend_config)


def test_candidate_cannot_run_validation_or_test(tmp_path, trend_config):
    for split in ["validation", "test"]:
        with pytest.raises(ValueError, match="development-only"):
            run_experiment(
                tmp_path / "absent", tmp_path / "out", trend_config, split, ["reference_trend"]
            )
    assert not (tmp_path / "out").exists()
