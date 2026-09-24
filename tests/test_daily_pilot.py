import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.contracts import Prediction, Window
from gnss_anomaly.daily_pilot import inject_daily
from gnss_anomaly.datasets import build_dataset, demo_backgrounds
from gnss_anomaly.detectors import detect, quality
from gnss_anomaly.experiments import run_experiment


def test_daily_injection_keeps_native_gaps_and_sixty_day_growth(config):
    x = np.zeros((180, 3))
    x[90:93] = np.nan
    w = Window(
        case_id="daily",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=180, freq="D")),
        values=[[None if np.isnan(v) else v for v in row] for row in x],
    )
    for kind in ("spike", "step", "drift", "acceleration", "variance", "normal"):
        y, truth, p = inject_daily(w, kind, np.random.default_rng(1))
        assert np.array_equal(np.isnan(y), np.isnan(x))
        if kind in ("drift", "acceleration"):
            assert p["duration_days"] == 60
            end = p["start"] + 59
            assert y[end, p["axis"]] == pytest.approx(p["amplitude_mm"] * p["sign"])
            assert y[-1, p["axis"]] == y[end, p["axis"]]
            assert truth[p["start"] :, p["axis"]].all()
        if kind == "normal":
            assert not truth.any()
    assert quality(w)["max_gap_days"] == 3
    with pytest.raises(ValueError, match="sampling interval"):
        detect(w, "hampel", config["detectors"])


def test_parallel_run_uses_independent_model_call_logs(tmp_path, config, monkeypatch):
    import gnss_anomaly.experiments as experiments

    clients = []

    class FakeModel:
        def __init__(self, config):
            self.calls = []
            self.cost = 0
            self.closed = False
            clients.append(self)

        def reset(self):
            self.calls = []

        def ask(self, prompt, schema, images=None):
            self.calls.append({"status": "ok", "images": len(images or [])})
            return Prediction()

        def close(self):
            self.closed = True

    monkeypatch.setattr(experiments, "ModelClient", FakeModel)
    demo_backgrounds(tmp_path / "backgrounds")
    build_dataset(tmp_path / "backgrounds", tmp_path / "dataset", 15)
    config.update(workers=2, repeats=1)
    summary = run_experiment(
        tmp_path / "dataset", tmp_path / "run", config, "development", ["visual", "fixed"], limit=2
    )
    assert summary["records_present"] == 4 and summary["all_records_present"]
    assert len(clients) == 2 and all(c.closed for c in clients)
    assert all(r["mean_model_requests"] == 1 for r in summary["by_repeat"])
