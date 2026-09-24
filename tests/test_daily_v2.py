import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.contracts import Event, Prediction, Window, prediction_schema
from gnss_anomaly.daily_detection import daily_cusum, daily_features
from gnss_anomaly.detectors import detect
from gnss_anomaly.diagnostics import runtime_diagnostics
from gnss_anomaly.evaluation import aggregate, counts
from gnss_anomaly.models import ModelClient
from gnss_anomaly.policies import Executor
from gnss_anomaly.storage import read_json


@pytest.fixture
def daily_config():
    result = read_json(Path(__file__).parents[1] / "configs/daily-pilot-v2.json")
    result["model"]["retry_delay_seconds"] = 0
    return result


def daily_window(x):
    return Window(
        case_id="anonymous",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=len(x), freq="D")),
        values=[[float(v) if np.isfinite(v) else None for v in row] for row in x],
    )


@pytest.mark.parametrize("name", ["hampel", "cusum", "iforest"])
def test_fragmented_daily_data_is_used_without_imputation(name, daily_config):
    x = np.zeros((180, 3))
    x[::4] = np.nan  # No complete segment reaches 12 days.
    x[41, 0] = 20
    w = daily_window(x)
    before = w.model_dump_json()
    diagnostic = {}
    prediction = detect(w, name, daily_config["detectors"], diagnostics=diagnostic)
    assert prediction.status == "ok" and diagnostic["numerical_coverage"] > 0.5
    mask = np.array(diagnostic["scorable_mask"])
    assert not mask[::4].any() and not prediction.mask(180)[::4].any()
    assert w.model_dump_json() == before
    if name == "hampel":
        assert prediction.mask(180)[41, 0]


def test_daily_slope_uses_calendar_positions_and_no_gap_difference():
    x = np.arange(60, dtype=float) * 2
    x[::4] = np.nan
    _, _, features = daily_features(x, 15, 8)
    assert np.allclose(features[np.isfinite(features[:, 2]), 2], 40.0)
    assert np.isnan(features[1::4, 1]).all()  # First observed day after missing has no daily delta.


def test_cusum_long_gap_resets_but_does_not_invent_velocity(daily_config):
    x = np.zeros(100)
    x[30:35] = np.arange(1, 6) * 2
    x[35:40] = np.nan
    x[40:] = 1000
    flags, scorable = daily_cusum(x, daily_config["detectors"])
    assert flags[34]
    assert not flags[35:].any()
    assert not scorable[35:41].any() and scorable[41]
    # A short gap preserves prior evidence without creating updates inside/over the gap.
    short = x.copy()
    short[37:] = 1000
    flags, scorable = daily_cusum(short, daily_config["detectors"])
    assert flags[38] and not scorable[35:38].any()


def test_unscorable_positive_still_counts_as_miss(daily_config):
    x = np.full((180, 3), np.nan)
    x[90, 0] = 20
    w = daily_window(x)
    diagnostic = {}
    pred = detect(w, "hampel", daily_config["detectors"], diagnostics=diagnostic)
    truth = Prediction(events=[Event(start=90, end=90, channels=["N"])])
    metric = counts(w, truth, pred)
    assert pred.status == "insufficient" and diagnostic["scorable_cells"] == 0
    assert metric["observed_cells"] == 1 and metric["fn"] == 1


def client(monkeypatch, daily_config, handler):
    monkeypatch.setenv("QWEN_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("QWEN_API_KEY", "not-for-logs")
    return ModelClient(daily_config["model"], httpx.Client(transport=httpx.MockTransport(handler)))


def response(events=None, status="ok"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps({"status": status, "events": events or []})}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def test_bounds_correction_is_bounded_and_logged(monkeypatch, daily_config):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(
            [{"start": 60, "end": 180 if len(requests) == 1 else 179, "channels": ["N"]}]
        )

    model = client(monkeypatch, daily_config, handler)
    pred = model.ask("observations only", prediction_schema(180))
    assert pred.events[0].end == 179
    assert [c["status"] for c in model.calls] == ["error", "ok"]
    assert [c["attempt"] for c in model.calls] == [1, 2]
    assert model.calls[0]["response"] != model.calls[1]["response"]
    assert runtime_diagnostics(pred, [], model.calls)["first_attempt_completed"] is False
    assert "not-for-logs" not in json.dumps(model.calls)
    with pytest.raises(RuntimeError, match="budget"):
        model.ask("more", prediction_schema(180))
    model.close()


@pytest.mark.parametrize(
    "first_code,second_bad,expected_calls", [(401, False, 1), (429, False, 2), (503, True, 2)]
)
def test_network_and_format_share_attempt_limit(
    monkeypatch, daily_config, first_code, second_bad, expected_calls
):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(first_code)
        return response([{"start": 0, "end": 180, "channels": ["N"]}]) if second_bad else response()

    model = client(monkeypatch, daily_config, handler)
    if first_code == 401 or second_bad:
        with pytest.raises(RuntimeError):
            model.ask("same input", prediction_schema(180))
    else:
        assert model.ask("same input", prediction_schema(180)).status == "ok"
    assert len(model.calls) == expected_calls
    model.close()


def test_connect_retry_then_valid_insufficient_is_not_resampled(monkeypatch, daily_config):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("private-url-must-not-appear", request=request)
        return response(status="insufficient")

    model = client(monkeypatch, daily_config, handler)
    pred = model.ask("only data", prediction_schema(180))
    assert pred.status == "insufficient" and len(model.calls) == 2
    assert "private-url" not in json.dumps(model.calls)
    model.close()


def test_visual_receives_exact_missing_metadata_not_labels(daily_config):
    class Capture:
        def ask(self, prompt, schema, images):
            self.prompt = prompt
            self.images = images
            return schema.model_validate({"status": "ok", "events": []})

    x = np.zeros((180, 3))
    x[143:146] = np.nan
    model = Capture()
    assert Executor(daily_config, model).run(daily_window(x), "fixed").status == "ok"
    assert '"N": [[143, 145]]' in model.prompt
    assert '"legal_index": [0, 179]' in model.prompt
    assert "SCWM" not in model.prompt and "六月" not in model.prompt
    assert '"labels"' not in model.prompt and '"injection"' not in model.prompt


def test_diagnostics_do_not_change_metrics(daily_config):
    w = daily_window(np.zeros((180, 3)))
    p = Prediction()
    row = {
        "metrics": counts(w, p, p),
        "seconds": 0,
        "tools": [],
        "model_requests": 2,
        "cost": None,
        "diagnostics": {
            "first_attempt_completed": False,
            "numerical_scorable_cells": 100,
            "numerical_observed_cells": 540,
            "logical_model_calls": 1,
        },
    }
    result = aggregate([row])
    assert result["f1"] is None and result["completion_rate"] == 1
    assert result["first_attempt_completion_rate"] == 0
    assert result["numerical_coverage"] == pytest.approx(100 / 540)
