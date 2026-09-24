from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gnss_anomaly.contracts import Event, Prediction, Window
from gnss_anomaly.paired_background import matched_control, response_counts


def window():
    return Window(
        case_id="source",
        sampling_hours=24,
        timestamps=[
            datetime(2026, 1, 1, 7, tzinfo=timezone.utc) + timedelta(days=i) for i in range(180)
        ],
        values=[(float(i), float(-i), 0.0) for i in range(180)],
    )


def prediction(a, b):
    return Prediction(events=[Event(start=a, end=b, channels=["N"])])


def test_control_restores_source_without_filling_or_mutating():
    source = window()
    injected = source.model_copy(deep=True)
    injected.values[0] = (None, -100.0, 0.0)
    injected.values[12] = (900.0, -12.0, 0.0)
    control = matched_control(source, injected)
    assert control.values[0] == (None, 0.0, 0.0)
    assert control.values[12] == source.values[12]
    assert injected.values[12][0] == 900.0
    assert source.values[0][0] == 0.0
    assert np.array_equal(np.isfinite(control.array()), np.isfinite(injected.array()))
    assert set(control.model_dump()) == {"case_id", "sampling_hours", "timestamps", "values"}


def test_control_rejects_invented_observation():
    source = window()
    injected = source.model_copy(deep=True)
    source.values[2] = (None, -2.0, 0.0)
    with pytest.raises(ValueError, match="absent"):
        matched_control(source, injected)


def test_response_partitions_and_preserves_target_denominator():
    w = window()
    result = response_counts(w, w, prediction(10, 14), prediction(12, 16), prediction(10, 12))
    assert result["target_cells"] == 5
    assert result["new_target"] == 2
    assert result["inherited_target"] == 1
    assert result["lost_target"] == 2
    assert result["injected_target"] == result["new_target"] + result["inherited_target"]
    assert result["new_outside"] == 2


def test_failed_control_is_not_silent_empty_prediction():
    w = window()
    result = response_counts(
        w, w, prediction(10, 14), prediction(10, 14), Prediction(status="error")
    )
    assert result == {"completed": False, "expected_target_cells": 5}


def test_response_rejects_mismatched_mask():
    w = window()
    control = w.model_copy(deep=True)
    control.values[10] = (None, None, None)
    with pytest.raises(ValueError, match="mask"):
        response_counts(w, control, prediction(10, 14), Prediction(), Prediction())
