"""Observed range contract, empty detections, missing gaps and negative controls."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp20_range_metrics import affiliation, aggregate, evaluate  # noqa: E402


def event(start, end, kind="transient"):
    return {"start": start, "end": end, "kind": kind, "status": "confirmed", "evidence": "test", "phases": []}


def record(events, predictions, missing=()):
    observation = {"dates": [str(i) for i in range(60)], "background_end": 9,
                   "displacement_mm": [None if i in missing else [0, 0, 0] for i in range(60)]}
    return evaluate(observation, {"events": events}, {"prediction": {"events": predictions}})


def test_author_readme_example():
    target = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    predicted = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    report = affiliation(target, predicted, 0)
    assert round(report["precision"], 2) == .82
    assert round(report["recall"], 2) == .84


def test_empty_prediction_is_not_discarded_and_normal_not_scored_as_perfect():
    rows = record([event(20, 30), event(35, 45, "deformation")], [])
    assert rows["transient"]["affiliation"] == {
        "precision": None, "recall": 0, "f1": 0, "empty_prediction": True}
    assert rows["transient"]["truth_best_ious"] == [0]
    normal = record([], [event(20, 30)])
    assert normal["transient"]["affiliation"]["f1"] is None
    summary = aggregate([{"metrics": rows}, {"metrics": normal}])["transient"]
    assert summary["affiliation"]["positive_stations"] == 1
    assert summary["affiliation"]["station_macro_f1"] == 0
    assert summary["negative_station_fp_days"] == 11


def test_missing_keeps_semantic_interval_and_original_date_distance():
    rows = record([event(20, 30)], [event(20, 30)], missing=(25,))
    r = rows["transient"]
    assert r["truth_events"] == 1
    assert r["truth_observed_fragments"] == 2
    assert r["daily"]["tp"] == 10
    assert r["events"]["iou_0.5"]["tp"] == 1
    assert r["affiliation"]["f1"] == 1
    assert r["affiliation_gap_spans_sensitivity"]["f1"] == 1
    target = np.zeros(60, dtype=bool)
    predicted = target.copy()
    target[20:31] = True
    predicted[40:51] = True
    assert affiliation(target, predicted, 10)["f1"] < 1


def test_event_boundaries_weighted_by_number_matched():
    a = record([event(20, 30), event(40, 50)], [event(21, 30), event(41, 50)])
    b = record([event(20, 30)], [event(23, 30)])
    report = aggregate([{"metrics": a}, {"metrics": b}])
    assert report["transient"]["events_micro"]["iou_0.5"]["matched_start_mae_days"] == pytest.approx(5 / 3)
