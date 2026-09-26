import json
from datetime import date, timedelta

import numpy as np

from gnss_sim.numerical import (
    _center_scores,
    matrix_profile,
    pelt_points,
    score_to_points,
    score_to_ranges,
    spectral_residual,
    theilsen_score,
)
from gnss_sim.numerical_runner import calibrate_normal, freeze_selection
from gnss_sim.schemas import CaseInput, PointResult, RangeResult


def _normal(case_id: str) -> CaseInput:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(365)]
    zeros = [(0.0, 0.0, 0.0)] * 365
    return CaseInput(case_id=case_id, dates=days, reference_coordinate_mm=(0, 0, 0),
                     observed_coordinate_mm=zeros, displacement_mm=zeros,
                     horizontal_offset_mm=[0.0] * 365, spatial_offset_mm=[0.0] * 365)


def test_point_dates_and_pelt_boundary_are_exact():
    step = np.r_[np.zeros(120), np.full(245, 50.0)]
    assert pelt_points(step, 1) == [120]
    scores = np.zeros(365)
    scores[[119, 120, 121]] = [0.5, 0.8, 0.5]
    assert score_to_points(scores, 0.5) == [120]
    assert PointResult(case_id="case_0001", method="pelt", status="success",
                       predictions={"N": [120], "E": [], "U": []})


def test_window_center_and_closed_ranges():
    centered = _center_scores(np.arange(336, dtype=float))
    assert centered.shape == (365,)
    assert np.all(centered[:15] == 0) and np.all(centered[351:] == 0)
    assert centered[15] == 0 and centered[350] == 335
    active = np.zeros(365)
    active[0:3] = 1
    active[5:8] = 1
    active[364] = 1
    assert score_to_ranges(active, 0.5) == [(0, 2), (5, 7), (364, 364)]
    assert RangeResult(case_id="case_0001", method="theilsen", status="success",
                       predictions={"N": [(0, 2)], "E": [], "U": []})


def test_scores_are_finite_and_deterministic():
    days = np.arange(365)
    series = np.sin(days / 19) + np.random.default_rng(42).normal(0, 0.1, 365)
    for scorer in (spectral_residual, matrix_profile, theilsen_score):
        first = scorer(series)
        assert first.shape == (365,) and np.isfinite(first).all()
        np.testing.assert_array_equal(first, scorer(series))
    np.testing.assert_array_equal(spectral_residual(np.zeros(365)), np.zeros(365))


def test_normal_only_calibration_and_fixed_pelt_candidates():
    normals = [_normal(f"normal_{i:02d}") for i in range(30)]
    thresholds, alarms = calibrate_normal("sr", normals)
    assert thresholds == {"threshold": {"N": 0.0, "E": 0.0, "U": 0.0}}
    assert alarms == {"N": 0, "E": 0, "U": 0}
    pelt, alarms = calibrate_normal("pelt", normals)
    assert pelt["beta"] == {"N": 1, "E": 1, "U": 1}
    assert alarms == {"N": 0, "E": 0, "U": 0}


def test_freeze_selects_each_task_by_its_own_metric(tmp_path):
    metrics = {"sr": ("point", 0.4, 0.02), "pelt": ("point", 0.5, 0.03),
               "matrix-profile": ("range", 0.7, 0.01),
               "theilsen": ("range", 0.8, 0.04)}
    for method, (task, f1, far) in metrics.items():
        directory = tmp_path / method
        directory.mkdir()
        report = {"task": task, "far": far, "execution_success_rate": 1.0}
        report.update({"precision": f1, "recall": f1, "f1": f1} if task == "point"
                      else {"affiliation_precision": f1, "affiliation_recall": f1,
                            "affiliation_f1": f1})
        run = {"task": task, "pilot_sha256": {"manifest": "fixed"},
               "source_sha256": {"numerical.py": "fixed"}, "failed_cases": 0,
               "parameters": {"chosen": method}, "runtime_seconds": 1.0}
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    frozen = freeze_selection(tmp_path)
    assert frozen["selected"]["point"]["method"] == "pelt"
    assert frozen["selected"]["range"]["method"] == "theilsen"
    assert "matrix-profile" in (tmp_path / "range-table.md").read_text(encoding="utf-8")
