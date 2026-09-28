import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "p7b_errors", Path(__file__).resolve().parents[1] / "scripts/analyze_p7b_errors.py")
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)


def test_overlap_loss_counted_once_and_redundant_deletion_not_loss():
    gt = set(range(5, 11))
    before = set(range(3, 13))
    after = set(range(7, 13))
    parts = errors.partition_days(gt, before, after)
    assert parts == {"lost_tp": {5, 6}, "removed_fp": {3, 4},
                     "kept_tp": {7, 8, 9, 10}, "kept_fp": {11, 12}}
    assert errors.partition_days(gt, before, before)["lost_tp"] == set()
    with pytest.raises(ValueError):
        errors.partition_days(gt, before, after | {20})


def test_original_covering_source_is_disjoint_day_partition():
    candidates = [{"axis": "N", "start": 3, "end": 9, "sources": ["N"]},
                  {"axis": "N", "start": 7, "end": 12, "sources": ["V"]},
                  {"axis": "N", "start": 7, "end": 10, "sources": ["N"]},
                  {"axis": "E", "start": 3, "end": 20, "sources": ["V"]}]
    assert errors.day_sources(candidates, "N", 5) == "N"
    assert errors.day_sources(candidates, "N", 8) == "N+V"
    assert errors.day_sources(candidates, "N", 11) == "V"


def test_inclusive_thirds_and_invalid_day():
    phases = [errors.phase(100, 189, day) for day in range(100, 190)]
    assert [phases.count(k) for k in ("early", "middle", "late")] == [30, 30, 30]
    assert errors.phase(100, 100, 100) == "early"
    with pytest.raises(ValueError):
        errors.phase(100, 189, 190)


def test_point_source_counts_reconcile_without_assigning_type_to_false_positive():
    rows = [{"sources": "N", "kept": False, "exact_tp": True},
            {"sources": "N", "kept": False, "exact_tp": False},
            {"sources": "N+V", "kept": True, "exact_tp": True}]
    groups = errors.group_rows(rows, "sources")
    assert groups["N"] == {"candidates": 2, "kept": 0, "exact_tp": 1,
                           "lost_tp": 1, "removed_fp": 1}
    assert groups["N+V"]["lost_tp"] == 0
