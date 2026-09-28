"""Independent checks for post-hoc date matching and frozen-output replay."""

import runpy
from itertools import combinations
from pathlib import Path

from gnss_sim.schemas import PointResult, RangeResult

analysis = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/analyze_p6.py"))


def brute_match(actual, predicted, tolerance):
    if not actual:
        return 0
    head, *tail = actual
    return max([brute_match(tail, predicted, tolerance)] + [
        1 + brute_match(tail, predicted - {day}, tolerance)
        for day in predicted if abs(day - head) <= tolerance])


def test_tolerant_matching_against_exhaustive_assignment():
    subsets = [set(s) for size in range(6) for s in combinations(range(5), size)]
    for actual in subsets:
        for predicted in subsets:
            for tolerance in (0, 1, 3):
                assert analysis["matched_count"](actual, predicted, tolerance) == brute_match(
                    sorted(actual), predicted, tolerance)


def test_union_does_not_use_payload_from_failed_visual_result():
    num = PointResult(case_id="case", method="sr", status="success",
                      predictions={"N": [10, 10], "E": [], "U": []})
    vis = PointResult(case_id="case", method="visual", status="failed",
                      predictions={"N": [20], "E": [], "U": []})
    row = analysis["union_results"]({"case": num}, {"case": vis}, "point")["case"]
    assert row.predictions.N == [10]
    assert num.predictions.N == [10, 10]
    assert vis.status == "failed"


def test_range_replay_measures_inclusive_union_not_interval_list_length():
    num = RangeResult(case_id="case", method="theilsen", status="success",
                      predictions={"N": [(10, 20)], "E": [], "U": []})
    vis = RangeResult(case_id="case", method="visual", status="success",
                      predictions={"N": [(15, 25)], "E": [], "U": []})
    row = analysis["union_results"]({"case": num}, {"case": vis}, "range")["case"]
    assert analysis["range_days"](row.predictions.N) == set(range(10, 26))
