import importlib.util
from pathlib import Path
from types import SimpleNamespace

from gnss_sim.visual import failed_result, parse_result

spec = importlib.util.spec_from_file_location(
    "evaluate_p8a", Path(__file__).resolve().parents[1] / "scripts/evaluate_p8a.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unique_day_edits_do_not_double_count_overlapping_ranges():
    truths = {"c": SimpleNamespace(events=[SimpleNamespace(
        type="slow_trend", axis="N", start_index=10, end_index=19)])}
    old = {"c": parse_result('{"N":[[8,15],[10,15]],"E":[],"U":[]}', "c", "range")}
    new = {"c": parse_result('{"N":[[12,22],[12,18]],"E":[],"U":[]}', "c", "range")}
    result = module.day_changes(truths, old, new)
    assert result["unique_days"] == dict(added_tp=4, removed_tp=2, added_fp=3, removed_fp=2)
    new["c"] = failed_result("c", "range")
    result = module.day_changes(truths, old, new)
    assert result["paired_success_cases"] == 0
    assert result["after_failed_before_success"] == 1
    assert not any(result["unique_days"].values())


def test_gate_rejects_iou_gain_that_harms_recall():
    def row(f1, recall, iou):
        return {"range_affiliation": {"f1": f1, "recall": recall},
                "range_daily_positive_axes": {"mean_iou": iou},
                "negative_axes": {"far": 0.2}, "execution_success_rate": 1.0}
    result = module.gate({"V0": row(0.8, 0.9, 0.5), "V1": row(0.8, 0.9, 0.5),
                          "V2": row(0.85, 0.85, 0.6)})
    assert not result["passed"]
    assert not result["checks"]["recall_vs_V0"]
