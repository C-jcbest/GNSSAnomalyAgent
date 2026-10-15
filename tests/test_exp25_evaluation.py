"""Evaluation keeps failures, missing dates and task identities explicit."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_author_parser import load_author_functions  # noqa: E402
from report_exp25 import aggregate_rows, author_replay, score_record  # noqa: E402

from gnss_research.files import write_json  # noqa: E402
from gnss_research.upstream import ROOT  # noqa: E402


def example():
    observation = {"dates": [str(i) for i in range(40)], "background_end": 9,
                   "displacement_mm": [None if i == 22 else [0, 0, 0] for i in range(40)]}
    event = {"kind": "deformation", "start": 20, "end": 25, "status": "confirmed",
             "phases": [], "evidence": "observed displacement activity"}
    return observation, {"events": [event]}, event


def test_failed_generic_partial_union_is_retained_with_zero_sensitivity():
    observation, truth, _ = example()
    result = {"status": "failed", "prediction": None, "generic_score": None,
              "generic_mask": [20 <= i <= 25 for i in range(40)]}
    row = score_record(observation, truth, result)
    assert row["range"] is None
    assert row["original"]["generic_daily"]["tp"] == 5
    assert row["generic_affiliation"]["f1"] == pytest.approx(1)
    assert row["failed_station_zero_sensitivity"]["generic_daily"]["fn"] == 5
    summary = aggregate_rows([row])
    assert summary["original"]["failed_records"] == 1
    assert "types" not in summary["original"]


def test_failed_typed_prediction_is_scored_as_empty_without_mutation():
    observation, truth, event = example()
    result = {"status": "failed", "prediction": {"events": [event]},
              "generic_score": None, "generic_mask": [True] * 40}
    row = score_record(observation, truth, result)
    assert row["original"]["generic_daily"]["tp"] == 0
    assert row["range"]["deformation"]["affiliation"]["f1"] == 0
    assert row["activity"]["missed_activity_days"] == 5
    assert result["prediction"]["events"] == [event]


def test_activity_counts_exclude_missing_and_history_days():
    observation, truth, event = example()
    predicted = {**event, "start": 8, "end": 28}
    result = {"status": "success", "prediction": {"events": [predicted]},
              "generic_score": None, "generic_mask": [8 <= i <= 28 for i in range(40)]}
    activity = score_record(observation, truth, result)["activity"]
    assert activity == {"activity_stations": 1, "observed_activity_days": 5,
                        "missed_activity_days": 0, "false_activity_days_before_onset": 10,
                        "false_activity_days_after_arrest": 3}


def test_original_parser_replay_retains_partial_success_and_missing_mask(tmp_path):
    if not (ROOT / "vendor/upstream/visualtime/.git").exists():
        pytest.skip("locked author source required")
    observation, truth, _ = example()
    for axis in "NEU":
        for task in ("point", "range"):
            work = tmp_path / f"{axis}-{task}"
            content = "Explanation: []" if task == "point" else "Range: [[20, 25]]"
            if axis == "N" and task == "point":
                content = "Unparseable response"
            write_json(work / "receipt.json", {"http_status": 200, "status": "failed"})
            write_json(work / "response.json", {"choices": [{"message": {"content": content}}]})
    functions, _ = load_author_functions()
    result = author_replay(observation, tmp_path, functions)
    assert result["status"] == "failed"
    assert sum(result["generic_mask"]) == 5
    assert not result["generic_mask"][22]
    assert sum(row["status"] == "success" for row in result["subrequests"]) == 5
    report = score_record(observation, truth, result)
    assert report["generic_affiliation"]["f1"] == 1
    assert report["failed_station_zero_affiliation"]["f1"] == 0
