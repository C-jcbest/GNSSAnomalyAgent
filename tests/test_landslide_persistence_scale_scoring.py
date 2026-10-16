import json

import numpy as np
import pytest
from analyze_landslide_adaptive_review import paired_record_interval
from score_landslide_claim_review import score_target
from score_landslide_persistence_scale import (
    enrich_scores,
    paired_count_interval,
    verify_complete_forecasts,
)

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case


def test_count_bootstrap_matches_old_whole_record_sampling_exactly():
    rows = [{"activity_label": 1, "feature": "steady_motion", "episode_id": "one"}] * 10
    rows += [{"activity_label": 0, "feature": "", "episode_id": ""}] * 10
    results = {}
    for name, predictions in {
        "control": [np.r_[np.ones(7), np.zeros(13)], np.ones(20), np.zeros(20)],
        "treatment": [np.r_[np.ones(9), np.zeros(11)], np.r_[np.ones(10), np.zeros(10)],
                      np.r_[np.ones(5), np.zeros(15)]],
    }.items():
        cases = {}
        for index, activity in enumerate(predictions):
            stage = np.where(activity == 1, 2, 0)
            cases[f"record_{index}"] = evaluate_case(rows, ActivityPrediction(activity, stage))
        results[name] = {"per_case": cases, "aggregate": aggregate_scores(list(cases.values()))}
    assert paired_count_interval(results, "control", "treatment") == (
        paired_record_interval(results, "control", "treatment")
    )


def test_bootstrap_preserves_undefined_f1_when_neither_method_has_positives():
    counts = {"tp": 0, "fp": 0, "fn": 0, "f1": None}
    result = {"per_case": {"quiet": {"daily": counts}}, "aggregate": {"daily": counts}}
    interval = paired_count_interval({"control": result, "treatment": result}, "control", "treatment")
    assert interval["draws_with_defined_f1"] == 0
    assert interval["difference"] is None
    assert interval["exploratory_95_percentile_interval"] is None


def test_partial_run_is_rejected_without_opening_any_references(tmp_path):
    data, heads, baselines = [tmp_path / name for name in ("data", "heads", "baselines")]
    for folder in (data, heads, baselines):
        folder.mkdir()
    protocol = {"case_ids": [f"case_{index:04d}" for index in range(217, 337)],
                "selection": {f"method_{index}": {} for index in range(10)}}
    (data / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (heads / "state.json").write_text('{"status": "inference"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Freeze all 19"):
        verify_complete_forecasts(data, heads, baselines)
    assert not (data / "mechanism-sealed").exists()


def test_missing_registered_method_is_rejected_after_freeze_verification(tmp_path):
    from run_landslide_persistence_scale_baselines import METHODS

    from gnss_sim.artifacts import sha

    data, heads, baselines = [tmp_path / name for name in ("data", "heads", "baselines")]
    for folder in (data, heads, baselines):
        folder.mkdir()
    selected = {f"method_{index}": {} for index in range(10)}
    protocol = {"case_ids": [f"case_{index:04d}" for index in range(217, 337)],
                "selection": selected}
    (data / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (heads / "new-predictions.json").write_text(json.dumps({name: {} for name in selected}), encoding="utf-8")
    # The missing baseline is genuinely absent from an otherwise valid result freeze.
    (baselines / "predictions.json").write_text(json.dumps({name: {} for name in METHODS[:-1]}), encoding="utf-8")
    for folder, freeze in ((heads, "result-freeze.json"), (baselines, "manifest.json")):
        (folder / "state.json").write_text('{"status": "predictions_complete"}', encoding="utf-8")
        bindings = {path.name: sha(path) for path in folder.iterdir()}
        (folder / freeze).write_text(json.dumps(bindings), encoding="utf-8")
    with pytest.raises(ValueError, match="Every registered method"):
        verify_complete_forecasts(data, heads, baselines)


def test_no_activity_family_keeps_stage_coverage_undefined_and_counts_unknowns():
    rows = [{"activity_label": 0, "feature": "", "episode_id": ""}] * 12
    rows += [{"activity_label": -1, "feature": "", "episode_id": ""}] * 2
    prediction = ActivityPrediction(np.r_[np.zeros(12), 1, -1], np.r_[np.zeros(12), 2, -1]).to_dict()
    labels = {"quiet": rows}
    observed = {"quiet": np.r_[np.ones(13, dtype=bool), False]}
    predictions = {"model": {"quiet": prediction}}
    with pytest.warns(RuntimeWarning):
        results, summary, records = score_target(
            labels, observed, {"quiet": np.zeros(14, dtype=bool)}, predictions, {}, {}
        )
    enrich_scores(labels, observed, predictions, results, summary, records)
    assert summary[0]["stage_internal_coverage"] is None
    assert summary[0]["f1"] is None
    assert summary[0]["weak_recall"] is None
    assert summary[0]["stage_internal_acceleration_support"] == 0
    assert records[0]["unknown_reference_predicted_positive"] == 1
    assert records[0]["missing_observation_days"] == 1
    assert records[0]["coverage"] == 1
    assert records[0]["negative_false_positive_rate_conditional_on_decision"] == 0
