import numpy as np
import pytest
from analyze_landslide_dual_reference_confirmation import score_target


def reference_rows():
    return [{"activity_label": activity, "feature": feature, "episode_id": episode,
             "weak_signal": 0, "day_index": day}
            for day, (activity, feature, episode) in enumerate([
                (0, "none", ""), (1, "acceleration", "motion"),
                (1, "steady_motion", "motion"), (1, "deceleration", "motion"),
                (-1, "unknown", ""), (-1, "unknown", "")])]


def prediction():
    return {"activity": [0, 1, -1, 1, 1, -1], "stage": [0, 1, -1, 3, 2, -1],
            "status": "success", "gated": True, "stage_status": "success"}


def test_unknown_reference_is_excluded_but_unknown_prediction_misses_activity():
    labels = {"record": reference_rows()}
    observed = {"record": np.array([True, True, True, True, True, False])}
    weak = {"record": np.array([False, False, True, True, False, False])}
    results, summaries, _ = score_target(labels, observed, weak, {"visual": {"record": prediction()}}, {}, {})
    daily = results["visual"]["aggregate"]["daily"]
    assert (daily["tp"], daily["fp"], daily["fn"]) == (2, 0, 1)
    assert daily["f1"] == .8
    assert summaries[0]["unknown_reference_predicted_positive"] == 1
    assert summaries[0]["weak_recall"] == .5
    assert results["visual"]["internal_stages"]["f1_ASD"] == [1., 0., 1.]


def test_missing_day_cannot_gain_a_decision():
    value = prediction()
    value["activity"][-1] = 0
    with pytest.raises(ValueError, match="Missing observation"):
        score_target({"record": reference_rows()},
            {"record": np.array([True, True, True, True, True, False])},
            {"record": np.zeros(6, dtype=bool)}, {"visual": {"record": value}}, {}, {})


def test_target_population_cannot_silently_drop_records():
    with pytest.raises(ValueError, match="Every method"):
        score_target({"record": reference_rows()}, {}, {}, {"visual": {}}, {}, {})
