import copy

import pytest

from gnss_sim.landslide_stage_agreement import (
    disagreement_spans,
    stage_agreement,
    support_filter_effect,
)


def rows(features, evaluable=None):
    if evaluable is None:
        evaluable = [feature != "unknown" for feature in features]
    reference = [{"case_id": "case_fixture", "day_index": day, "date": f"date-{day}",
                  "activity_label": 1, "feature": feature, "stage_evaluable": evaluable[day]}
                 for day, feature in enumerate(features)]
    prediction = [{key: value for key, value in row.items() if key != "stage_evaluable"}
                  for row in reference]
    return reference, prediction


def test_abstention_keeps_common_denominator_and_unknown_reference_is_not_negative():
    actual, predicted = rows(["acceleration", "acceleration", "steady_motion", "deceleration", "unknown"])
    predicted[1]["feature"] = "unknown"
    predicted[2]["feature"] = "acceleration"
    predicted[4]["feature"] = "acceleration"
    result = stage_agreement(actual, predicted)
    assert result["reference_evaluable_days"] == 4
    assert result["exact_agreement_days"] == 2
    assert result["different_class_days"] == 1
    assert result["abstained_evaluable_days"] == 1
    assert result["agreement_rate"] == 0.5
    assert result["prediction_coverage_on_evaluable"] == 0.75
    assert result["per_class"]["acceleration"]["fp"] == 1
    assert result["per_class"]["acceleration"]["fn"] == 1
    assert result["per_class"]["acceleration"]["f1"] == 0.5
    assert result["development_agreement_macro_f1"] == 0.5
    assert result["reference_unknown_prediction_features"] == {"acceleration": 1}


def test_empty_reference_and_absent_classes_do_not_create_perfect_macro_scores():
    actual, predicted = rows(["unknown", "unknown"])
    result = stage_agreement(actual, predicted)
    assert result["agreement_rate"] is None
    assert result["development_agreement_macro_f1"] is None
    actual, predicted = rows(["acceleration"])
    assert stage_agreement(actual, predicted)["development_agreement_macro_f1"] is None


def test_support_filter_reports_lost_agreement_without_dropping_dates():
    actual, native = rows(["acceleration", "deceleration", "unknown"])
    native[1]["feature"] = "acceleration"
    native[2]["feature"] = "steady_motion"
    final = copy.deepcopy(native)
    for row in final:
        row["feature"] = "unknown"
    result = support_filter_effect(actual, native, final)
    assert result == {"suppressed_activity_days": 3, "suppressed_evaluable_days": 2,
                      "native_agreement_to_abstention": 1, "native_disagreement_to_abstention": 1,
                      "suppressed_reference_unknown_days": 1}
    assert stage_agreement(actual, final)["reference_evaluable_days"] == 2
    assert stage_agreement(actual, final)["per_class"]["acceleration"]["fn"] == 1


def test_disagreement_bands_split_at_unassessed_dates_and_class_changes():
    actual, predicted = rows(["acceleration", "acceleration", "unknown", "acceleration", "deceleration"])
    for row in predicted:
        row["feature"] = "unknown"
    assert disagreement_spans(actual, predicted) == [
        {"case_id": "case_fixture", "start": 0, "stop": 2, "days": 2,
         "reference": "acceleration", "prediction": "unknown"},
        {"case_id": "case_fixture", "start": 3, "stop": 4, "days": 1,
         "reference": "acceleration", "prediction": "unknown"},
        {"case_id": "case_fixture", "start": 4, "stop": 5, "days": 1,
         "reference": "deceleration", "prediction": "unknown"},
    ]


@pytest.mark.parametrize("change", ["dropped", "date", "activity", "duplicate", "bad_evaluable", "outside_stage"])
def test_calendar_and_activity_changes_fail_before_scoring(change):
    actual, predicted = rows(["acceleration", "deceleration"])
    if change == "dropped":
        predicted.pop()
    elif change == "date":
        predicted[0]["date"] = "wrong"
    elif change == "activity":
        predicted[0]["activity_label"] = 0
    elif change == "duplicate":
        actual[1] = copy.deepcopy(actual[0])
        predicted[1] = copy.deepcopy(predicted[0])
    elif change == "bad_evaluable":
        actual[0]["stage_evaluable"] = "false"
    elif change == "outside_stage":
        actual[0]["stage_evaluable"] = False
        actual[0]["activity_label"] = -1
        predicted[0]["activity_label"] = -1
    with pytest.raises(ValueError):
        stage_agreement(actual, predicted)


def test_support_filter_cannot_reclassify_predictions():
    actual, native = rows(["acceleration"])
    final = copy.deepcopy(native)
    final[0]["feature"] = "deceleration"
    with pytest.raises(ValueError, match="only suppress"):
        support_filter_effect(actual, native, final)
