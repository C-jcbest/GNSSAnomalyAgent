from copy import deepcopy
from datetime import timedelta

import pytest

from gnss_sim.evaluation import (
    aggregate_cases,
    evaluate_case,
    load_results_jsonl,
    match_events,
    temporal_iou,
)
from gnss_sim.generator import generate_case
from gnss_sim.labels import event_masks
from gnss_sim.pilot import _planned_cases, _single_metadata, generate_pilot, verify_pilot
from gnss_sim.schemas import DetectionResult, PredictedEvent


def truth(kind):
    return generate_case("case_0001", 42, kind)[1]


def prediction(event, start=None, end=None, axis=None, prediction_id="pred_001",
               predicted_type=None):
    return PredictedEvent(
        prediction_id=prediction_id, axes=[axis or event.axis],
        start_index=event.start_index if start is None else start,
        end_index=event.end_index if end is None else end, type=predicted_type,
    )


def score(case_truth, events, group="single", status="success"):
    return evaluate_case(case_truth, DetectionResult(
        case_id=case_truth.case_id, method="fixture", status=status, events=events), group)


@pytest.mark.parametrize("kind,delta,expected", [
    ("spike", 0, (1, 0, 0)), ("spike", 1, (1, 0, 0)),
    ("spike", 2, (0, 1, 1)), ("step", 3, (1, 0, 0)),
    ("step", 4, (0, 1, 1)),
])
def test_point_tolerances(kind, delta, expected):
    case_truth = truth(kind)
    event = case_truth.events[0]
    case = score(case_truth, [prediction(event, event.start_index + delta,
                                          event.start_index + delta)])
    assert (case["tp"], case["fp"], case["fn"]) == expected


@pytest.mark.parametrize("days,expected", [(54, (1, 0, 0)), (36, (0, 1, 1))])
def test_interval_iou_threshold(days, expected):
    case_truth = truth("slow_trend")
    event = case_truth.events[0]
    case = score(case_truth, [prediction(event, event.end_index - days + 1,
                                          event.end_index)])
    assert (case["tp"], case["fp"], case["fn"]) == expected
    assert temporal_iou(100, 189, 136, 189) == pytest.approx(0.6)


@pytest.mark.parametrize("kind", ["slow_trend", "acceleration", "transient_shift"])
def test_interval_types_share_rule(kind):
    case_truth = truth(kind)
    event = case_truth.events[0]
    assert score(case_truth, [prediction(event)])["tp"] == 1
    assert score(case_truth, [prediction(event, event.start_index,
                                          event.start_index)])["tp"] == 0


def test_prediction_type_is_ignored_but_shape_is_required():
    case_truth = truth("step")
    event = case_truth.events[0]
    assert score(case_truth, [prediction(event, predicted_type="spike")])["tp"] == 1
    assert score(case_truth, [prediction(event, event.start_index,
                                          event.start_index + 3)])["tp"] == 0


def test_strict_axis_and_extra_prediction():
    case_truth = truth("spike")
    event = case_truth.events[0]
    wrong = next(axis for axis in ("N", "E", "U") if axis != event.axis)
    assert (score(case_truth, [prediction(event, axis=wrong)])["tp"],
            score(case_truth, [prediction(event, axis=wrong)])["fp"]) == (0, 1)
    multi_axis = prediction(event).model_copy(update={"axes": ["N", "E", "U"]})
    assert score(case_truth, [multi_axis])["fn"] == 1
    assert score(case_truth, [prediction(event), prediction(event, prediction_id="pred_002")])[
        "fp"] == 1


def test_one_prediction_cannot_cover_two_truth_events():
    case_truth = truth("spike").model_copy(deep=True)
    duplicate = case_truth.events[0].model_copy(update={"event_id": "event_002"})
    part = case_truth.event_contributions[0].model_copy(update={"event_id": "event_002"})
    case_truth.events.append(duplicate)
    case_truth.event_contributions.append(part)
    case = score(case_truth, [prediction(case_truth.events[0])], "multi")
    assert (case["tp"], case["fp"], case["fn"]) == (1, 0, 1)


def test_matching_prioritizes_cardinality_then_quality():
    case_truth = truth("spike").model_copy(deep=True)
    first_event = case_truth.events[0]
    other = first_event.model_copy(update={
        "event_id": "event_002", "start_index": first_event.start_index + 1,
        "end_index": first_event.end_index + 1,
        "start_date": first_event.start_date + timedelta(days=1),
        "end_date": first_event.end_date + timedelta(days=1),
    })
    case_truth.events.append(other)
    first = prediction(first_event, first_event.start_index, first_event.start_index)
    second = prediction(first_event, first_event.start_index - 1, first_event.start_index - 1,
                        prediction_id="pred_002")
    pairs = match_events(case_truth, [first, second])
    assert len(pairs) == 2
    case = score(case_truth, [first, second], "multi")
    assert case["onset_error_sum_days"] == 2


def test_order_and_id_invariance():
    case_truth = truth("complex_multiaxis")
    predictions = [prediction(event, prediction_id=f"pred_{i:03d}")
                   for i, event in enumerate(case_truth.events)]
    baseline = score(case_truth, predictions, "multi")
    changed = deepcopy(case_truth)
    changed.events.reverse()
    changed.event_contributions.reverse()
    for index, event in enumerate(changed.events):
        event.event_id = changed.event_contributions[index].event_id = f"event_{index + 101:03d}"
    reordered = [item.model_copy(update={"prediction_id": f"renamed_{i}"})
                 for i, item in enumerate(reversed(predictions))]
    comparison = score(changed, reordered, "multi")
    assert {key: baseline[key] for key in ("tp", "fp", "fn", "onset_error_sum_days",
                                            "interval_iou_sum", "interval_end_error_sum_days")} == {
        key: comparison[key] for key in ("tp", "fp", "fn", "onset_error_sum_days",
                                          "interval_iou_sum", "interval_end_error_sum_days")}


def test_normal_far_and_failure_denominators():
    normal = truth("normal")
    event = truth("spike").events[0]
    clean = score(normal, [], "normal")
    alarm = score(normal, [prediction(event)], "normal")
    failed_normal = score(normal, [], "normal", "failed")
    missed = score(truth("spike"), [], "single", "failed")
    report = aggregate_cases([clean, alarm, failed_normal, missed])
    assert report["normal_far"] == 0.5
    assert report["fp_per_normal"] == 0.5
    assert report["normal_failure_rate"] == pytest.approx(1 / 3)
    assert report["execution_success_rate"] == 0.5
    assert (report["tp"], report["fp"], report["fn"]) == (0, 1, 1)


def test_micro_metrics_and_onset_mae():
    case_truth = truth("step")
    event = case_truth.events[0]
    hit = score(case_truth, [prediction(event, event.start_index + 2,
                                        event.start_index + 2)])
    miss = score(case_truth, [], "single")
    report = aggregate_cases([hit, miss])
    assert (report["event_precision"], report["event_recall"], report["event_f1"]) == (
        1, 0.5, pytest.approx(2 / 3))
    assert report["onset_mae_days"] == 2


def test_interval_secondary_and_multi_completeness():
    case_truth = truth("slow_trend")
    event = case_truth.events[0]
    full = score(case_truth, [prediction(event)], "multi")
    missing = score(case_truth, [], "multi")
    report = aggregate_cases([full, missing])
    assert report["interval_mean_iou"] == 1
    assert report["interval_end_mae_days"] == 0
    assert report["multi_mean_case_recall"] == 0.5
    assert report["multi_complete_case_rate"] == 0.5
    extra = score(case_truth, [prediction(event), prediction(event,
                  prediction_id="pred_002")], "multi")
    assert aggregate_cases([extra])["multi_complete_case_rate"] == 0


def test_active_and_effect_masks():
    case_truth = truth("step")
    active, effect = event_masks(case_truth)
    event = case_truth.events[0]
    column = ("N", "E", "U").index(event.axis)
    assert active.sum() == 1 and active[event.start_index, column]
    assert effect[event.start_index:, column].all()
    case_truth = truth("slow_trend")
    active, effect = event_masks(case_truth)
    event = case_truth.events[0]
    column = ("N", "E", "U").index(event.axis)
    assert active[:, column].sum() == 90
    assert not active[event.end_index + 1, column]
    assert effect[event.end_index + 1, column]


def test_fixed_plan_has_only_metadata_stratification():
    planned = list(_planned_cases(20260925))
    assert len(planned) == 300
    assert [kind for kind, *_ in planned].count("normal") == 30
    for kind, case_seed, axis, sign in planned[30:150]:
        assert _single_metadata(case_seed) == (axis, sign)
    assert len({case_seed for _, case_seed, _, _ in planned}) == 300


@pytest.fixture(scope="module")
def pilot_directory(tmp_path_factory):
    return generate_pilot(tmp_path_factory.mktemp("p4") / "pilots", 20260925)


def test_pilot_freeze_and_truth_validation(pilot_directory):
    manifest = verify_pilot(pilot_directory)
    assert len(manifest["cases"]) == 300
    assert generate_pilot(pilot_directory.parent, 20260925) == pilot_directory
    with pytest.raises(ValueError, match="different seed"):
        generate_pilot(pilot_directory.parent, 20260926)


def test_missing_results_are_failures_in_all_denominators(pilot_directory):
    from gnss_sim.evaluation import evaluate_pilot

    report = evaluate_pilot(pilot_directory, [], "fixture")
    summary = report["summary"]
    assert summary["cases"] == 300
    assert summary["execution_success_rate"] == 0
    assert summary["normal_failure_rate"] == 1
    assert summary["normal_far"] is None
    assert summary["event_recall"] == 0
    assert summary["multi_complete_case_rate"] == 0


def test_invalid_jsonl_line_is_reported_and_missing_case_fails(tmp_path, pilot_directory):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"case_id":"case_0001","method":"fixture","status":"success","events":[]}'
                    "\n{broken JSON\n", encoding="utf-8")
    results, errors = load_results_jsonl(path)
    assert len(results) == 1 and errors == [{"line": 2, "error": "JSONDecodeError"}]
    from gnss_sim.evaluation import evaluate_pilot

    summary = evaluate_pilot(pilot_directory, results, "fixture")["summary"]
    assert summary["execution_success_rate"] == pytest.approx(1 / 300)
    assert summary["normal_failure_rate"] == pytest.approx(29 / 30)
