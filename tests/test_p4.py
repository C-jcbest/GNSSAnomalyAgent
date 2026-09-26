import json

import pytest
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

from gnss_sim.evaluation import (
    _point_case,
    _point_vector,
    _range_case,
    _range_vector,
    evaluate_pilot,
    load_results_jsonl,
)
from gnss_sim.generator import generate_case
from gnss_sim.pilot import _planned_cases, _single_metadata, generate_pilot, verify_pilot
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult


def truth(kind):
    return generate_case("case_0001", 42, kind)[1]


def point_result(case_id="case_0001", status="success", **axes):
    return PointResult(case_id=case_id, method="fixture", status=status,
                       predictions={axis: axes.get(axis, []) for axis in ("N", "E", "U")})


def range_result(case_id="case_0001", status="success", **axes):
    return RangeResult(case_id=case_id, method="fixture", status=status,
                       predictions={axis: axes.get(axis, []) for axis in ("N", "E", "U")})


def test_strict_task_specific_result_schemas():
    assert point_result(N=[0, 364]).predictions.N == [0, 364]
    assert range_result(E=[[0, 0], [363, 364]]).predictions.E == [(0, 0), (363, 364)]
    for bad in (-1, 365, 1.5, True, "3"):
        with pytest.raises(ValueError):
            point_result(N=[bad])
    for interval in ([1, 0], [-1, 2], [0, 365], [1.5, 2], [1]):
        with pytest.raises(ValueError):
            range_result(N=[interval])
    with pytest.raises(ValueError):
        PointResult(case_id="case_0001", method="fixture", status="success",
                    predictions={"N": [], "E": []})
    with pytest.raises(ValueError):
        PointResult(case_id="case_0001", method="fixture", status="success",
                    predictions={"N": [], "E": [], "U": [], "X": []})
    with pytest.raises(ValueError):
        PointResult(case_id="case_0001", method="fixture", status="success",
                    predictions={"N": [], "E": [], "U": []}, events=[])


@pytest.mark.parametrize("kind", ["spike", "step"])
def test_point_task_requires_exact_day_for_spike_and_step(kind):
    case_truth = truth(kind)
    event = case_truth.events[0]
    exact = _point_case(case_truth, point_result(**{event.axis: [event.start_index]}))
    off_by_one = _point_case(case_truth, point_result(**{event.axis: [event.start_index + 1]}))
    assert exact[:3] == (1, 0, 0)
    assert off_by_one[:3] == (0, 1, 1)
    if kind == "step":
        assert _point_vector([event.start_index]).count(1) == 1


def test_point_vectors_deduplicate_and_keep_axes_separate():
    case_truth = truth("spike")
    event = case_truth.events[0]
    wrong = next(axis for axis in ("N", "E", "U") if axis != event.axis)
    duplicate = _point_case(case_truth, point_result(**{
        event.axis: [event.start_index, event.start_index]}))
    wrong_axis = _point_case(case_truth, point_result(**{wrong: [event.start_index]}))
    assert duplicate[:3] == (1, 0, 0)
    assert wrong_axis[:3] == (0, 1, 1)


def test_point_task_ignores_range_truth_and_counts_negative_axis_alarm():
    case_truth = truth("slow_trend")
    event = case_truth.events[0]
    result = _point_case(case_truth, point_result(**{event.axis: [event.start_index]}))
    assert result == (0, 1, 0, 3, 1)


def test_range_vectors_are_inclusive_and_merge_on_binary_grid():
    vector = _range_vector([(10, 10), (12, 15), (14, 18)])
    assert vector[10] == vector[12] == vector[18] == 1
    assert vector[9] == vector[11] == vector[19] == 0
    assert convert_vector_to_events(vector) == [(10, 11), (12, 19)]


@pytest.mark.parametrize("kind", ["slow_trend", "acceleration", "transient_shift"])
def test_range_task_scores_all_interval_truth_types(kind):
    case_truth = truth(kind)
    event = case_truth.events[0]
    scores, _, _ = _range_case(case_truth, range_result(**{
        event.axis: [[event.start_index, event.end_index]]}))
    assert scores == pytest.approx([(1.0, 1.0, 1.0)])


def test_range_task_calls_upstream_affiliation_with_same_vectors():
    case_truth = truth("slow_trend")
    event = case_truth.events[0]
    predicted = (event.start_index + 10, event.end_index)
    scores, _, _ = _range_case(case_truth, range_result(**{event.axis: [predicted]}))
    upstream = pr_from_events(
        convert_vector_to_events(_range_vector([predicted])),
        convert_vector_to_events(_range_vector([(event.start_index, event.end_index)])),
        Trange=(0, 365),
    )
    p, r, f1 = scores[0]
    assert p == pytest.approx(upstream["precision"])
    assert r == pytest.approx(upstream["recall"])
    assert f1 == pytest.approx(2 * p * r / (p + r))


def test_range_task_ignores_point_truth_and_counts_negative_axis_alarm():
    case_truth = truth("spike")
    event = case_truth.events[0]
    scores, negative, alarms = _range_case(case_truth, range_result(**{
        event.axis: [[event.start_index, event.start_index]]}))
    assert scores == [] and (negative, alarms) == (3, 1)


def test_empty_and_failed_range_prediction_score_zero_on_positive_axis():
    case_truth = truth("slow_trend")
    event = case_truth.events[0]
    assert _range_case(case_truth, range_result())[0] == [(0, 0, 0)]
    failed = range_result(status="failed", **{
        event.axis: [[event.start_index, event.end_index]]})
    assert _range_case(case_truth, failed)[0] == [(0, 0, 0)]
    assert _range_case(case_truth, None)[0] == [(0, 0, 0)]


def test_failed_point_prediction_becomes_false_negative_not_false_alarm():
    case_truth = truth("spike")
    event = case_truth.events[0]
    failed = point_result(status="failed", **{event.axis: [event.start_index]})
    assert _point_case(case_truth, failed) == (0, 0, 1, 0, 0)
    assert _point_case(case_truth, None) == (0, 0, 1, 0, 0)


def test_fixed_plan_is_unchanged_and_stratifies_only_on_metadata():
    planned = list(_planned_cases(20260925))
    assert len(planned) == 300
    assert [kind for kind, *_ in planned].count("normal") == 30
    for kind, case_seed, axis, sign in planned[30:150]:
        assert _single_metadata(case_seed) == (axis, sign)
    assert len({case_seed for _, case_seed, _, _ in planned}) == 300


@pytest.fixture(scope="module")
def pilot_directory(tmp_path_factory):
    return generate_pilot(tmp_path_factory.mktemp("p4") / "pilots", 20260925)


def test_pilot_files_and_generator_are_still_frozen(pilot_directory):
    manifest = verify_pilot(pilot_directory)
    assert len(manifest["cases"]) == 300
    assert generate_pilot(pilot_directory.parent, 20260925) == pilot_directory
    with pytest.raises(ValueError, match="different seed"):
        generate_pilot(pilot_directory.parent, 20260926)


def _pilot_truth(directory, entry):
    return CaseTruth.model_validate_json(
        (directory / "cases" / entry["case_id"] / "truth.json").read_bytes())


def test_point_report_micro_counts_far_and_fixed_failure_denominator(pilot_directory):
    manifest = verify_pilot(pilot_directory)
    normal = manifest["cases"][0]
    spike = next(entry for entry in manifest["cases"] if entry["case_type"] == "spike")
    point = _pilot_truth(pilot_directory, spike).events[0]
    gt_points = sum(len({event.start_index for event in _pilot_truth(pilot_directory, entry).events
                         if event.axis == axis and event.type in ("spike", "step")})
                    for entry in manifest["cases"] for axis in ("N", "E", "U"))
    report = evaluate_pilot(pilot_directory, [
        point_result(normal["case_id"], N=[5]),
        point_result(spike["case_id"], **{point.axis: [point.start_index]}),
    ], "fixture", "point")
    assert set(report) == {"task", "precision", "recall", "f1", "far",
                           "execution_success_rate"}
    assert report["precision"] == 0.5
    assert report["recall"] == pytest.approx(1 / gt_points)
    assert report["f1"] == pytest.approx(2 / (2 + 1 + gt_points - 1))
    assert report["far"] == pytest.approx(1 / 5)
    assert report["execution_success_rate"] == pytest.approx(2 / 300)


def test_range_report_macro_over_positive_case_axes(pilot_directory):
    manifest = verify_pilot(pilot_directory)
    normal = manifest["cases"][0]
    single = next(entry for entry in manifest["cases"] if entry["case_type"] == "slow_trend")
    event = _pilot_truth(pilot_directory, single).events[0]
    positive_units = sum(any(item.type in ("slow_trend", "acceleration", "transient_shift")
                             and item.axis == axis for item in _pilot_truth(pilot_directory, entry).events)
                         for entry in manifest["cases"] for axis in ("N", "E", "U"))
    report = evaluate_pilot(pilot_directory, [
        range_result(normal["case_id"], N=[[5, 7]]),
        range_result(single["case_id"], **{
            event.axis: [[event.start_index, event.end_index]]}),
    ], "fixture", "range")
    assert set(report) == {"task", "affiliation_precision", "affiliation_recall",
                           "affiliation_f1", "far", "execution_success_rate"}
    for key in ("affiliation_precision", "affiliation_recall", "affiliation_f1"):
        assert report[key] == pytest.approx(1 / positive_units)
    assert report["far"] == pytest.approx(1 / 5)
    assert report["execution_success_rate"] == pytest.approx(2 / 300)


def test_range_report_averages_each_axis_f1_before_reporting(pilot_directory):
    manifest = verify_pilot(pilot_directory)
    slow = next(entry for entry in manifest["cases"] if entry["case_type"] == "slow_trend")
    transient = next(entry for entry in manifest["cases"]
                     if entry["case_type"] == "transient_shift")
    slow_event = _pilot_truth(pilot_directory, slow).events[0]
    transient_truth = _pilot_truth(pilot_directory, transient)
    transient_event = transient_truth.events[0]
    shifted = (transient_event.start_index + 3, transient_event.end_index)
    partial = _range_case(transient_truth, range_result(transient["case_id"], **{
        transient_event.axis: [shifted]}))[0][0]
    positive_units = sum(any(item.type in ("slow_trend", "acceleration", "transient_shift")
                             and item.axis == axis for item in _pilot_truth(pilot_directory, entry).events)
                         for entry in manifest["cases"] for axis in ("N", "E", "U"))
    report = evaluate_pilot(pilot_directory, [
        range_result(slow["case_id"], **{
            slow_event.axis: [[slow_event.start_index, slow_event.end_index]]}),
        range_result(transient["case_id"], **{transient_event.axis: [shifted]}),
    ], "fixture", "range")
    assert report["affiliation_precision"] == pytest.approx((1 + partial[0]) / positive_units)
    assert report["affiliation_recall"] == pytest.approx((1 + partial[1]) / positive_units)
    assert report["affiliation_f1"] == pytest.approx((1 + partial[2]) / positive_units)


@pytest.mark.parametrize("task", ["point", "range"])
def test_all_missing_cases_still_enter_execution_and_positive_denominators(pilot_directory, task):
    report = evaluate_pilot(pilot_directory, [], "fixture", task)
    assert report["execution_success_rate"] == 0
    assert report["far"] is None  # failed negative axes are not certified clean
    if task == "point":
        assert (report["precision"], report["recall"], report["f1"]) == (0, 0, 0)
    else:
        assert (report["affiliation_precision"], report["affiliation_recall"],
                report["affiliation_f1"]) == (0, 0, 0)


def test_invalid_jsonl_and_old_schema_become_missing_case_failures(tmp_path, pilot_directory):
    path = tmp_path / "point.jsonl"
    valid = point_result().model_dump(mode="json")
    path.write_text(json.dumps(valid) + "\n{broken JSON\n" + json.dumps({
        "case_id": "case_0002", "method": "fixture", "status": "success", "events": []
    }) + "\n", encoding="utf-8")
    results, errors = load_results_jsonl(path, "point")
    assert len(results) == 1
    assert [error["line"] for error in errors] == [2, 3]
    report = evaluate_pilot(pilot_directory, results, "fixture", "point")
    assert report["execution_success_rate"] == pytest.approx(1 / 300)


def test_task_mismatch_duplicate_and_unknown_case_are_rejected(pilot_directory):
    with pytest.raises(ValueError, match="schema"):
        evaluate_pilot(pilot_directory, [range_result()], "fixture", "point")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_pilot(pilot_directory, [point_result(), point_result()], "fixture", "point")
    with pytest.raises(ValueError, match="unknown"):
        evaluate_pilot(pilot_directory, [point_result("case_9999")], "fixture", "point")
