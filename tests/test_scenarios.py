from collections import Counter

import numpy as np
import pytest

from gnss_sim.generator import generate_case
from gnss_sim.scenarios import MAKERS
from gnss_sim.schemas import SCENARIO_TYPES, CaseTruth, GenerationRequest, PointResult, RangeResult
from gnss_sim.storage import DatasetStore


@pytest.mark.parametrize("scenario", SCENARIO_TYPES)
@pytest.mark.parametrize("seed", range(30))
def test_p3_template_constraints_and_contributions(scenario, seed):
    case_input, truth = generate_case("case_0001", seed, scenario)
    events = truth.events
    counts = Counter(event.type for event in events)
    assert truth.scenario_type == scenario
    assert 2 <= len(events) <= 6
    assert counts["slow_trend"] + counts["acceleration"] <= 1
    assert counts["step"] <= 2 and counts["transient_shift"] <= 2 and counts["spike"] <= 4
    assert [event.event_id for event in events] == [f"event_{i:03d}" for i in range(1, len(events) + 1)]
    assert [event.start_index for event in events] == sorted(event.start_index for event in events)
    assert all(60 <= event.start_index <= event.end_index <= 304 for event in events)
    spikes = [event.start_index for event in events if event.type == "spike"]
    steps = [event.start_index for event in events if event.type == "step"]
    transients = [event for event in events if event.type == "transient_shift"]
    assert all(b - a >= 7 for a, b in zip(spikes, spikes[1:]))
    assert all(b - a >= 45 for a, b in zip(steps, steps[1:]))
    assert all(a.end_index < b.start_index for a, b in zip(transients, transients[1:]))
    if scenario == "complex_multiaxis":
        assert len({event.axis for event in events}) >= 2
    if scenario in ("longterm_with_local", "longterm_with_change"):
        longterm = next(event for event in events if event.type in ("slow_trend", "acceleration"))
        assert any(longterm.start_index <= event.start_index <= longterm.end_index
                   for event in events if event.type == "spike")

    deformation = np.zeros((365, 3))
    artifact = np.zeros((365, 3))
    for event, part in zip(events, truth.event_contributions):
        assert event.event_id == part.event_id
        values = np.asarray(part.values_mm)
        assert values.shape == (365, 3)
        assert not np.delete(values, ("N", "E", "U").index(event.axis), axis=1).any()
        magnitude = (event.parameters.final_offset_mm if event.type in ("slow_trend", "acceleration")
                     else event.parameters.amplitude_mm)
        expected, _ = MAKERS[event.type](case_input.dates, event.axis, event.start_index, magnitude)
        np.testing.assert_array_equal(values, expected)
        if part.component == "injected_deformation":
            deformation += values
        else:
            artifact += values
    np.testing.assert_allclose(deformation, truth.injected_deformation_mm, rtol=0, atol=1e-14)
    np.testing.assert_allclose(artifact, truth.observation_artifact_mm, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        case_input.observed_coordinate_mm,
        np.asarray(case_input.reference_coordinate_mm) + np.asarray(truth.normal_background_mm)
        + np.asarray(truth.measurement_noise_mm) + deformation + artifact,
        rtol=0, atol=5e-15,
    )
    _, normal = generate_case("case_0001", seed, "normal")
    assert truth.normal_background_mm == normal.normal_background_mm
    assert truth.measurement_noise_mm == normal.measurement_noise_mm


def test_p3_mixed_batch_has_six_scenarios_and_is_reproducible(tmp_path):
    store = DatasetStore(tmp_path / "generated")
    request = GenerationRequest(seed=20260924, count=54, case_type="all_scenarios")
    first = store.generate_sync(request)
    second = store.generate_sync(request)
    assert first.status == second.status == "complete"
    assert set(first.type_counts) == set(SCENARIO_TYPES)
    assert sum(first.type_counts.values()) == 54
    assert all(first.type_counts[scenario] == 9 for scenario in SCENARIO_TYPES)
    for left, right in zip(first.cases, second.cases):
        assert left == right
        assert store.get_case_input(first.dataset_id, left.case_id) == store.get_case_input(second.dataset_id, right.case_id)
        assert store.get_case_truth(first.dataset_id, left.case_id) == store.get_case_truth(second.dataset_id, right.case_id)
        assert left.event_count >= 2


def test_point_and_range_result_have_no_prediction_count_limit():
    base = dict(case_id="case_0001", method="future_method", status="success")
    points = PointResult(**base, predictions={"N": list(range(30)), "E": [], "U": []})
    ranges = RangeResult(**base, predictions={"N": [[i, i] for i in range(30)],
                                               "E": [], "U": []})
    assert len(points.predictions.N) == len(ranges.predictions.N) == 30


def test_truth_schema_does_not_encode_p3_event_cap():
    _, truth = generate_case("case_0001", 42, "complex_multiaxis")
    payload = truth.model_dump(mode="json")
    while len(payload["events"]) <= 6:
        index = len(payload["events"]) + 1
        event = payload["events"][0].copy()
        part = payload["event_contributions"][0].copy()
        event["event_id"] = part["event_id"] = f"event_{index:03d}"
        payload["events"].append(event)
        payload["event_contributions"].append(part)
    assert len(CaseTruth.model_validate(payload).events) == 7


def test_same_and_cross_axis_cases_are_reachable():
    same = cross = False
    for seed in range(30):
        _, truth = generate_case("case_0001", seed, "multi_spike")
        axes = [event.axis for event in truth.events]
        same |= len(set(axes)) == 1
        cross |= len(set(axes)) > 1
    assert same and cross


def test_optional_repetitions_and_both_longterm_profiles_are_reachable():
    observed = set()
    for scenario in SCENARIO_TYPES:
        for seed in range(30):
            _, truth = generate_case("case_0001", seed, scenario)
            counts = Counter(event.type for event in truth.events)
            if counts["step"] == 2:
                observed.add("two_steps")
            if counts["transient_shift"] == 2:
                observed.add("two_transients")
            if counts["spike"] == 4:
                observed.add("four_spikes")
            observed.update(event.type for event in truth.events if event.type in ("slow_trend", "acceleration"))
    assert observed == {"two_steps", "two_transients", "four_spikes", "slow_trend", "acceleration"}
