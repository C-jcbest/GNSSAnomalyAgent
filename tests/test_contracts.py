
import numpy as np
import pytest

from gnss_research.data import generate, simulate
from gnss_research.experiment import register, verify_dataset
from gnss_research.features import numerical_evidence
from gnss_research.files import read_json
from gnss_research.metrics import evaluate_record, match_events
from gnss_research.plots import export_labels
from gnss_research.schema import Prediction, event_masks, phase_labels
from gnss_research.upstream import parse_strict_list
from gnss_research.vision import parse_prediction, prepare_request


def deformation(start=5, end=10):
    return {"kind": "deformation", "start": start, "end": end, "status": "confirmed",
            "evidence": "raw displacement persists", "phases": [
                {"kind": "acceleration", "start": 6, "end": 8}]}


@pytest.mark.parametrize("mutate", [
    lambda event: event.update(status="uncertain"),
    lambda event: event["phases"][0].update(start=4),
    lambda event: event.update(kind="step"),
])
def test_phase_requires_confirmed_activity_interior(mutate):
    event = deformation()
    mutate(event)
    with pytest.raises(ValueError):
        Prediction.model_validate({"events": [event]})


def test_model_output_rejects_duplicate_keys_and_out_of_bounds():
    with pytest.raises(ValueError):
        parse_prediction('{"events":[],"events":[]}', 30)
    with pytest.raises(ValueError):
        parse_prediction('{"events":[{"kind":"step","start":30,"end":30,'
                         '"status":"confirmed","evidence":"jump","phases":[]}]}', 30)
    with pytest.raises(ValueError):
        parse_strict_list("[[2,31]]", "range", 30)


def test_missing_masks_and_phase_export_share_labels(tmp_path):
    observation, _ = simulate(2, "s0000", "normal")
    observation["displacement_mm"][7] = None
    prediction = Prediction.model_validate({"events": [deformation()]})
    observed = np.array([row is not None for row in observation["displacement_mm"]])
    assert not event_masks(prediction, observed)["deformation"][7]
    assert phase_labels(prediction, observed)[7] is None
    path = tmp_path / "labels.csv"
    export_labels(observation, prediction, path)
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[7]["phase"] == ""
    assert rows[7]["deformation"] == ""
    assert rows[6]["phase"] == "acceleration"


def test_background_scale_does_not_use_future_event():
    observation, _ = simulate(3, "s0000", "normal")
    original = numerical_evidence(observation)
    for day in range(observation["background_end"] + 1, len(observation["dates"])):
        if observation["displacement_mm"][day] is not None:
            observation["displacement_mm"][day] = [1e6, -1e6, 1e6]
    changed = numerical_evidence(observation)
    np.testing.assert_array_equal(original.noise, changed.noise)
    np.testing.assert_array_equal(original.residual[:original.background_end + 1],
                                  changed.residual[:changed.background_end + 1])


def test_dataset_disjointness_and_tamper_rejection(tmp_path):
    dataset = tmp_path / "data"
    generate(dataset, 5, 6, 365)
    manifest = verify_dataset(dataset)
    assert len({row["station_id"] for row in manifest["records"]}) == 18
    target = dataset / manifest["records"][0]["observation"]
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset changed"):
        verify_dataset(dataset)


def test_visual_input_contains_no_labels_and_reference_ablation_removes_derived_scale(tmp_path):
    observation, _ = simulate(6, "s0000", "mixed")
    payload = prepare_request(observation, "fusion_no_reference", tmp_path / "request", "model")
    import json

    evidence = json.loads(payload["messages"][1]["content"][0]["text"])
    assert set(evidence) == {"length", "visible_axes", "missing_indices", "numerical_blocks"}
    assert all(set(block) <= {"start", "end", "observed_days", "raw_median_mm"}
               for block in evidence["numerical_blocks"])
    request = read_json(tmp_path / "request/request.json")
    assert request == payload


def test_uncertain_detections_count_as_misses():
    truth = {"events": [deformation()]}
    uncertain = deformation()
    uncertain.update(status="uncertain", phases=[])
    result = {"prediction": {"events": [uncertain]}, "generic_mask": [False] * 30,
              "generic_score": None}
    report = evaluate_record(truth, result, np.ones(30, dtype=bool), 0)
    assert report["types"]["deformation"]["daily"]["fn"] == 6
    assert report["uncertain_events"] == 1


def test_event_matching_maximizes_cardinality_and_respects_missingness():
    report = match_events([(0, 4), (4, 8)], [(0, 8), (0, 3)],
                          np.ones(9, dtype=bool), 0.3)
    assert report["tp"] == 2
    observed = np.array([True, False, True])
    report = match_events([(0, 2)], [(0, 2)], observed, 0.5)
    assert report["f1"] == 1.0


def test_registration_enforces_budget_without_calling_model(tmp_path):
    dataset = tmp_path / "data"
    generate(dataset, 7, 6, 365)
    with pytest.raises(ValueError, match="budget"):
        register(dataset, tmp_path / "run", ["fusion"], "test", 5, "model", 5, 1)
    assert not (tmp_path / "run").exists()
