import numpy as np

from gnss_research.features import numerical_evidence
from gnss_research.segmentation import DisplacementLinearCost, segment_proposals
from gnss_research.segmented_review import (
    prepare_pair_focus,
    repair_return_pairs,
    segmented_fusion,
    stabilized_displacement,
)


def observation(values):
    return {"dates": list(range(len(values))), "background_end": 89,
            "displacement_mm": values.tolist()}


def empty_numeric():
    return {"prediction": {"events": []}}


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "evidence": "raw displacement", "phases": []}


def test_multiaxis_cost_matches_direct_fit_at_irregular_original_times():
    rng = np.random.default_rng(54)
    time = np.array([0, 1, 3, 6, 7, 9, 12, 15, 19, 20, 25, 29])
    values = rng.normal(size=(len(time), 3)) + time[:, None] * [.1, -.2, .3]
    cost = DisplacementLinearCost().fit(np.column_stack([time, values]))
    local_time = time[2:11]
    design = np.column_stack([np.ones(len(local_time)), local_time])
    coefficients = np.linalg.lstsq(design, values[2:11], rcond=None)[0]
    expected = np.square(values[2:11] - design @ coefficients).sum()
    assert np.isclose(cost.error(2, 11), expected)


def test_stationary_step_is_not_active_motion_and_return_is_one_transient():
    rng = np.random.default_rng(8)
    values = rng.normal(0, .2, (730, 3))
    values[533:] += [4, -3, 2]
    obs = observation(values)
    obs["background_end"] = 364
    result = segment_proposals(obs, empty_numeric())
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("step", 533, 533)]
    values[558:] -= [4, -3, 2]
    obs = observation(values)
    obs["background_end"] = 364
    obs["displacement_mm"][545] = None
    result = segment_proposals(obs, empty_numeric())
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("transient", 533, 557)]
    assert not result["generic_mask"][545]
    assert numerical_evidence(obs).observed[546]


def test_visual_pair_owns_edges_and_numeric_activity_stops_before_platform():
    values = np.zeros((210, 3))
    obs = observation(values)
    obs["displacement_mm"][145] = None
    proposals = [{"id": "a", **event("transient", 133, 157)},
                 {"id": "b", **event("step", 133, 133)},
                 {"id": "c", **event("step", 158, 158)},
                 {"id": "d", **event("deformation", 120, 175)}]
    candidates = {"proposals": proposals, "numerical_prediction": empty_numeric()}
    visual = {"status": "success", "prediction": {"events": [
        event("transient", 134, 157), event("step", 133, 133), event("step", 158, 158),
        event("deformation", 125, 180)]}}
    result = segmented_fusion(obs, visual, candidates)
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("transient", 133, 157), ("deformation", 120, 175)]
    assert not result["generic_mask"][145]
    assert not result["generic_mask"][176]
    ablation = segmented_fusion(obs, visual, candidates, visual_bounds=True)
    assert ablation["prediction"]["events"][-1]["end"] == 180
    visual["prediction"]["events"] = [{**e, "status": "uncertain"}
                                         for e in visual["prediction"]["events"]]
    assert segmented_fusion(obs, visual, candidates)["prediction"]["events"] == []
    assert segmented_fusion(obs, {"status": "failed"}, candidates)["status"] == "failed"


def test_semantic_repair_preserves_activity_and_discloses_failed_visual_path():
    obs = observation(np.zeros((210, 3)))
    obs["displacement_mm"][145] = None
    numerical_events = [event("step", 133, 133), event("step", 158, 158),
                        event("deformation", 120, 175)]
    candidate = {"id": "pair", **event("transient", 133, 157), "status": "uncertain"}
    candidates = {"proposals": [candidate], "numerical_prediction": {
        "prediction": {"events": numerical_events}}}
    visual = {"status": "success", "prediction": {"events": [event("transient", 134, 157)]}}
    result = repair_return_pairs(obs, visual, candidates)
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("deformation", 120, 175), ("transient", 133, 157)]
    assert result["semantic_repairs"] == [{"proposal_id": "pair", "edges": [133, 158]}]
    assert not result["generic_mask"][145]
    failed = repair_return_pairs(obs, {"status": "failed"}, candidates)
    assert failed["visual_dependency_status"] == "failed"
    assert failed["prediction"]["events"] == numerical_events
    uncertain = {"status": "success", "prediction": {"events": [
        {**event("transient", 134, 157), "status": "uncertain"}]}}
    assert repair_return_pairs(obs, uncertain, candidates)["prediction"]["events"] == numerical_events


def test_auxiliary_view_keeps_level_artifact_and_missing_mask_after_motion_subtraction():
    rng = np.random.default_rng(11)
    time = np.arange(730)
    values = rng.normal(0, .2, (730, 3)) + time[:, None] * [.02, -.01, .01]
    values[533:558] += [4, -3, 2]
    obs = observation(values)
    obs["background_end"] = 364
    obs["displacement_mm"][545] = None
    numeric = {"prediction": {"events": []}, "jumps": [{"index": 533}, {"index": 558}]}
    adjusted = stabilized_displacement(obs, numeric)
    before = np.nanmean(adjusted[520:533], axis=0)
    middle = np.nanmean(adjusted[538:550], axis=0)
    after = np.nanmean(adjusted[561:574], axis=0)
    assert np.allclose(middle - before, [4, -3, 2], atol=.4)
    assert np.allclose(after - before, [0, 0, 0], atol=.4)
    assert np.isnan(adjusted[545]).all()


def test_no_ambiguity_preserves_numeric_result_without_requesting_model(tmp_path):
    import json

    rng = np.random.default_rng(8)
    obs = observation(rng.normal(0, .2, (730, 3)))
    obs["background_end"] = 364
    work = tmp_path / "review"
    assert prepare_pair_focus(obs, empty_numeric(), work, "test-model", "test system") is None
    result = json.loads((work / "result.json").read_text())
    assert result["request_count"] == 0
    assert result["review_mode"] == "not_requested_no_ambiguity"
    assert not (work / "request.json").exists()
    candidates = json.loads((work / "candidates.json").read_text())
    repaired = repair_return_pairs(obs, result, candidates)
    assert repaired["visual_dependency_status"] == "not_requested_no_ambiguity"
    assert repaired["prediction"] == candidates["numerical_prediction"]["prediction"]
