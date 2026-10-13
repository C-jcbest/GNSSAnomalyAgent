from gnss_research.data import generate
from gnss_research.files import read_json
from gnss_research.fusion import boundary_review, ranked_boundary_review


def event(kind, start, end, status="confirmed"):
    return {"kind": kind, "start": start, "end": end, "status": status,
            "evidence": "raw displacement observation", "phases": []}


def test_activity_boundaries_can_correct_numeric_platform_and_missing_is_unknown():
    observation = {"dates": list(range(100)), "displacement_mm": [[0., 0., 0.]] * 100}
    observation["displacement_mm"][15] = None
    candidates = {"numerical_prediction": {"prediction": {"events": [
        event("deformation", 10, 80), event("spike", 30, 30)]}}, "windows": []}
    visual = {"status": "success", "prediction": {"events": [
        event("deformation", 12, 40), event("spike", 31, 31),
        event("deformation", 90, 95), event("deformation", 5, 9, "uncertain")]}}
    result = boundary_review(observation, visual, candidates)
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("deformation", 12, 40)]
    assert not result["generic_mask"][15]
    assert not any(result["generic_mask"][41:])
    ablation = boundary_review(observation, visual, candidates, numerical_points=True)
    assert [(e["start"], e["end"]) for e in ablation["prediction"]["events"]
            if e["kind"] == "spike"] == [(30, 30)]


def test_only_missing_overlap_and_visual_failure_cannot_confirm_activity():
    observation = {"dates": list(range(30)), "displacement_mm": [[0., 0., 0.]] * 30}
    observation["displacement_mm"][10] = None
    candidates = {"numerical_prediction": {"prediction": {"events": [
        event("deformation", 5, 10)]}}, "windows": []}
    visual = {"status": "success", "prediction": {"events": [event("deformation", 10, 20)]}}
    assert boundary_review(observation, visual, candidates)["prediction"]["events"] == []
    assert boundary_review(observation, {"status": "failed"}, candidates)["status"] == "failed"


def test_more_training_preserves_validation_and_test_random_streams(tmp_path):
    original, expanded = tmp_path / "base", tmp_path / "large"
    generate(original, 6, 6, 365)
    generate(expanded, 6, 6, 365, 18)
    base = read_json(original / "manifest.json")
    large = read_json(expanded / "manifest.json")
    assert large["split_counts"] == {"train": 18, "validation": 6, "test": 6}
    assert len({r["station_id"] for r in large["records"]}) == 30
    assert set(r["observation_sha256"] for r in base["records"]) <= set(
        r["observation_sha256"] for r in large["records"])
    for split in ("validation", "test"):
        assert [r for r in base["records"] if r["split"] == split] == [
            r for r in large["records"] if r["split"] == split]


def test_ranked_points_do_not_require_numeric_positive_but_need_exact_visual_day():
    observation = {"dates": list(range(30)), "displacement_mm": [[0., 0., 0.]] * 30}
    candidates = {"numerical_prediction": {"prediction": {"events": []}}, "windows": [
        {"inspection_kind": "spike", "peak_index": 10, "peak_classifier_score": .22},
        {"inspection_kind": "step", "peak_index": 20, "peak_classifier_score": .1}]}
    visual = {"status": "success", "prediction": {"events": [
        event("spike", 10, 10), event("step", 21, 21)]}}
    result = ranked_boundary_review(observation, visual, candidates)
    assert [(e["kind"], e["start"]) for e in result["prediction"]["events"]] == [("spike", 10)]
    observation["displacement_mm"][10] = None
    assert ranked_boundary_review(observation, visual, candidates)["prediction"]["events"] == []
