from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_sim.landslide_background import (
    generate_background,
    initial_ticks,
    local_review_windows,
    mechanism_episodes,
    profile_schedule,
    render_raw_review,
)
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_observation_review import (
    ActivityEpisode,
    ObservationReview,
    ObservedStage,
    StableRange,
    activity_review_sha256,
    compile_observation_review,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "configs/landslide-background-v1.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def background():
    return generate_background(CONFIG, "development", 0, "mixed")


def test_background_reproducibility_and_split_isolation(background):
    case, truth, design = background
    repeated = generate_background(CONFIG, "development", 0, "mixed")
    assert repeated == background
    case.model_dump_json(warnings="error")
    truth.model_dump_json(warnings="error")
    other_case, other_truth, _ = generate_background(CONFIG, "sealed_test", 0, "mixed")
    assert case.displacement_mm != other_case.displacement_mm
    assert truth.measurement_noise_mm != other_truth.measurement_noise_mm
    assert profile_schedule(CONFIG, "development").count("weak_embedded") == 3
    assert design["profile"] == "mixed"


def test_weak_intervention_preserves_noise_quality_and_other_motion(background):
    base_case, base_truth, _ = background
    weak_case, weak_truth, design = generate_background(CONFIG, "development", 0, "weak_embedded")
    assert weak_truth.measurement_noise_mm == base_truth.measurement_noise_mm
    assert weak_truth.observation_artifact_mm == base_truth.observation_artifact_mm
    assert weak_truth.missing_indices == base_truth.missing_indices
    assert weak_truth.phases == base_truth.phases
    start, end = design["selected_episode"]
    episodes = mechanism_episodes(base_truth.phases)
    assert end + 1 - start == max(stop - left for left, stop in episodes)
    weak_rate = np.asarray(weak_truth.daily_rate_mm)
    base_rate = np.asarray(base_truth.daily_rate_mm)
    np.testing.assert_allclose(weak_rate[start:end + 1], base_rate[start:end + 1] * design["gain"])
    np.testing.assert_array_equal(weak_rate[:start], base_rate[:start])
    np.testing.assert_array_equal(weak_rate[end + 1:], base_rate[end + 1:])
    assert 0.025 <= np.max(weak_rate[start:end + 1]) <= 0.075
    assert np.max(weak_rate[start:end + 1]) == pytest.approx(design["target_peak_mm_day"])
    latent = np.asarray(weak_truth.latent_displacement_mm)
    np.testing.assert_allclose(np.linalg.norm(np.diff(latent, axis=0), axis=1), weak_rate[1:],
                               atol=1e-12)
    observed = latent + weak_truth.measurement_noise_mm + weak_truth.observation_artifact_mm
    for day, row in enumerate(weak_case.displacement_mm):
        if row is not None:
            np.testing.assert_array_equal(row, observed[day])
    assert [row is None for row in weak_case.displacement_mm] == [
        row is None for row in base_case.displacement_mm
    ]


def test_nuisance_does_not_create_physical_motion():
    case, truth, design = generate_background(CONFIG, "development", 0, "nuisance")
    assert not np.asarray(truth.latent_displacement_mm).any()
    assert not np.asarray(truth.daily_rate_mm).any()
    assert 8 <= design["amplitude_mm"] <= 24
    assert len(case.displacement_mm) == CONFIG["days"]


def test_fixed_windows_cover_record_without_label_selection():
    windows = local_review_windows(1095, 365, 45)
    assert [window["target"] for window in windows] == [[0, 364], [365, 729], [730, 1094]]
    assert [window["view"] for window in windows] == [[0, 409], [320, 774], [685, 1094]]


def test_new_tick_rule_removes_endpoint_collision_and_preserves_data(background):
    case, _, _ = background
    png, audit = render_raw_review(case, 238, 405, CONFIG["plot"], 30)
    assert png.startswith(b"\x89PNG")
    assert audit["tick_audit"]["ticks"][0] == 238
    assert audit["tick_audit"]["ticks"][-1] == 404
    bounds = audit["tick_audit"]["label_bounds_pixels"]
    assert all(right[0] - left[1] >= 8 for left, right in zip(bounds, bounds[1:]))
    for axis, panel in enumerate(audit["panels"]):
        values = [np.nan if row is None else row[axis] for row in case.displacement_mm[238:405]]
        coordinates = np.column_stack([np.arange(238, 405), values])
        assert panel["data_sha256"] == hashlib.sha256(coordinates.tobytes()).hexdigest()
        assert panel["xlim"] == [238, 404]
        assert panel["ylim"][1] - panel["ylim"][0] >= 15


def test_tick_endpoints_and_validation():
    assert initial_ticks(0, 77, 30) == [0, 30, 60, 76]
    with pytest.raises(ValueError):
        initial_ticks(7, 4, 30)


@pytest.fixture(scope="module")
def diagnostics(background):
    return derive_motion(background[0], 61)


def reviewed_template(case, diagnostics):
    return ObservationReview(
        case_id=case.case_id, input_sha256=diagnostics.input_sha256, raw_image_sha256="raw-hash",
        activity_status="reviewed", reviewer="test reference", provenance="test fixture only",
        episodes=[ActivityEpisode(possible_start=180, confirmed_start=200, confirmed_end=400,
                                  possible_end=420, raw_evidence="sustained change")],
        stable_ranges=[StableRange(start=30, end=160, raw_evidence="stationary raw curve")],
    )


def test_pending_review_cannot_silently_become_negative(background, diagnostics):
    case = background[0]
    review = ObservationReview(case_id=case.case_id, input_sha256=diagnostics.input_sha256,
                               raw_image_sha256="raw-hash")
    with pytest.raises(ValueError, match="Pending"):
        compile_observation_review(case, review, diagnostics, "raw-hash")


def test_review_masks_uncertainty_missing_and_stage_support(background, diagnostics):
    case = background[0]
    review = reviewed_template(case, diagnostics)
    review.stage_activity_sha256 = activity_review_sha256(review)
    review.stages = [ObservedStage(start=210, end=390, feature="acceleration", evidence="curvature")]
    rows = compile_observation_review(case, review, diagnostics, "raw-hash")
    assert rows[0]["activity_label"] == rows[175]["activity_label"] == -1
    assert rows[190]["activity_label"] == rows[410]["activity_label"] == -1
    for day, row in enumerate(rows):
        if case.displacement_mm[day] is None:
            assert row["activity_label"] == -1 and row["feature"] == "unknown"
        if row["feature"] == "acceleration":
            assert row["activity_label"] == 1
            assert diagnostics.velocity_mm_day[day] is not None
            assert diagnostics.tangential_acceleration_mm_day2[day] is not None
    assert any(row["activity_label"] == 0 for row in rows)
    assert any(row["activity_label"] == 1 for row in rows)


@pytest.mark.parametrize("mistake", ["outside_activity", "changed_activity", "overlap", "image", "input"])
def test_review_rejects_invalid_hierarchy_or_identity(background, diagnostics, mistake):
    case = background[0]
    review = reviewed_template(case, diagnostics)
    review.stage_activity_sha256 = activity_review_sha256(review)
    review.stages = [ObservedStage(start=210, end=390, feature="steady_motion", evidence="steady")]
    if mistake == "outside_activity":
        review.stages[0].start = 190
    elif mistake == "changed_activity":
        review.episodes[0].confirmed_start = 220
    elif mistake == "overlap":
        review.stages.append(ObservedStage(start=300, end=380, feature="deceleration", evidence="slow"))
    elif mistake == "image":
        review.raw_image_sha256 = "another-image"
    else:
        review.input_sha256 = "another-input"
    with pytest.raises(ValueError):
        compile_observation_review(case, review, diagnostics, "raw-hash")
