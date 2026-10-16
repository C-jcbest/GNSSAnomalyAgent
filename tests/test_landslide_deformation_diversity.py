"""Check mechanism/reference separation, coexisting labels and calendar identity."""

import numpy as np

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_deformation_diversity import diverse_case, motion_profile


def test_mixed_motion_and_step_are_not_mutually_exclusive():
    case, rows, truth = diverse_case(
        LandslideRequest(seed=2026101247, days=730),
        1000,
        "deformation_step",
        "case_9001",
    )
    day = truth["quality_events"][0]["day"]
    assert truth["speed_mm_day"][day] > 0
    if case.displacement_mm[day] is not None:
        assert rows[day]["activity_label"] == rows[day]["quality_event_label"] == 1
    latent = np.array(truth["latent_displacement_mm"])
    noise = np.array(truth["measurement_noise_mm"])
    artifact = np.array(truth["artifact_mm"])
    for day, observation in enumerate(case.displacement_mm):
        if observation is None:
            assert rows[day]["activity_label"] == rows[day]["quality_event_label"] == -1
        else:
            np.testing.assert_allclose(observation, latent[day] + noise[day] + artifact[day])


def test_offset_platform_never_becomes_deformation_or_repeated_quality_events():
    case, rows, truth = diverse_case(
        LandslideRequest(seed=2026101247, days=730),
        1001,
        "step_only",
        "case_9002",
    )
    assert not np.any(truth["speed_mm_day"])
    assert all(row["activity_label"] in {-1, 0} for row in rows)
    observed_events = {
        event["day"]
        for event in truth["quality_events"]
        if case.displacement_mm[event["day"]] is not None
    }
    assert {row["day_index"] for row in rows if row["quality_event_label"] == 1} == observed_events


def test_same_stream_is_reproducible_without_regeneration_selection():
    request = LandslideRequest(seed=2026101247, days=730)
    first = diverse_case(request, 1002, "deformation_fast", "case_9003")
    second = diverse_case(request, 1002, "deformation_fast", "case_9003")
    assert first[0].model_dump_json() == second[0].model_dump_json()
    assert first[1:] == second[1:]
    period = first[2]["schedule"][0]
    assert 8 <= period["stop"] - period["start"] <= 35


def test_profiles_do_not_require_an_asd_sequence():
    profiles = [motion_profile(np.random.default_rng(seed), 80, 0.3) for seed in range(30)]
    shapes = {shape for _, _, shape in profiles}
    assert shapes == {"constant", "rising", "falling", "hump", "irregular"}
    for speed, _, shape in profiles:
        assert np.all(speed > 0)
        if shape == "rising":
            assert np.all(np.diff(speed) > 0)
        if shape == "falling":
            assert np.all(np.diff(speed) < 0)
