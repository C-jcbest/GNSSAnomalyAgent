import csv
import hashlib
import io
import time
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gnss_sim.api import create_app
from gnss_sim.landslide import SCENARIOS, LandslideRequest, allocate_scenarios, simulate_case
from gnss_sim.landslide_store import LandslideStore


def test_reproducible_multiyear_motion_with_separate_observation_errors():
    request = LandslideRequest(seed=20261007)
    scenarios = allocate_scenarios(request)
    assert scenarios.count("mixed_stages") == 20
    assert scenarios.count("stable") == 4
    assert scenarios == allocate_scenarios(request)
    for index, scenario in enumerate(SCENARIOS):
        case, truth = simulate_case(request, index, scenario)
        assert (case, truth) == simulate_case(request, index, scenario)
        assert len(case.dates) == 1095
        assert case.dates[-1] == date(2024, 12, 30)
        assert set(case.model_dump()) == {
            "schema_version", "case_id", "dates", "reference_coordinate_mm", "displacement_mm",
        }
        latent = np.asarray(truth.latent_displacement_mm)
        rate = np.asarray(truth.daily_rate_mm)
        np.testing.assert_allclose(np.linalg.norm(np.diff(latent, axis=0), axis=1),
                                   rate[1:], atol=1e-12)
        assert np.all(rate >= 0)
        assert np.isfinite(latent).all()
        assert len(truth.missing_indices) > 0
        observed_indices = [i for i, row in enumerate(case.displacement_mm) if row is not None]
        assert truth.missing_indices == [i for i, row in enumerate(case.displacement_mm) if row is None]
        expected = (latent + np.asarray(truth.measurement_noise_mm)
                    + np.asarray(truth.observation_artifact_mm))
        np.testing.assert_allclose([case.displacement_mm[i] for i in observed_indices],
                                   expected[observed_indices])
        assert all(0 <= phase.start_index <= phase.end_index < request.days for phase in truth.phases)
        if scenario == "stable":
            assert not latent.any()
        elif scenario == "progressive_acceleration":
            assert rate[-90:].mean() > rate[:90].mean() * 4
        elif scenario == "acceleration_arrest":
            peak = truth.phases[0].end_index
            assert rate[peak - 20:peak + 20].mean() > rate[-90:].mean() * 3
            # Deceleration slows movement; accumulated displacement never resets to zero.
            assert np.linalg.norm(latent[-1]) > np.linalg.norm(latent[peak])
        else:
            assert np.linalg.norm(latent[-1]) > 5
    changed = simulate_case(LandslideRequest(seed=20261008), 1, "slow_creep")
    assert changed != simulate_case(request, 1, "slow_creep")


def test_case_variability_does_not_repeat_an_annual_template():
    request = LandslideRequest(seed=602, days=2192)
    _, first = simulate_case(request, 0, "seasonal_steps")
    _, second = simulate_case(request, 1, "seasonal_steps")
    rainfall = np.asarray(first.rainfall_mm)
    rates = np.asarray(first.daily_rate_mm)
    assert not np.array_equal(rainfall[:365], rainfall[365:730])
    assert abs(rates[:365].sum() - rates[365:730].sum()) > 1
    assert first.daily_rate_mm != second.daily_rate_mm
    assert first.parameters["azimuth_degrees"] != second.parameters["azimuth_degrees"]
    noise = np.asarray(first.measurement_noise_mm)
    assert np.corrcoef(noise[:-1, 0], noise[1:, 0])[0, 1] > 0.1


@pytest.mark.parametrize("days", [730, 1095, 2192])
def test_mixed_stages_cover_record_and_keep_quiet_plateaus(days):
    request = LandslideRequest(seed=20261009, days=days)
    beginnings, endings, observed_kinds = set(), set(), set()
    repeated_activity = 0
    for index, scenario in enumerate(allocate_scenarios(request)):
        if scenario != "mixed_stages":
            continue
        _, truth = simulate_case(request, index, scenario)
        phases = truth.phases
        rate = np.asarray(truth.daily_rate_mm)
        latent = np.asarray(truth.latent_displacement_mm)
        assert phases[0].start_index == 0
        assert phases[-1].end_index == days - 1
        assert all(right.start_index == left.end_index + 1 for left, right in zip(phases, phases[1:]))
        observed_kinds.update(phase.kind for phase in phases)
        beginnings.add(phases[0].kind)
        endings.add(phases[-1].kind)
        if any(phase.kind == "dormant" for phase in phases[:-1]):
            repeated_activity += 1
        for phase in phases:
            if phase.kind in ("stable", "dormant"):
                assert not rate[phase.start_index:phase.end_index + 1].any()
                if phase.end_index > phase.start_index:
                    np.testing.assert_allclose(
                        np.diff(latent[phase.start_index:phase.end_index + 1], axis=0), 0, atol=1e-12)
        assert len(phases) >= 3
    assert {"stable", "creep", "acceleration", "steady_slip", "deceleration", "dormant"} <= observed_kinds
    assert len(beginnings) > 1 and len(endings) > 1
    assert repeated_activity >= 3


def test_longterm_api_persists_batch_isolates_truth_and_exports_gaps(tmp_path):
    with TestClient(create_app(tmp_path / "data", tmp_path / "no-web")) as client:
        response = client.post("/api/landslide-datasets", json={"seed": 99, "count": 6, "days": 730})
        assert response.status_code == 202
        dataset_id = response.json()["dataset_id"]
        base = f"/api/landslide-datasets/{dataset_id}"
        for _ in range(100):
            manifest = client.get(base).json()
            if manifest["status"] in ("complete", "failed"):
                break
            time.sleep(0.02)
        assert manifest["status"] == "complete", manifest
        assert manifest["generated_cases"] == 6
        assert client.get("/api/datasets").json() == []
        assert client.get("/api/landslide-datasets").json()[0] == manifest
        source = tmp_path / "data" / dataset_id / "generator.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["generator_sha256"]

        values = [0.0]
        for summary in manifest["cases"]:
            case_url = f'{base}/cases/{summary["case_id"]}'
            case = client.get(case_url).json()
            assert "scenario" not in case and "phases" not in case
            assert "latent_displacement_mm" not in case
            truth = client.get(case_url + "/truth").json()
            assert len(case["dates"]) == 730
            assert summary["missing_days"] == len(truth["missing_indices"])
            values.extend(value for row in case["displacement_mm"] if row is not None for value in row)
            download = client.get(case_url + "/observations.csv")
            assert "attachment" in download.headers["content-disposition"]
            rows = list(csv.DictReader(io.StringIO(download.text)))
            assert len(rows) == 730
            assert set(rows[0]) == {"date", "N_mm", "E_mm", "U_mm"}
            missing_index = truth["missing_indices"][0]
            assert rows[missing_index]["N_mm"] == ""
        assert manifest["observed_extent_mm"] == [min(values), max(values)]
        for path in ("/api/landslide-datasets/missing", base + "/cases/nope",
                     base + "/cases/case_9999/truth", base + "/cases/case_9999/observations.csv"):
            assert client.get(path).status_code == 404
        for invalid in ({"days": 365}, {"count": 121}, {"seed": -1}, {"seed": 1.5},
                        {"days": 2193}, {"start_date": "9999-12-31"}, {"preset": "old"}):
            assert client.post("/api/landslide-datasets", json={"seed": 99, **invalid}).status_code == 422

    store = LandslideStore(tmp_path / "data")
    try:
        assert store.get_dataset(dataset_id).generated_cases == 6
        assert len(store.get_case(dataset_id, "case_0001").dates) == 730
    finally:
        store.executor.shutdown(wait=True)


def test_date_range_and_lengths_reject_invalid_input():
    request = LandslideRequest(seed=0, days=730, start_date=date(2023, 3, 1))
    case, _ = simulate_case(request, 0, "stable")
    assert date(2024, 2, 29) in case.dates
    with pytest.raises(ValidationError, match="长度"):
        type(case).model_validate({**case.model_dump(), "displacement_mm": []})
    with pytest.raises(ValidationError, match="日期范围"):
        LandslideRequest(seed=0, start_date=date.max)
