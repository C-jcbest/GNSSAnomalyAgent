from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gnss_sim.api import create_app
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion


def polynomial_case():
    day = np.arange(730, dtype=float)
    observations = np.column_stack((0.0008 * day ** 2 + 0.1 * day,
                                   -0.0004 * day ** 2 - 0.02 * day,
                                   np.full(len(day), 4.0)))
    case = LandslideInput(case_id="case_0001",
                          dates=[date(2022, 1, 1) + timedelta(days=int(i)) for i in day],
                          displacement_mm=observations.tolist())
    return case, day


@pytest.mark.parametrize("window_days", [31, 61, 91])
def test_derivatives_recover_known_motion_with_correct_units_and_edges(window_days):
    case, day = polynomial_case()
    result = derive_motion(case, window_days)
    half = window_days // 2
    velocity = np.asarray(result.velocity_mm_day[half:-half])
    acceleration = np.asarray(result.acceleration_mm_day2[half:-half])
    np.testing.assert_allclose(velocity[:, 0], 0.0016 * day[half:-half] + 0.1, atol=1e-10)
    np.testing.assert_allclose(velocity[:, 1], -0.0008 * day[half:-half] - 0.02, atol=1e-10)
    np.testing.assert_allclose(acceleration[:, 0], 0.0016, atol=1e-10)
    np.testing.assert_allclose(acceleration[:, 1], -0.0008, atol=1e-10)
    np.testing.assert_allclose(velocity[:, 2], 0, atol=1e-10)
    assert result.velocity_mm_day[:half] == [None] * half
    assert result.acceleration_mm_day2[-half:] == [None] * half
    assert result.source == "observations_only"
    expected_tangential = np.sum(velocity * acceleration, axis=1) / np.linalg.norm(velocity, axis=1)
    np.testing.assert_allclose(result.tangential_acceleration_mm_day2[half:-half], expected_tangential)


def test_robust_estimates_resist_isolated_outlier_and_do_not_bridge_long_gaps():
    case, day = polynomial_case()
    case.displacement_mm[300] = (8000, -6000, 5000)
    case.displacement_mm[400:415] = [None] * 15
    result = derive_motion(case, 61)
    for center in (290, 300, 310):
        assert abs(result.velocity_mm_day[center][0] - (0.0016 * day[center] + 0.1)) < 0.002
        assert abs(result.acceleration_mm_day2[center][0] - 0.0016) < 0.0003
    assert result.velocity_mm_day[400:415] == [None] * 15
    assert result.velocity_mm_day[420] is None  # Center exists, but long gap invalidates support.
    assert result.velocity_mm_day[500] is not None
    stable = case.model_copy(update={"displacement_mm": [(1.0, 2.0, 3.0)] * 730})
    assert all(value is None for value in derive_motion(stable).tangential_acceleration_mm_day2)


def test_diagnostics_api_uses_only_observation_file_and_exports_png(tmp_path):
    app = create_app(tmp_path / "data", tmp_path / "no-web")
    with TestClient(app) as client:
        manifest = app.state.landslide_store.generate_sync(LandslideRequest(seed=301, count=1, days=730))
        dataset = tmp_path / "data" / manifest.dataset_id
        (dataset / "cases/case_0001/truth.json").unlink()
        url = f"/api/landslide-datasets/{manifest.dataset_id}/cases/case_0001/diagnostics"
        response = client.get(url)
        assert response.status_code == 200
        diagnostics = response.json()
        assert diagnostics["source"] == "observations_only"
        assert "phases" not in diagnostics and "scenario" not in diagnostics
        assert len(diagnostics["velocity_mm_day"]) == 730
        png = client.get(url + ".png?window_days=31")
        assert png.status_code == 200
        assert png.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert "attachment" in png.headers["content-disposition"]
        assert client.get(url + "?window_days=32").status_code == 422
        assert client.get(url + ".png?window_days=500").status_code == 422
        assert client.get(url.replace("case_0001", "case_9999")).status_code == 404
