"""Regression checks for onset-only local trend removal, without provider calls."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_exp27 import trend_vectors

from gnss_research.features import numerical_evidence


def observation_with_motion():
    rng = np.random.default_rng(27)
    values = rng.normal(size=(130, 3))
    days = np.arange(80)[:, None]
    values[50:] = days * np.array([2., -.7, .4])
    values[85:110] += np.array([12., -8., 6.])
    return {"station_id": "trend-check", "background_end": 49, "dates": list(range(130)),
            "displacement_mm": values.tolist()}


def test_local_trend_separates_offset_from_linear_motion():
    observation = observation_with_motion()
    graph = {"jumps": [{"id": "J1", "index": 85}, {"id": "J2", "index": 110}]}
    scale = numerical_evidence(observation).noise
    estimates = trend_vectors(observation, graph)
    np.testing.assert_allclose(np.array(estimates[0]["delta"]) * scale, [12., -8., 6.], atol=1e-9)
    np.testing.assert_allclose(np.array(estimates[1]["delta"]) * scale, [-12., 8., -6.], atol=1e-9)


def test_onset_estimate_does_not_use_return_to_force_a_pair():
    observation = observation_with_motion()
    graph = {"jumps": [{"id": "J1", "index": 85}, {"id": "J2", "index": 110}]}
    before = trend_vectors(observation, graph)[0]["delta"]
    values = np.array(observation["displacement_mm"])
    values[110:] += [30., 12., -40.]
    observation["displacement_mm"] = values.tolist()
    after = trend_vectors(observation, graph)[0]["delta"]
    np.testing.assert_array_equal(before, after)


def test_missing_one_side_leaves_offset_unestimated():
    observation = observation_with_motion()
    for index in range(71, 85):
        observation["displacement_mm"][index] = None
    result = trend_vectors(observation, {"jumps": [{"id": "J1", "index": 85}]})[0]
    assert result["estimable"] is False
    assert result["delta"] == [None, None, None]
    assert result["before_observed"] == 0
