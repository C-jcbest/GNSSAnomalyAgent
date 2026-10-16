"""Physical counterexamples and missing-calendar contracts for the pilot evidence."""

import numpy as np
import pytest

from gnss_sim.landslide_nuisance_evidence import nuisance_evidence, record_nuisance_evidence


def observation(ramp=False, step=False):
    time = np.arange(168)
    values = np.zeros((168, 3))
    if ramp:
        values += np.clip((time - 28) / 112, 0, 1)[:, None] * np.array([30, 15, -10])
    if step:
        values += (time >= 84)[:, None] * np.array([60, -30, 20])
    return values


def evidence(values):
    return nuisance_evidence(values, np.ones(3), 0, len(values), 84)


def test_step_alone_is_explained_without_continuous_deformation():
    result = evidence(observation(step=True))
    assert result["null"]["kind"] == "step"
    assert result["deformation_evidence"] < 0


def test_continuous_deformation_survives_a_large_simultaneous_offset():
    mixed = evidence(observation(ramp=True, step=True))
    clean = evidence(observation(ramp=True))
    assert clean["deformation_evidence"] > 0
    assert mixed["deformation"]["kind"] == "step_and_ramp"
    assert mixed["deformation_evidence"] > 0
    assert mixed["deformation_evidence"] > mixed["exclusive_evidence"]


def test_isolated_spike_does_not_create_motion_evidence():
    values = observation()
    values[84] = [1000, -500, 300]
    assert evidence(values)["deformation_evidence"] < 0


def test_long_gap_is_unavailable_and_missing_dates_never_get_evidence():
    values = observation(ramp=True)
    values[80:90] = np.nan
    assert evidence(values)["status"] == "unavailable"
    matrix, _ = record_nuisance_evidence(values, np.ones(3))
    assert np.isnan(matrix[80:90]).all()


def test_translation_does_not_change_evidence():
    values = observation(ramp=True, step=True)
    assert evidence(values)["deformation_evidence"] == pytest.approx(
        evidence(values + [500, -200, 50])["deformation_evidence"], abs=1e-7
    )


def test_invalid_noise_does_not_silently_become_a_confident_fit():
    with pytest.raises(ValueError, match="positive"):
        nuisance_evidence(observation(), np.array([1, 0, 1]), 0, 168, 84)
