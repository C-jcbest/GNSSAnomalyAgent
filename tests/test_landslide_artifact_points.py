import numpy as np

from gnss_sim.landslide_artifact_points import artifact_point_features


def test_a_spike_has_opposed_side_evidence_and_an_offset_keeps_its_direction():
    spike = np.zeros((60, 3))
    spike[30] = [10, 20, -30]
    offset = np.zeros((60, 3))
    offset[30:] = [10, 20, -30]
    spike_features, _, _ = artifact_point_features(spike, np.ones(3))
    offset_features, _, _ = artifact_point_features(offset, np.ones(3))
    assert np.isclose(spike_features[30, 17], -1)
    assert offset_features[30, 18] > 0
    assert np.allclose(spike_features[30, :3], -spike_features[30, 3:6])
    assert np.array_equal(offset_features[30, 3:6], np.zeros(3))


def test_missing_query_and_insufficient_side_observations_have_no_artifact_support():
    values = np.zeros((60, 3))
    values[20:28] = np.nan
    matrix, observed, supported = artifact_point_features(values, np.ones(3))
    assert not np.any(supported[20:28])
    assert not supported[28]
    assert np.isnan(matrix[~supported]).all()
    assert not np.any(observed[20:28])


def test_translation_does_not_change_artifact_evidence_during_continuous_motion():
    values = np.arange(60)[:, None] * np.array([[0.1, -0.2, 0.05]])
    values[30:] += [10, -15, 5]
    first, _, support = artifact_point_features(values, np.ones(3))
    second, _, _ = artifact_point_features(values + [500, -200, 300], np.ones(3))
    assert np.allclose(first[support], second[support], atol=1e-10, equal_nan=True)
    assert support[30]
