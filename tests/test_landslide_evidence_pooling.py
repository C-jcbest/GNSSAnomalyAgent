import numpy as np

from gnss_sim.landslide_evidence_pooling import continuity_guides, feature_weights


def test_continuity_suppresses_isolated_equal_endpoint_jump():
    smooth = np.repeat(np.arange(13)[None, :, None], 3, axis=2).astype(float)
    jump = np.zeros_like(smooth)
    jump[:, 6:] = 12
    valid = np.ones((1, 12), dtype=bool)
    smooth_guide, smooth_change = continuity_guides(dict(blocks=smooth, edge_available=valid))
    jump_guide, jump_change = continuity_guides(dict(blocks=jump, edge_available=valid))
    assert smooth_guide.sum() > 0
    assert jump_guide.sum() == 0
    np.testing.assert_allclose(smooth_change.sum(), jump_change.sum())
    jump_weights = feature_weights(jump_guide)
    np.testing.assert_array_equal(jump_weights, np.full((1, 8, 8), 1 / 64, dtype=np.float32))
    assert np.isfinite(jump_weights).all()


def test_missing_middle_block_does_not_create_continuity_across_gap():
    blocks = np.zeros((1, 13, 3))
    blocks[:, 4:7] = 10
    blocks[:, 5] = np.nan
    available = np.ones((1, 12), dtype=bool)
    available[:, 4:6] = False
    guide, change = continuity_guides(dict(blocks=blocks, edge_available=available))
    assert guide.sum() == 0
    assert np.isfinite(change).all()
    assert change.sum() == 60


def test_guide_location_control_preserves_weight_multiset():
    guide = np.zeros((2, 3, 8))
    guide[:, 1, 4] = 6
    original = feature_weights(guide)
    permuted = feature_weights(guide, permuted=True)
    np.testing.assert_array_equal(
        np.sort(original.reshape(2, 64), axis=1),
        np.sort(permuted.reshape(2, 64), axis=1),
    )
    np.testing.assert_allclose(original.sum(axis=(1, 2)), 1, rtol=0, atol=2e-7)
    assert not np.array_equal(original, permuted)
    assert np.unravel_index(original[0].argmax(), (8, 8)) == (3, 4)
