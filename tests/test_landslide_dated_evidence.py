import numpy as np

from gnss_sim.landslide_dated_evidence import (
    dated_inputs,
    original_edges,
)
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling


def test_edges_do_not_bridge_an_original_missing_block():
    blocks = np.repeat(np.arange(13)[None, :, None], 3, axis=2).astype(float)
    blocks[0, 4] = np.nan
    evidence = dict(
        blocks=blocks,
        edge_available=np.array([[True] * 3 + [False, False] + [True] * 7]),
    )
    edges = original_edges(evidence).reshape(1, 12, 3)
    assert np.isnan(edges[0, 3:5]).all()
    np.testing.assert_array_equal(edges[0, [2, 5]], np.ones((2, 3)))


def test_order_control_preserves_target_edges_and_information_with_missing_flags():
    blocks = np.repeat((np.arange(13) ** 2)[None, :, None], 3, axis=2).astype(float)
    valid = np.ones((1, 12), dtype=bool)
    valid[0, 2] = False
    evidence = dict(blocks=blocks, edge_available=valid)
    scaling = fit_numeric_scaling(np.repeat(original_edges(evidence), 2, axis=0))
    numbers = np.zeros((1, 78), dtype=np.float32)
    embeddings = np.zeros((1, 512), dtype=np.float32)
    ordered = dated_inputs(evidence, scaling, numbers, embeddings, "center")[:, :72].reshape(12, 6)
    permuted = dated_inputs(evidence, scaling, numbers, embeddings, "permuted")[:, :72].reshape(12, 6)
    np.testing.assert_array_equal(ordered[5:7], permuted[5:7])
    assert sorted(map(tuple, ordered)) == sorted(map(tuple, permuted))
    assert not np.array_equal(ordered, permuted)
    assert np.isfinite(permuted).all()
    assert (permuted[:, 3:] == 1).all(axis=1).sum() == 1


