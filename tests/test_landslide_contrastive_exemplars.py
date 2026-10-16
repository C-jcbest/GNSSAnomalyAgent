"""Observable descriptor and training-example retrieval contracts."""

import numpy as np

from gnss_sim.landslide_contrastive_exemplars import (
    contrastive_neighbors,
    motion_shape,
    shape_distance,
)


def test_calendar_gap_cannot_become_a_new_increment_or_move_center():
    blocks = np.arange(13, dtype=float)[:, None] * [[1., 2., -1.]]
    complete = motion_shape(blocks)
    blocks[2] = np.nan
    missing = motion_shape(blocks)
    assert np.isnan(missing[[1, 2, 12, 13, 14]]).all()
    np.testing.assert_allclose(missing[5:7], complete[5:7])
    blocks[6] = np.nan
    assert np.isinf(shape_distance(motion_shape(blocks), complete[None])).all()


def test_descriptor_ignores_reference_offset_and_coordinate_reflections():
    blocks = np.arange(13, dtype=float)[:, None] ** 2 * [[.1, -.3, .2]]
    np.testing.assert_allclose(motion_shape(blocks), motion_shape(blocks + [20, -70, 15]))
    np.testing.assert_allclose(motion_shape(blocks), motion_shape(blocks * [-1, 1, -1]))


def test_retrieval_keeps_both_classes_and_reports_unavailable_queries():
    query = np.ones(23)
    bank = [{"example_id": "quiet", "center_label": 0, "descriptor": query.tolist()},
            {"example_id": "far_active", "center_label": 1, "descriptor": (query + 2).tolist()},
            {"example_id": "near_active", "center_label": 1, "descriptor": (query + .1).tolist()}]
    neighbors = contrastive_neighbors(query, bank)
    assert neighbors["inactive"]["example_id"] == "quiet"
    assert neighbors["active"]["example_id"] == "near_active"
    assert contrastive_neighbors(np.full(23, np.nan), bank) is None
