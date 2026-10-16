"""Date alignment, unknown masks and locality of visual activity readout."""

import numpy as np
import pytest

from gnss_sim.landslide_temporal_readout import (
    column_calendar,
    column_numeric_input,
    column_supervision,
    temporal_activity_model,
)
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input


def test_column_evidence_keeps_original_dates_and_unavailable_edges():
    matrix = np.arange(120, dtype=float).reshape(60, 2)
    scaling = fit_numeric_scaling(matrix)
    columns = column_numeric_input(matrix, scaling)
    original = numeric_input(matrix, scaling)
    np.testing.assert_array_equal(columns[:, 30], original)
    np.testing.assert_array_equal(columns[20, 33], original[29])
    assert np.all(columns[0, :30, :2] == 0)
    assert np.all(columns[0, :30, 2:] == 1)
    indices, valid = column_calendar(60)
    assert indices[20, 33] == 29
    assert valid[0, 30] and not valid[0, 29]


def test_dense_labels_never_fill_unknown_or_unsupported_original_dates():
    actual = np.zeros(100, dtype=int)
    actual[40:60] = 1
    actual[49] = -1
    support = np.ones(100, dtype=bool)
    support[52] = False
    values, mask = column_supervision(actual, support)
    assert values[40, 30] == 1 and mask[40, 30]
    assert not mask[40, 33]  # Original unknown day49.
    assert not mask[40, 34]  # Original unsupported day52.
    assert not mask[0, 29]  # Outside the original calendar.
    assert not mask[49, 30]
    assert not mask[52, 30]
    assert values[60, 30] == 0 and mask[60, 30]


def test_aligned_center_does_not_read_remote_pixels_and_numeric_control_ignores_images():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    original = torch.zeros(2, 6, 32, 61)
    changed = original.clone()
    changed[:, :, :, :4] = 100
    numbers = torch.ones(2, 61, 36)
    aligned = temporal_activity_model(16, 36)
    assert aligned(original, numbers).shape == (2, 61)
    assert torch.equal(aligned(original, numbers)[:, 30], aligned(changed, numbers)[:, 30])
    numeric = temporal_activity_model(16, 36, readout="numeric")
    assert torch.equal(numeric(original, numbers), numeric(changed, numbers))
    with pytest.raises(ValueError, match="dated numerical"):
        aligned(original, numbers[:, :60])
