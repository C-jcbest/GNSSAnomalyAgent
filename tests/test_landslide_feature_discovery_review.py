"""Regression checks for the frozen pre-score feature contract."""

import numpy as np
import pytest
from probe_landslide_feature_discovery import compile_reviewed, expert_features, review_definition


def test_overlength_original_is_rejected_without_source_repair():
    code = "def center_change(m):\n" + "".join(f"    value{i} = {i}\n" for i in range(11))
    code += "    return float(np.linalg.norm(m[6] - m[5]))"
    definition = {"name": "center_change", "code": code}
    assert review_definition(definition) == ["exceeds_frozen_12_line_limit"]
    assert definition["code"] == code
    with pytest.raises(ValueError, match="Unreviewed"):
        compile_reviewed({"pool": [definition]})


@pytest.mark.parametrize("code", [
    "def feature(m):\n    import os\n    return 0.0",
    "def feature(m):\n    return open('path')",
    "def feature(m):\n    m[6] = 0\n    return 0.0",
    "def feature(m):\n    return np.random.normal()",
])
def test_external_actions_and_input_writes_never_compile(code):
    with pytest.raises(ValueError):
        compile_reviewed({"pool": [{"name": "feature", "code": code}]})


def test_missing_center_cannot_gain_expert_motion_and_remote_gap_keeps_center_slot():
    blocks = np.arange(13, dtype=float)[None, :, None] * np.array([[[1., 2., 3.]]])
    complete = expert_features(blocks)
    remote_gap = blocks.copy()
    remote_gap[:, 0] = np.nan
    np.testing.assert_allclose(expert_features(remote_gap)[:, :5], complete[:, :5])
    blocks[:, 6] = np.nan
    assert np.isnan(expert_features(blocks)[:, :3]).all()
    assert np.isnan(expert_features(blocks)[:, 4]).all()
    # The end-to-end5->7 displacement does not use slot6 and remains descriptive.
    assert np.isfinite(expert_features(blocks)[:, 3]).all()
