"""Competing explanations must preserve physical motion and unobserved samples."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_exp28 import counterfactual_views


def test_temporary_removal_retains_motion_and_independent_step():
    motion = np.arange(80)[:, None] * np.array([.8, -.3, .4])
    motion[40:] += [3., 5., -2.]
    raw = motion.copy()
    offset = np.array([8., -7., 4.])
    raw[20:61] += offset
    original, temporary, permanent, masked = counterfactual_views(raw, {"start": 20, "end": 60}, offset)
    np.testing.assert_allclose(temporary, motion)
    np.testing.assert_array_equal(original, raw)
    np.testing.assert_allclose(permanent[61:], raw[61:] - offset)
    np.testing.assert_array_equal(masked[:20], raw[:20])
    np.testing.assert_array_equal(masked[21:], raw[21:])
    assert np.isnan(masked[20]).all()


def test_hypotheses_preserve_existing_missing_samples_without_mutation():
    raw = np.ones((60, 3))
    raw[25:28] = np.nan
    before = raw.copy()
    views = counterfactual_views(raw, {"start": 15, "end": 40}, np.array([2., 3., -1.]))
    for view in views:
        assert np.isnan(view[25:28]).all()
    np.testing.assert_array_equal(raw, before)
