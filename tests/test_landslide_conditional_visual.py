import numpy as np
import pytest

from gnss_sim.landslide_conditional_visual import (
    conditional_visual_features,
    fit_visual_expectation,
)


def test_projection_uses_paired_observations_and_keeps_independent_visual_component():
    rng = np.random.default_rng(91)
    numbers = rng.normal(size=(1000, 2))
    independent = rng.normal(size=1000)
    embeddings = np.column_stack((3 * numbers[:, 0] + 1, independent))
    fitted = fit_visual_expectation(numbers, embeddings)
    residual = conditional_visual_features(embeddings, numbers, fitted, "unnormalized")
    assert np.std(residual[:, 0]) < 0.01
    assert np.std(residual[:, 1]) > 0.9
    shuffled = fit_visual_expectation(numbers, embeddings, shuffled=True)
    shuffled_residual = conditional_visual_features(embeddings, numbers, shuffled, "unnormalized")
    assert np.std(shuffled_residual[:, 0]) > 2.5


def test_train_expectation_is_reused_for_new_observations_without_refitting():
    numbers = np.array([[-1.0], [0.0], [1.0]])
    embeddings = np.column_stack((numbers[:, 0], np.ones(3)))
    fitted = fit_visual_expectation(numbers, embeddings)
    future_numbers = np.array([[4.0], [7.0]])
    future_visual = np.array([[5.0, 2.0], [8.0, 3.0]])
    output = conditional_visual_features(future_visual, future_numbers, fitted, "conditional")
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1)
    assert fitted["samples"] == 3
    assert fitted["intercept"] == [0.0, 1.0]


def test_zero_visual_residual_is_finite_and_stays_zero():
    fitted = dict(coefficient=[[0.0], [0.0]], intercept=[2.0, 3.0], visual_mean=[2.0, 3.0])
    output = conditional_visual_features(
        np.array([[2.0, 3.0]]), np.array([[5.0]]), fitted, "conditional"
    )
    np.testing.assert_array_equal(output, np.zeros((1, 2)))
    assert np.isfinite(output).all()


def test_projection_rejects_unpaired_or_unavailable_training_inputs():
    with pytest.raises(ValueError, match="correspond"):
        fit_visual_expectation(np.ones((3, 2)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        fit_visual_expectation(np.array([[0.0], [np.nan]]), np.ones((2, 4)))
