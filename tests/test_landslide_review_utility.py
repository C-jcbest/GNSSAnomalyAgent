from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_review_utility import (
    correction_utility,
    fit_utility_model,
    predict_utility,
    select_utility_windows,
    window_review_features,
)


def test_unknown_cannot_earn_correct_negative_credit():
    actual = np.array([0, 0, 1, 1, -1])
    initial = np.array([0, 1, 0, 1, -1])
    reviewed = np.array([-1, -1, 1, 1, 1])
    result = correction_utility(actual, initial, reviewed, np.ones(5, dtype=bool))
    assert result == {"eligible_days": 4, "gained_correct": 1, "lost_correct": 1,
                      "net_correct": 0, "utility_fraction": 0.}


def test_selection_is_bounded_disjoint_and_abstains_on_nonpositive_utility():
    windows = [(0, 180), (90, 270), (180, 360), (360, 540)]
    _, selected = select_utility_windows(windows, [.1, .3, .2, -.1], [180] * 4)
    assert [row["index"] for row in selected] == [1]
    _, selected = select_utility_windows(windows, [.1, .3, .2, .1], [180] * 4)
    assert [row["index"] for row in selected] == [1, 3]
    assert not select_utility_windows(windows, [0, -.1, -.2, -.1], [180] * 4)[1]


def test_reference_free_window_features_keep_missing_dates_out_of_denominators():
    feature = SimpleNamespace(observed=np.array([True, True, False, True]),
                              progressive_rate=np.array([.1, 0, np.nan, .1]), signal_to_noise=np.array([3, 0, np.nan, 3]))
    long = {"rate": np.array([.1, 0, np.nan, .1]), "signal_to_noise": np.array([3, 0, np.nan, 3])}
    initial = np.array([0, 1, -1, -1])
    matrix = window_review_features(initial, feature, long, np.array([.8, .2, .5, .3]), [(0, 4)], ReviewPolicy())
    np.testing.assert_allclose(matrix, [[1/3, 1/3, 2/3, 1/3, 2.6/3]])


def test_saved_linear_model_replays_and_visual_ablation_ignores_numerical_columns():
    matrix = np.array([[0, 0, 0, 0, 0], [1, 0, 1, 0, .8], [0, 1, 0, 1, .4], [1, 1, 1, 1, .6]], dtype=float)
    target = np.array([-.2, .2, -.1, .1])
    model = fit_utility_model(matrix, target, [0, 1])
    altered = matrix.copy()
    altered[:, 2:] = 500
    np.testing.assert_allclose(predict_utility(model, matrix), predict_utility(model, altered))
    assert len(model["coefficients"]) == 2
