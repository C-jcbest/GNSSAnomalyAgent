"""Auxiliary quality scoring preserves unknown and raw support denominators."""

import numpy as np
from train_landslide_deformation_quality import quality_query_scores


def test_unsupported_positive_remains_a_miss_and_unknown_reference_is_excluded():
    bank = {
        "quality": np.array([1, 0, 1, -1, 0]),
        "observed": np.array([True, True, True, False, True]),
        "supported": np.array([True, True, False, False, True]),
    }
    result = quality_query_scores({"case": bank}, {"case": np.array([0.9, 0.8, 0.9, 0.9, 0.1])})
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["reference_known_queries"] == 4
    assert result["unknown_predictions_on_known_queries"] == 1
    assert result["coverage"] == 0.75
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_no_predicted_quality_events_does_not_claim_perfect_precision():
    bank = {
        "quality": np.array([1, 0]),
        "observed": np.array([True, True]),
        "supported": np.array([True, True]),
    }
    result = quality_query_scores({"case": bank}, {"case": np.array([0.1, 0.1])})
    assert result["precision"] is None
    assert result["recall"] == result["f1"] == 0
