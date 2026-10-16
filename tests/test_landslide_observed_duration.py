import numpy as np
import pytest

from gnss_sim.landslide_observed_duration import activity_across_missing


def test_short_missing_gap_preserves_fourteen_positive_observations_without_imputation():
    observed = np.ones(17, dtype=bool)
    observed[7:10] = False
    result = activity_across_missing(np.ones(17, dtype=bool), observed, observed)
    assert result.tolist() == [1] * 7 + [-1] * 3 + [1] * 7
    # Thirteen observations plus missing days do not establish duration.
    observed[-1] = False
    result = activity_across_missing(np.ones(17, dtype=bool), observed, observed)
    assert not np.any(result == 1)


@pytest.mark.parametrize("separator", ["negative", "unsupported", "long_missing"])
def test_observed_contradictions_and_long_raw_gaps_separate_short_runs(separator):
    observed = np.ones(22, dtype=bool)
    supported = observed.copy()
    mask = np.ones(22, dtype=bool)
    if separator == "negative":
        mask[7:15] = False
    elif separator == "unsupported":
        supported[7:15] = False
    else:
        observed[7:15] = False
        supported = observed.copy()
    result = activity_across_missing(mask, supported, observed)
    assert not np.any(result == 1)
    assert np.array_equal(result == -1, ~supported)


def test_missing_day_with_claimed_feature_support_is_rejected():
    with pytest.raises(ValueError, match="Missing original"):
        activity_across_missing([True], [True], [False])
