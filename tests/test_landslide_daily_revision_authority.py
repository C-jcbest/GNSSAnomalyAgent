import numpy as np
from probe_landslide_daily_revision_authority import authority_outputs


def test_local_positive_outside_conflict_does_not_gain_revision_authority():
    observed = np.ones(80, dtype=bool)
    initial = np.zeros(80, dtype=int)
    conflict = np.zeros(80, dtype=bool)
    conflict[40:60] = True
    reviews = [dict(start=0, stop=80, activity=np.ones(80, dtype=int).tolist(), status="success")]
    outputs, diagnostics, _ = authority_outputs(
        initial, reviews, observed, observed, observed, np.full(80, 2),
        conflict, observed, [dict(start=0, stop=80)],
    )
    activity = np.array(outputs["daily_partial"]["activity"])
    np.testing.assert_array_equal(activity == 1, conflict)
    assert diagnostics["daily_partial"]["propagated_outside_permission"] == 0
    assert np.all(np.array(outputs["whole_partial"]["activity"]) == 1)


def test_successful_abstention_retains_prior_but_failure_stays_unknown():
    observed = np.ones(80, dtype=bool)
    initial = np.zeros(80, dtype=int)
    initial[20:50] = 1
    conflict = initial == 1
    target = [dict(start=0, stop=80)]
    for status in ("success", "failed"):
        reviews = [dict(start=0, stop=80, activity=np.full(80, -1).tolist(), status=status)]
        outputs, _, _ = authority_outputs(
            initial, reviews, observed, observed, observed, np.full(80, 2),
            conflict, observed, target,
        )
        activity = np.array(outputs["daily_partial"]["activity"])
        if status == "success":
            np.testing.assert_array_equal(activity, initial)
        else:
            assert np.all(activity[conflict] == -1)
            assert np.all(activity[~conflict] == initial[~conflict])
