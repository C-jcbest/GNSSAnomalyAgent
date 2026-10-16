import numpy as np
from run_landslide_diversity_local_review import assemble


def test_failed_global_call_does_not_destroy_unreviewed_numeric_prior():
    observed = np.ones(60, dtype=bool)
    observed[25:28] = False
    numeric = np.zeros(60, dtype=int)
    numeric[20:40] = 1
    numeric[~observed] = -1
    outputs = assemble(
        np.full(60, -1), "failed", numeric, np.full(60, 2), observed,
        observed, observed, {}, dict(fixed=[], visual_selected=[], numeric_selected=[]),
    )
    assert outputs["direct_visual"]["status"] == "failed"
    for name in ("fixed_numeric", "adaptive_numeric"):
        assert outputs[name]["status"] == "success"
        np.testing.assert_array_equal(outputs[name]["activity"], numeric)
        stage = np.array(outputs[name]["stage"])
        assert np.all(stage[~observed] == -1)
        assert np.all(stage[numeric == 1] == 2)


def test_visual_decisions_use_common_activity_and_separate_stage_support():
    observed = np.ones(64, dtype=bool)
    activity_support = np.zeros(64, dtype=bool)
    activity_support[16:48] = True
    stage_support = np.zeros(64, dtype=bool)
    stage_support[20:44] = True
    outputs = assemble(
        np.ones(64, dtype=int), "success", np.ones(64, dtype=int),
        np.full(64, 1), observed, activity_support, stage_support, {},
        dict(fixed=[], visual_selected=[], numeric_selected=[]),
    )
    for value in outputs.values():
        activity, stage = np.array(value["activity"]), np.array(value["stage"])
        assert np.all(activity[activity_support] == 1)
        assert np.all(activity[~activity_support] == -1)
        assert np.all(stage[stage_support] == 1)
        assert np.all(stage[~stage_support] == -1)
