import numpy as np


def test_all_unknown_prediction_keeps_undefined_precision_and_missed_activity():
    from score_landslide_claim_review import score_target

    from gnss_sim.landslide_evaluation import ActivityPrediction

    rows = [{"activity_label": 1, "feature": "steady_motion", "episode_id": "one"}] * 14
    rows += [{"activity_label": 0, "feature": "", "episode_id": ""}] * 14
    prediction = ActivityPrediction(np.full(28, -1), np.full(28, -1)).to_dict()
    results, summaries, _ = score_target(
        {"q": rows}, {"q": np.ones(28, dtype=bool)}, {"q": np.r_[np.ones(14, dtype=bool), np.zeros(14, dtype=bool)]},
        {"unknown": {"q": prediction}}, {}, {},
    )
    counts = results["unknown"]["aggregate"]["daily"]
    assert counts == {"tp": 0, "fp": 0, "fn": 14, "precision": None, "recall": 0.0, "f1": 0.0}
    assert summaries[0]["coverage"] == 0
    assert summaries[0]["weak_recall"] == 0
