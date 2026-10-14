import numpy as np

from gnss_research.features import numerical_evidence
from gnss_research.fusion import localized_review
from gnss_research.localization import locate_step


def observation_with_step():
    rng = np.random.default_rng(24)
    time = np.arange(150)
    values = rng.normal(0, 0.1, (150, 3)) + time[:, None] * np.array([.02, -.01, .03])
    values[107:] += [6, -4, 3]
    return {"dates": list(range(150)), "background_end": 79,
            "displacement_mm": values.tolist()}


def test_step_fit_finds_observed_onset_despite_local_trend_and_off_center_candidate():
    observation = observation_with_step()
    evidence = numerical_evidence(observation)
    window = {"id": "view3", "start": 85, "end": 125}
    assert locate_step(evidence, window)["index"] == 107
    observation["displacement_mm"][106] = None
    localized = locate_step(numerical_evidence(observation), window)
    assert localized["index"] != 107  # A missing previous day cannot support an exact transition.
    assert locate_step(evidence, {"id": "edge", "start": 105, "end": 110})["index"] is None


def test_window_confirmation_corrects_date_but_does_not_create_unconfirmed_steps():
    observation = observation_with_step()
    window = {"id": "view3", "start": 85, "end": 125,
              "inspection_kind": "step", "peak_index": 100}
    candidates = {"windows": [window], "numerical_prediction": {"prediction": {"events": []}},
                  "step_localizations": [locate_step(numerical_evidence(observation), window)]}
    decision = {"kind": "step", "start": 108, "end": 108, "status": "confirmed",
                "evidence": "persistent level change", "phases": []}
    visual = {"status": "success", "prediction": {"events": [decision, decision]}}
    result = localized_review(observation, visual, candidates)
    assert [(event["kind"], event["start"]) for event in result["prediction"]["events"]] == [
        ("step", 107)]
    for replacement in ({**decision, "status": "uncertain"},
                        {**decision, "start": 130, "end": 130},
                        {**decision, "kind": "spike"}):
        visual["prediction"]["events"] = [replacement]
        assert localized_review(observation, visual, candidates)["prediction"]["events"] == []
    assert localized_review(observation, {"status": "failed"}, candidates)["status"] == "failed"
