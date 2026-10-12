import numpy as np

from gnss_research.schema import event_masks
from gnss_research.temporal import bridge_unknown, decode_interval, interval_prediction


def test_temporal_activity_stops_when_observed_platform_evidence_persists():
    probabilities = np.r_[np.full(20, .01), np.full(30, .99), np.full(30, .01)]
    observed = np.ones(80, dtype=bool)
    observed[32:35] = False
    latent = decode_interval(probabilities, observed, np.array([[.99, .01], [.02, .98]]),
                             np.array([.8, .2]))
    prediction = interval_prediction({"deformation": latent}, observed, "test evidence")
    assert [(event.start, event.end) for event in prediction.events] == [(20, 49)]
    assert not event_masks(prediction, observed)["deformation"][32:35].any()
    assert not latent[50:].any()


def test_missing_bridge_does_not_connect_an_observed_inactive_day():
    observed = np.array([True, False, False, True, True, True])
    mask = np.array([True, False, False, True, False, True])
    latent = bridge_unknown(mask, observed)
    prediction = interval_prediction({"deformation": latent}, observed, "test evidence")
    assert [(event.start, event.end) for event in prediction.events] == [(0, 3), (5, 5)]
    np.testing.assert_array_equal(event_masks(prediction, observed)["deformation"], mask)


def test_independent_forests_use_only_training_stations_and_preserve_missing_labels(tmp_path):
    import json

    from gnss_research.baselines import fit_baselines, predict_numerical
    from gnss_research.data import generate
    from gnss_research.files import read_json

    generate(tmp_path / "data", 81, 6, 365)
    dataset = tmp_path / "data"
    records = read_json(dataset / "manifest.json")["records"]
    models, receipt = fit_baselines(dataset, records, 81)
    json.dumps(receipt, allow_nan=False)
    assert receipt["training_stations"] == [r["station_id"] for r in records if r["split"] == "train"]
    validation = next(r for r in records if r["split"] == "validation")
    observation = read_json(dataset / validation["observation"])
    result = predict_numerical(models["rf_independent_temporal"], "rf_independent_temporal",
                               observation)
    observed = np.array([row is not None for row in observation["displacement_mm"]])
    assert set(result["type_scores"]) == {"spike", "step", "transient", "deformation"}
    assert all(len(scores) == 365 for scores in result["type_scores"].values())
    assert not np.asarray(result["generic_mask"])[~observed].any()
    combined = predict_numerical(models["rf_split_temporal"], "rf_split_temporal", observation)
    points = predict_numerical(models["rf_morphology"], "rf_morphology", observation)
    intervals = predict_numerical(models["rf_temporal"], "rf_temporal", observation)
    for kind in result["type_scores"]:
        source = points if kind in ("spike", "step") else intervals
        assert combined["type_scores"][kind] == source["type_scores"][kind]


def test_short_context_distinguishes_immediate_spike_rebound_from_persistent_step():
    from gnss_research.features import Evidence, morphology_features

    def features(series):
        values = np.column_stack([series] * 3)
        evidence = Evidence(values, np.ones(len(series), dtype=bool), values,
                            np.zeros((len(series), 30)), [], 10, np.ones(3))
        return morphology_features(evidence)

    spike = np.zeros(50)
    spike[25] = 10
    step = np.zeros(50)
    step[25:] = 10
    a, b = features(spike), features(step)
    assert a[25, 30] == -10 and b[25, 30] == 0
    assert a[25, 32] == -100 and b[25, 32] == 0
    assert a[25, 33] == 0 and b[25, 33] == 10
