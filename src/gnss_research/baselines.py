"""Library baselines with default decision rules, plus disclosed GNSS features."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

from gnss_research.features import Evidence, morphology_features, numerical_evidence
from gnss_research.files import read_json, sha256
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks
from gnss_research.temporal import (
    PointIntervalForest,
    TemporalForest,
    bridge_unknown,
    decode_interval,
    interval_prediction,
    learn_transitions,
)

NUMERICAL_METHODS = ("random_forest", "rf_balanced", "extra_trees", "isolation_forest",
                     "rf_gap", "rf_temporal", "rf_independent", "rf_independent_temporal",
                     "rf_morphology", "rf_morphology_temporal", "rf_split", "rf_split_temporal",
                     "segment_numeric")


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    return list(zip(np.flatnonzero(transitions == 1).tolist(),
                    (np.flatnonzero(transitions == -1) - 1).tolist()))


def masks_to_prediction(masks: dict[str, np.ndarray]) -> Prediction:
    events = []
    for kind, mask in masks.items():
        if kind in ("spike", "step"):
            spans = [(int(day), int(day)) for day in np.flatnonzero(mask)]
        else:
            spans = contiguous_runs(mask)
        for start, end in spans:
            events.append({"kind": kind, "start": start, "end": end, "status": "confirmed",
                           "evidence": "default classifier decision on GNSS observation features",
                           "phases": []})
    return Prediction.model_validate({"events": events})


def fit_baselines(dataset: Path, records: list[dict], seed: int) -> tuple[dict, dict]:
    matrices = []
    targets = []
    backgrounds = []
    provenance = []
    sequences = []
    morphology_matrices = []
    for record in records:
        if record["split"] != "train":
            continue
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        evidence = numerical_evidence(observation)
        labels = event_masks(Prediction.model_validate({"events": truth["events"]}),
                             evidence.observed)
        eligible = evidence.observed & (np.arange(len(evidence.values)) > evidence.background_end)
        matrices.append(evidence.matrix[eligible])
        morphology_matrices.append(morphology_features(evidence)[eligible])
        targets.append(np.column_stack([labels[kind][eligible] for kind in EVENT_TYPES]))
        sequences.append((np.column_stack([labels[kind] for kind in EVENT_TYPES]), eligible))
        backgrounds.append(evidence.matrix[:evidence.background_end + 1][
            evidence.observed[:evidence.background_end + 1]])
        provenance.append(record["station_id"])
    if not matrices:
        raise ValueError("no training stations")
    matrix = np.concatenate(matrices)
    target = np.concatenate(targets)
    models = {
        "random_forest": make_pipeline(SimpleImputer(), RandomForestClassifier(random_state=seed)),
        "rf_balanced": make_pipeline(SimpleImputer(), RandomForestClassifier(
            random_state=seed, class_weight="balanced")),
        "rf_independent": make_pipeline(SimpleImputer(), MultiOutputClassifier(
            RandomForestClassifier(random_state=seed, class_weight="balanced"))),
        "rf_morphology": make_pipeline(SimpleImputer(), RandomForestClassifier(
            random_state=seed, class_weight="balanced")),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesClassifier(random_state=seed)),
        "isolation_forest": make_pipeline(SimpleImputer(), IsolationForest(random_state=seed)),
    }
    for name, model in models.items():
        if name == "isolation_forest":
            model.fit(np.concatenate(backgrounds))
        elif name == "rf_morphology":
            model.fit(np.concatenate(morphology_matrices), target)
        else:
            model.fit(matrix, target)
    import sklearn.ensemble._iforest as iforest_source

    receipt = {
        "sklearn_version": version("scikit-learn"), "training_stations": provenance,
        "classification_rule": "library predict; probability argmax; no validation/test tuning",
        "parameters": {name: model[-1].get_params() for name, model in models.items()
                       if name != "rf_independent"},
        "adjustments": ["random_state fixed for reproducibility", "shared GNSS feature adapter",
                        "training-only median imputation for numerical features",
                        "IsolationForest trained on normal station prefixes",
                        "rf_balanced: class_weight=balanced; derivatives inherit it, "
                        "separate from default RF"],
        "paper_reproduction": False,
        "isolation_forest_library_source_sha256": sha256(Path(iforest_source.__file__)),
    }
    receipt["parameters"]["rf_independent"] = {
        "wrapper": "sklearn.multioutput.MultiOutputClassifier", "n_jobs": None,
        "binary_estimator": models["rf_independent"][-1].estimator.get_params()}
    transitions, initial = learn_transitions(sequences)
    models["rf_gap"] = models["rf_balanced"]
    models["rf_temporal"] = TemporalForest(models["rf_balanced"], transitions, initial)
    models["rf_independent_temporal"] = TemporalForest(models["rf_independent"], transitions,
                                                       initial)
    models["rf_morphology_temporal"] = TemporalForest(models["rf_morphology"], transitions, initial)
    models["rf_split"] = PointIntervalForest(models["rf_morphology"], models["rf_balanced"], False)
    models["rf_split_temporal"] = PointIntervalForest(models["rf_morphology"],
                                                     models["rf_temporal"], True)
    receipt["adjustments"].append(
        "rf_independent: four independent balanced binary RFs via MultiOutputClassifier; "
        "same features and per-type predict; avoids multioutput class-weight products")
    receipt["adjustments"].append(
        "rf_morphology: balanced RF, 30 original + 18 short-context features; next increment, "
        "neighbor departure, increment reversal product, before/after median contrast, "
        "current minus before/after median; 5-day windows, all station normalized")
    receipt["adjustments"].append(
        "rf_split: point types from 48-feature RF, interval types from 30-feature RF; "
        "rf_split_temporal adds identical interval decoder; no extra training or vote cutoff")
    receipt["temporal_decoder"] = {
        "transitions": {kind: value.tolist() for kind, value in transitions.items()},
        "initial": {kind: value.tolist() for kind, value in initial.items()},
        "adjustments": ["same balanced RF, no retraining", "Laplace pseudocount=1",
                        "transient/deformation: training-only Viterbi transition prior",
                        "spike/step: unchanged library predict",
                        "missing: neutral emission, excluded from all exported labels"],
        "probability_note": "RF posteriors are heuristic emissions, not a generative HMM fit",
    }
    return models, receipt


def predict_numerical(model, name: str, observation: dict,
                      evidence: Evidence | None = None) -> dict:
    if evidence is None:
        evidence = numerical_evidence(observation)
    if name == "segment_numeric":
        from gnss_research.segmentation import segment_proposals

        spike_source = predict_numerical(model, "rf_split_temporal", observation, evidence)
        return segment_proposals(observation, spike_source, evidence)
    if isinstance(model, PointIntervalForest):
        points = predict_numerical(model.point_model, "rf_morphology", observation, evidence)
        interval_name = "rf_temporal" if model.temporal else "rf_balanced"
        intervals = predict_numerical(model.interval_model, interval_name, observation, evidence)
        events = []
        type_scores = {}
        for kind in EVENT_TYPES:
            source = points if kind in ("spike", "step") else intervals
            events.extend(event for event in source["prediction"]["events"]
                          if event["kind"] == kind)
            type_scores[kind] = source["type_scores"][kind]
        prediction = Prediction.model_validate({"events": events})
        masks = event_masks(prediction, evidence.observed)
        return {"prediction": prediction.model_dump(),
                "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
                "generic_score": np.max(np.column_stack(list(type_scores.values())), axis=1).tolist(),
                "type_scores": type_scores, "task": "point and interval feature separation"}
    if name == "isolation_forest":
        generic = (model.predict(evidence.matrix) == -1) & evidence.observed
        scores = -model.score_samples(evidence.matrix)
        return {"prediction": None, "generic_mask": generic.tolist(),
                "generic_score": scores.tolist(), "task": "generic anomaly only"}
    temporal_model = model if isinstance(model, TemporalForest) else None
    if temporal_model is not None:
        model = temporal_model.forest
    matrix = morphology_features(evidence) if name.startswith("rf_morphology") else evidence.matrix
    binary = model.predict(matrix).astype(bool)
    masks = {kind: binary[:, index] & evidence.observed
             for index, kind in enumerate(EVENT_TYPES)}
    probabilities = model.predict_proba(matrix)
    positive = []
    for classes, probability in zip(model[-1].classes_, probabilities):
        indices = np.flatnonzero(classes == True)  # noqa: E712
        positive.append(probability[:, indices[0]] if len(indices) else np.zeros(len(binary)))
    prediction = masks_to_prediction(masks)
    if name == "rf_gap" or temporal_model is not None:
        latent = {kind: mask.copy() for kind, mask in masks.items()}
        start = evidence.background_end + 1
        for index, kind in enumerate(EVENT_TYPES):
            latent[kind][:start] = False
            if kind in ("spike", "step"):
                continue
            if temporal_model is not None:
                latent[kind][start:] = decode_interval(
                    positive[index][start:], evidence.observed[start:],
                    temporal_model.transitions[kind], temporal_model.initial[kind])
            else:
                latent[kind] = bridge_unknown(latent[kind], evidence.observed)
        prediction = interval_prediction(latent, evidence.observed,
                                         "balanced RF with declared interval decoder")
        masks = event_masks(prediction, evidence.observed)
    generic = np.any(np.column_stack(list(masks.values())), axis=1)
    return {"prediction": prediction.model_dump(),
            "generic_mask": generic.tolist(),
            "generic_score": np.max(np.column_stack(positive), axis=1).tolist(),
            "type_scores": {kind: positive[index].tolist()
                            for index, kind in enumerate(EVENT_TYPES)},
            "task": "four nonexclusive observed morphologies"}
