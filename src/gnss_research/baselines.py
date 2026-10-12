"""Library baselines with default decision rules, plus disclosed GNSS features."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks

NUMERICAL_METHODS = ("random_forest", "rf_balanced", "extra_trees", "isolation_forest")


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
        targets.append(np.column_stack([labels[kind][eligible] for kind in EVENT_TYPES]))
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
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesClassifier(random_state=seed)),
        "isolation_forest": make_pipeline(SimpleImputer(), IsolationForest(random_state=seed)),
    }
    for name, model in models.items():
        if name == "isolation_forest":
            model.fit(np.concatenate(backgrounds))
        else:
            model.fit(matrix, target)
    import sklearn.ensemble._iforest as iforest_source

    receipt = {
        "sklearn_version": version("scikit-learn"), "training_stations": provenance,
        "classification_rule": "library predict; probability argmax; no validation/test tuning",
        "parameters": {name: model[-1].get_params() for name, model in models.items()},
        "adjustments": ["random_state fixed for reproducibility", "shared GNSS feature adapter",
                        "training-only median imputation for numerical features",
                        "IsolationForest trained on normal station prefixes",
                        "rf_balanced only: class_weight=balanced; separate from default RF"],
        "paper_reproduction": False,
        "isolation_forest_library_source_sha256": sha256(Path(iforest_source.__file__)),
    }
    return models, receipt


def predict_numerical(model, name: str, observation: dict) -> dict:
    evidence = numerical_evidence(observation)
    if name == "isolation_forest":
        generic = (model.predict(evidence.matrix) == -1) & evidence.observed
        scores = -model.score_samples(evidence.matrix)
        return {"prediction": None, "generic_mask": generic.tolist(),
                "generic_score": scores.tolist(), "task": "generic anomaly only"}
    binary = model.predict(evidence.matrix).astype(bool)
    masks = {kind: binary[:, index] & evidence.observed
             for index, kind in enumerate(EVENT_TYPES)}
    probabilities = model.predict_proba(evidence.matrix)
    positive = []
    for classes, probability in zip(model[-1].classes_, probabilities):
        indices = np.flatnonzero(classes == True)  # noqa: E712
        positive.append(probability[:, indices[0]] if len(indices) else np.zeros(len(binary)))
    generic = np.any(np.column_stack(list(masks.values())), axis=1)
    return {"prediction": masks_to_prediction(masks).model_dump(),
            "generic_mask": generic.tolist(),
            "generic_score": np.max(np.column_stack(positive), axis=1).tolist(),
            "type_scores": {kind: positive[index].tolist()
                            for index, kind in enumerate(EVENT_TYPES)},
            "task": "four nonexclusive observed morphologies"}
