"""Training-only transition priors for offline interval decoding, not alarm thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_research.schema import EVENT_TYPES, Prediction


@dataclass
class TemporalForest:
    forest: object
    transitions: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]


@dataclass
class PointIntervalForest:
    point_model: object
    interval_model: object
    temporal: bool


def learn_transitions(sequences: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, dict]:
    transitions = {}
    initial = {}
    for column, kind in enumerate(EVENT_TYPES):
        # One pseudocount per cell is a declared statistical prior, not a physical cutoff.
        counts = np.ones((2, 2))
        state_counts = np.ones(2)
        for labels, eligible in sequences:
            states = labels[:, column].astype(int)
            state_counts += np.bincount(states[eligible], minlength=2)
            adjacent = eligible[:-1] & eligible[1:]
            np.add.at(counts, (states[:-1][adjacent], states[1:][adjacent]), 1)
        transitions[kind] = counts / counts.sum(axis=1, keepdims=True)
        initial[kind] = state_counts / state_counts.sum()
    return transitions, initial


def decode_interval(probability: np.ndarray, observed: np.ndarray,
                    transition: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Viterbi over classifier evidence; missing days have neutral emission evidence."""
    epsilon = np.finfo(float).eps
    positive = np.clip(probability, epsilon, 1 - epsilon)
    emission = np.log(np.column_stack([1 - positive, positive]))
    emission[~observed] = 0
    log_transition = np.log(transition)
    score = np.log(initial) + emission[0]
    predecessors = np.zeros((len(probability), 2), dtype=int)
    for day in range(1, len(probability)):
        alternatives = score[:, None] + log_transition
        predecessors[day] = alternatives.argmax(axis=0)
        score = alternatives.max(axis=0) + emission[day]
    states = np.zeros(len(probability), dtype=bool)
    states[-1] = score.argmax()
    for day in range(len(probability) - 1, 0, -1):
        states[day - 1] = predecessors[day, int(states[day])]
    return states


def bridge_unknown(mask: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Join only missing spans flanked by detections; never bridge an observed negative."""
    from gnss_research.baselines import contiguous_runs

    latent = mask.copy()
    for left, right in contiguous_runs(~observed):
        if left > 0 and right + 1 < len(mask) and mask[left - 1] and mask[right + 1]:
            latent[left:right + 1] = True
    return latent


def interval_prediction(latent: dict[str, np.ndarray], observed: np.ndarray,
                        evidence: str) -> Prediction:
    from gnss_research.baselines import contiguous_runs

    events = []
    for kind, states in latent.items():
        spans = [(int(day), int(day)) for day in np.flatnonzero(states & observed)] \
            if kind in ("spike", "step") else contiguous_runs(states)
        for left, right in spans:
            days = np.flatnonzero(observed[left:right + 1]) + left
            if not len(days):
                continue
            events.append({"kind": kind, "start": int(days[0]), "end": int(days[-1]),
                           "status": "confirmed", "evidence": evidence, "phases": []})
    return Prediction.model_validate({"events": events})
