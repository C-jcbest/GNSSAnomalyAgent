"""Author-library PELT with disclosed GNSS costs and morphology proposals.

Segmentation operates on observed displacement residuals, not on derivatives.
Statistical model selection supplies candidates; it is not a physical alarm rule.
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints

from gnss_research.features import Evidence, numerical_evidence
from gnss_research.schema import Prediction, event_masks


class DisplacementLinearCost(BaseCost):
    """Three response axes sharing original, possibly irregular observation times."""

    model = "gnss-linear"
    min_size = 7

    def fit(self, signal):
        self.signal = signal
        time = signal[:, 0]
        values = signal[:, 1:]
        terms = np.column_stack([time, time * time, values, values * time[:, None],
                                 values * values])
        self.sums = np.vstack([np.zeros(terms.shape[1]), np.cumsum(terms, axis=0)])
        return self

    def error(self, start, end):
        count = end - start
        if count < self.min_size:
            raise NotEnoughPoints
        sums = self.sums[end] - self.sums[start]
        sum_time, sum_squared_time = sums[:2]
        sum_values, sum_time_values, sum_squared_values = np.split(sums[2:], 3)
        centered_time = sum_squared_time - sum_time * sum_time / count
        if centered_time <= 0:
            raise NotEnoughPoints
        covariance = sum_time_values - sum_time * sum_values / count
        loss = sum_squared_values - sum_values * sum_values / count
        loss -= covariance * covariance / centered_time
        return float(np.maximum(loss, 0).sum())


def regression(time: np.ndarray, values: np.ndarray, columns: list[np.ndarray]) -> dict:
    design = np.column_stack([np.ones(len(time)), time - time.mean(), *columns])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    loss = float(np.square(values - design @ coefficients).sum())
    # Noise is normalized by the station's normal historical MAD, not fitted on events.
    criterion = loss + design.shape[1] * 3 * np.log(values.size)
    return {"loss": loss, "criterion": float(criterion), "coefficients": coefficients}


def jump_evidence(time: np.ndarray, values: np.ndarray, index: int,
                  left: int, right: int) -> dict:
    keep = (time >= max(left, index - 21)) & (time <= min(right, index + 21))
    local_time, local_values = time[keep], values[keep]
    if (local_time < index).sum() < 5 or (local_time >= index).sum() < 5:
        return {"index": index, "selected": "unsupported"}
    jump = (local_time >= index).astype(float)
    hinge = np.maximum(local_time - index, 0)
    fits = {"linear": regression(local_time, local_values, []),
            "step": regression(local_time, local_values, [jump]),
            "hinge": regression(local_time, local_values, [hinge]),
            "step_hinge": regression(local_time, local_values, [jump, hinge])}
    selected = min(fits, key=lambda name: fits[name]["criterion"])
    return {"index": index, "selected": selected,
            "criteria": {name: fit["criterion"] for name, fit in fits.items()},
            "jump_vector": fits["step"]["coefficients"][-1].tolist()}


def returning_pair(time: np.ndarray, values: np.ndarray, first: int, second: int) -> dict:
    keep = (time >= first - 30) & (time <= second + 30)
    local_time, local_values = time[keep], values[keep]
    if (local_time < first).sum() < 5 or (local_time >= second).sum() < 5:
        return {"paired": False}
    first_jump = (local_time >= first).astype(float)
    second_jump = (local_time >= second).astype(float)
    # Continuous nuisance slope changes allow motion to coexist with an artifact.
    hinges = [np.maximum(local_time - first, 0), np.maximum(local_time - second, 0)]
    independent = regression(local_time, local_values, [*hinges, first_jump, second_jump])
    pulse = regression(local_time, local_values, [*hinges, first_jump - second_jump])
    return {"paired": pulse["criterion"] < independent["criterion"],
            "pulse_criterion": pulse["criterion"],
            "independent_criterion": independent["criterion"]}


def activity_fit(time: np.ndarray, values: np.ndarray, jumps: list[dict],
                 first: int, last: int) -> dict:
    """Compare background-plus-jumps with one bounded displacement trajectory.

    Linear and smoothstep trajectories are generic regression shapes. Neither uses
    the simulator's speed phases or its latent displacement. Jump coefficients are
    nuisance terms so a stationary shifted platform need not become activity.
    """
    design = np.column_stack([np.ones(len(time)), time - time.mean(),
                              *[(time >= item["index"]).astype(float) for item in jumps]])
    basis = np.linalg.qr(design, mode="reduced")[0]
    residual = values - basis @ (basis.T @ values)

    def search(pairs):
        pairs = np.asarray(pairs)
        fraction = np.clip((time[:, None] - pairs[:, 0] + 1)
                           / (pairs[:, 1] - pairs[:, 0] + 1), 0, 1)
        templates = np.column_stack([fraction, fraction * fraction * (3 - 2 * fraction)])
        projected = templates - basis @ (basis.T @ templates)
        norm = np.square(projected).sum(axis=0)
        covariance = projected.T @ residual
        gain = np.square(covariance).sum(axis=1) / np.maximum(norm, np.finfo(float).eps)
        winner = int(gain.argmax())
        pair = pairs[winner % len(pairs)]
        template = templates[:, winner]
        coefficient = covariance[winner] / max(norm[winner], np.finfo(float).eps)
        return {"start": int(pair[0]), "end": int(pair[1]), "gain": float(gain[winner]),
                "shape": "linear" if winner < len(pairs) else "smoothstep",
                "curve": template[:, None] * coefficient}

    grid = sorted(set([*range(first, last + 1, 7), last]))
    coarse = search([(left, right) for left in grid for right in grid if right - left >= 14])
    fine = search([(left, right)
                   for left in range(max(first, coarse["start"] - 7),
                                     min(last, coarse["start"] + 7) + 1)
                   for right in range(max(first, coarse["end"] - 7),
                                      min(last, coarse["end"] + 7) + 1)
                   if right - left >= 14])
    penalty = float(5 * np.log(values.size) + 2 * np.log(2))
    fine.update(selected=fine["gain"] > penalty, criterion_gain=fine["gain"] - penalty,
                complexity_penalty=penalty)
    return fine


def segment_proposals(observation: dict, numerical: dict,
                      evidence: Evidence | None = None) -> dict:
    if evidence is None:
        evidence = numerical_evidence(observation)
    length = len(evidence.values)
    spike_events = [event for event in numerical["prediction"]["events"]
                    if event["kind"] == "spike"]
    eligible = evidence.observed.copy()
    for event in spike_events:
        eligible[event["start"]:event["end"] + 1] = False
    # Retain 90 historical days to distinguish a new level from the reference trajectory.
    eligible[:max(0, evidence.background_end - 89)] = False
    time = np.flatnonzero(eligible)
    standardized = evidence.residual[eligible] / evidence.noise
    signal = np.column_stack([time, standardized])
    penalty = float(6 * np.log(standardized.size))
    ends = rpt.Pelt(custom_cost=DisplacementLinearCost(), min_size=7, jump=1).fit(
        signal).predict(pen=penalty)
    segments = []
    start = 0
    for end in ends:
        local_time, local_values = time[start:end], standardized[start:end]
        linear = regression(local_time, local_values, [])
        constant_loss = float(np.square(local_values - local_values.mean(axis=0)).sum())
        slope_gain = constant_loss - linear["loss"] - 3 * np.log(local_values.size)
        segments.append({"start": int(local_time[0]), "end": int(local_time[-1]),
                         "slope_gain": float(slope_gain),
                         "slope": linear["coefficients"][1].tolist(),
                         "observed_samples": len(local_time)})
        start = end
    jumps = []
    break_positions = [0, *ends]
    for position, end in enumerate(ends[:-1], start=1):
        index = int(time[end])
        if index <= evidence.background_end or not evidence.observed[index - 1]:
            continue
        left = int(time[break_positions[position - 1]])
        right = int(time[break_positions[position + 1] - 1])
        candidates = [jump_evidence(time, standardized, day, left, right)
                      for day in range(max(evidence.background_end + 1, index - 3),
                                       min(length - 1, index + 3) + 1)
                      if evidence.observed[day] and evidence.observed[day - 1]]
        supported = [item for item in candidates if "criteria" in item]
        if not supported:
            continue
        best = min(supported, key=lambda item: min(item["criteria"].values()))
        if best["selected"] in ("step", "step_hinge"):
            if not any(item["index"] == best["index"] for item in jumps):
                jumps.append(best)
    jumps.sort(key=lambda item: item["index"])
    activity = activity_fit(time, standardized, jumps, evidence.background_end + 1, length - 1)
    pairing_values = standardized - activity["curve"] if activity["selected"] else standardized
    activity.pop("curve")
    events = list(spike_events)
    pair_receipts = []
    used = set()
    for position, first in enumerate(jumps):
        if position in used:
            continue
        for second_position in range(position + 1, len(jumps)):
            if second_position in used:
                continue
            second = jumps[second_position]
            if second["index"] - first["index"] > 120:
                break
            receipt = returning_pair(time, pairing_values, first["index"], second["index"])
            pair_receipts.append({"first": first["index"], "second": second["index"], **receipt})
            if receipt["paired"]:
                events.append({"kind": "transient", "start": first["index"],
                               "end": second["index"] - 1, "status": "confirmed",
                               "evidence": "Paired shift-and-return regression wins information criterion",
                               "phases": []})
                used.update([position, second_position])
                break
    events.extend({"kind": "step", "start": item["index"], "end": item["index"],
                   "status": "confirmed", "evidence": "Unpaired segmented displacement level jump",
                   "phases": []} for position, item in enumerate(jumps) if position not in used)
    if activity["selected"]:
        events.append({"kind": "deformation", "start": activity["start"], "end": activity["end"],
                       "status": "confirmed", "evidence": "Bounded raw-displacement trajectory improves information criterion over background plus jumps",
                       "phases": []})
    prediction = Prediction.model_validate({"events": events})
    masks = event_masks(prediction, evidence.observed)
    return {"status": "success", "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None, "request_count": 0,
            "segments": segments, "jumps": jumps, "pair_receipts": pair_receipts,
            "activity_fit": activity,
            "parameters": {"library": "ruptures 1.1.10 Pelt", "cost": "project three-axis linear",
                           "min_observations": 7, "jump": 1, "penalty": penalty,
                           "penalty_rule": "6 log(3n)", "normal_tail_days": 90,
                           "jump_refinement_days": 3, "jump_fit_half_window": 21,
                           "pair_max_separation_days": 120, "pair_context_days": 30,
                           "activity_shapes": ["linear", "smoothstep"],
                           "activity_max_components": 1, "activity_min_duration_days": 15,
                           "activity_grid_days": 7, "activity_refinement_days": 7,
                           "activity_penalty": "5 log(3n) + 2 log(2)"}}
