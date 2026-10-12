"""Station-relative numerical evidence. No shared physical displacement threshold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_research.schema import observed_values


@dataclass
class Evidence:
    values: np.ndarray
    observed: np.ndarray
    residual: np.ndarray
    matrix: np.ndarray
    names: list[str]
    background_end: int
    noise: np.ndarray


def numerical_evidence(observation: dict, window: int = 21) -> Evidence:
    values = observed_values(observation)
    observed = np.isfinite(values).all(axis=1)
    end = observation["background_end"]
    if type(end) is not int or not 29 <= end < len(values):
        raise ValueError("background_end must identify an observed historical reference")
    reference = values[:end + 1]
    reference_observed = observed[:end + 1]
    if reference_observed.sum() < 30:
        raise ValueError("at least 30 background observations required")
    # Background trend and annual terms are fit only to the declared normal prefix.
    time = np.arange(len(values))
    design = np.column_stack([np.ones(len(values)), time / 365.25,
                              np.sin(2 * np.pi * time / 365.25),
                              np.cos(2 * np.pi * time / 365.25)])
    coefficients = np.linalg.lstsq(design[:end + 1][reference_observed],
                                  reference[reference_observed], rcond=None)[0]
    residual = values - design @ coefficients
    reference_residual = residual[:end + 1][reference_observed]
    noise = 1.4826 * np.median(np.abs(reference_residual
                                    - np.median(reference_residual, axis=0)), axis=0)
    if np.any(noise <= np.finfo(float).eps):
        raise ValueError("background scale is degenerate; do not invent a scale floor in mm")
    standardized = residual / noise
    names = []
    columns = []
    for axis_index, axis in enumerate(("N", "E", "U")):
        series = standardized[:, axis_index]
        median = np.full(len(values), np.nan)
        slope = np.full(len(values), np.nan)
        spread = np.full(len(values), np.nan)
        support = np.zeros(len(values))
        half = window // 2
        for day in range(len(values)):
            left = max(0, day - half)
            right = min(len(values), day + half + 1)
            local = series[left:right]
            valid = np.isfinite(local)
            support[day] = valid.mean()
            if valid.sum() >= 3:
                median[day] = np.median(local[valid])
                spread[day] = np.median(np.abs(local[valid] - median[day]))
                local_time = np.arange(left, right)[valid] - day
                slope[day] = np.linalg.lstsq(
                    np.column_stack([np.ones(valid.sum()), local_time]),
                    local[valid], rcond=None)[0][1]
        increments = np.concatenate([[np.nan], np.diff(series)])
        acceleration = np.concatenate([[np.nan], np.diff(slope)])
        for name, vector in (("residual", series), ("increment", increments),
                             ("median", median), ("slope", slope),
                             ("slope_change", acceleration), ("spread", spread),
                             ("support", support), ("absolute_increment", np.abs(increments)),
                             ("absolute_slope", np.abs(slope)),
                             ("local_departure", np.abs(series - median))):
            names.append(f"{axis}_{name}")
            columns.append(vector)
    matrix = np.column_stack(columns)
    return Evidence(values, observed, residual, matrix, names, end, noise)


def summarize_blocks(evidence: Evidence, block_days: int = 21) -> list[dict]:
    blocks = []
    for start in range(0, len(evidence.values), block_days):
        end = min(len(evidence.values), start + block_days)
        valid = evidence.observed[start:end]
        record = {"start": start, "end": end - 1, "observed_days": int(valid.sum())}
        if valid.any():
            record["raw_median_mm"] = np.median(evidence.values[start:end][valid], axis=0).tolist()
            record["relative_evidence"] = np.nanmedian(
                evidence.matrix[start:end][valid], axis=0).tolist()
        blocks.append(record)
    return blocks
