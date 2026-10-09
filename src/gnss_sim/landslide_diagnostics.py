"""Observation-only, retrospective motion estimates and label-free review figures."""
from __future__ import annotations

import hashlib
import io
from threading import Lock
from typing import Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnss_sim.landslide import LandslideInput
from gnss_sim.schemas import StrictModel, Vector

WindowDays = Literal[31, 61, 91]
_RENDER_LOCK = Lock()


class LandslideDiagnostics(StrictModel):
    schema_version: Literal["landslide-diagnostics-v1"] = "landslide-diagnostics-v1"
    case_id: str
    input_sha256: str
    source: Literal["observations_only"] = "observations_only"
    window_days: WindowDays
    edge_policy: Literal["full_centered_window"] = "full_centered_window"
    valid_counts: list[int]
    fitted_displacement_mm: list[Vector | None]
    velocity_mm_day: list[Vector | None]
    acceleration_mm_day2: list[Vector | None]
    speed_mm_day: list[float | None]
    tangential_acceleration_mm_day2: list[float | None]


def _fit_local_motion(time: np.ndarray, positions: np.ndarray, half_window: int):
    """Robustly fit positions directly; never differentiate noisy daily differences."""
    design = np.column_stack((np.ones(len(time)), time, time ** 2))
    coefficients = np.empty((3, 3))
    velocity_error = np.empty(3)
    for axis in range(3):
        weights = np.ones(len(time))
        observed = positions[:, axis]
        for _ in range(5):
            root_weight = np.sqrt(weights)
            coefficient = np.linalg.lstsq(design * root_weight[:, None],
                                          observed * root_weight, rcond=None)[0]
            residual = observed - design @ coefficient
            scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-9)
            weights = np.minimum(1, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
        # Use the final weights for both coefficients and the display-gating error scale.
        root_weight = np.sqrt(weights)
        coefficients[:, axis] = np.linalg.lstsq(
            design * root_weight[:, None], observed * root_weight, rcond=None)[0]
        covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
        velocity_error[axis] = scale * np.sqrt(covariance[1, 1]) / half_window
    return coefficients, float(np.linalg.norm(velocity_error))


def derive_motion(case: LandslideInput, window_days: WindowDays = 61) -> LandslideDiagnostics:
    if window_days not in (31, 61, 91):
        raise ValueError("辅助估计窗口只能为 31、61 或 91 日")
    days = len(case.dates)
    half_window = window_days // 2
    values = np.array([row if row is not None else (np.nan, np.nan, np.nan)
                       for row in case.displacement_mm])
    observed = np.isfinite(values).all(axis=1)
    valid_counts = np.convolve(observed.astype(int), np.ones(window_days, dtype=int), mode="same")
    fitted: list[Vector | None] = [None] * days
    velocity: list[Vector | None] = [None] * days
    acceleration: list[Vector | None] = [None] * days
    speed: list[float | None] = [None] * days
    tangential: list[float | None] = [None] * days
    offsets = np.arange(-half_window, half_window + 1) / half_window
    for center in range(half_window, days - half_window):
        if not observed[center] or valid_counts[center] < np.ceil(0.75 * window_days):
            continue
        start, stop = center - half_window, center + half_window + 1
        support = observed[start:stop]
        if min(support[:half_window].sum(), support[half_window + 1:].sum()) < 0.6 * half_window:
            continue
        valid_positions = np.flatnonzero(np.r_[True, support, True])
        if np.max(np.diff(valid_positions) - 1) > 7:
            continue
        coefficient, error_scale = _fit_local_motion(offsets[support], values[start:stop][support],
                                                     half_window)
        current_velocity = coefficient[1] / half_window
        current_acceleration = 2 * coefficient[2] / half_window ** 2
        magnitude = float(np.linalg.norm(current_velocity))
        fitted[center] = tuple(coefficient[0])
        velocity[center] = tuple(current_velocity)
        acceleration[center] = tuple(current_acceleration)
        speed[center] = magnitude
        # A heuristic display gate, not a confidence bound: GNSS errors are correlated.
        if magnitude > max(3 * error_scale, 1e-9):
            tangential[center] = float(current_velocity @ current_acceleration / magnitude)
    return LandslideDiagnostics(
        case_id=case.case_id,
        input_sha256=hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest(),
        window_days=window_days, valid_counts=valid_counts.tolist(),
        fitted_displacement_mm=fitted, velocity_mm_day=velocity, acceleration_mm_day2=acceleration,
        speed_mm_day=speed, tangential_acceleration_mm_day2=tangential,
    )


def render_diagnostics(case: LandslideInput, diagnostics: LandslideDiagnostics) -> bytes:
    """Export raw positions and auxiliary estimates, with no scenario or phase labels."""
    days = np.arange(len(case.dates))
    colors = ("#147d72", "#cc6d4b", "#a38a32")
    observed = np.array([row if row is not None else (np.nan,) * 3 for row in case.displacement_mm])
    fitted = np.array([row if row is not None else (np.nan,) * 3
                       for row in diagnostics.fitted_displacement_mm])
    velocity = np.array([row if row is not None else (np.nan,) * 3
                         for row in diagnostics.velocity_mm_day])
    acceleration = np.array([row if row is not None else (np.nan,) * 3
                             for row in diagnostics.acceleration_mm_day2])
    # Matplotlib has shared renderer/font state even when each request owns a Figure.
    with _RENDER_LOCK:
        figure = Figure(figsize=(14, 11), dpi=150, facecolor="white")
        FigureCanvasAgg(figure)
        panels = figure.subplots(4, 1, sharex=True)
        for axis, color in enumerate(colors):
            panels[0].plot(days, observed[:, axis], color=color, alpha=0.4, linewidth=0.65)
            panels[0].plot(days, fitted[:, axis], color=color, linewidth=1.4, label=("N", "E", "U")[axis])
            panels[1].plot(days, velocity[:, axis], color=color, linewidth=1,
                           label=("N", "E", "U")[axis])
            panels[2].plot(days, acceleration[:, axis], color=color, linewidth=1)
        panels[1].plot(days, diagnostics.speed_mm_day, color="#283847", linewidth=1.2,
                       linestyle="--", label="3D speed (positive noise bias near zero)")
        panels[3].plot(days, diagnostics.tangential_acceleration_mm_day2,
                       color="#283847", linewidth=1.1)
        low, high = min(0, float(np.nanmin(observed))), max(0, float(np.nanmax(observed)))
        middle, span = (low + high) / 2, max(60, high - low)
        panels[0].set_ylim(middle - 0.75 * span, middle + 0.75 * span)
        labels = ("Displacement (mm)", "Velocity (mm/day)",
                  "Component acceleration\n(mm/day^2)", "Tangential acceleration\n(mm/day^2)")
        for panel, label in zip(panels, labels):
            panel.set_ylabel(label)
            panel.axhline(0, color="#9ca8a0", linewidth=0.6, linestyle="--")
            panel.grid(alpha=0.18)
            panel.set_xlim(0, len(days) - 1)
        panels[0].legend(loc="upper left", ncol=3)
        panels[1].legend(loc="upper left", ncol=4, fontsize=8)
        ticks = np.unique(np.linspace(0, len(days) - 1, 8, dtype=int))
        panels[-1].set_xticks(ticks, [f"{case.dates[i]}\nDay {i}" for i in ticks], fontsize=9)
        panels[-1].set_xlabel("Observation date / zero-based day index")
        figure.suptitle(f"{case.case_id} | Raw observations + {diagnostics.window_days}-day robust local quadratic fit",
                        fontsize=14)
        figure.text(0.10, 0.02,
                    "Retrospective centered estimates; gaps/unsupported windows omitted. "
                    "No generator labels. Tangential gate is heuristic.\n"
                    f"Canonical input SHA-256: {diagnostics.input_sha256}", fontsize=8, color="#536459")
        figure.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.12, hspace=0.20)
        output = io.BytesIO()
        figure.savefig(output, format="png")
        return output.getvalue()
