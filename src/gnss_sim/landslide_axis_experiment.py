"""Cross block position with tick position without changing plotted observations."""
from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np

from gnss_sim.landslide_boundary_experiment import boundary_prompt, parse_boundary_response

ARMS = ("p0_t0", "p0_t15", "p15_t0", "p15_t15")
FACTORS = {
    "p0_t0": (0, 0), "p0_t15": (0, 15),
    "p15_t0": (15, 0), "p15_t15": (15, 15),
}


def prompt_arm(arm: str) -> str:
    phase, _ = FACTORS[arm]
    return "b1_endpoints" if phase == 0 else "b2_phase15"


def axis_prompt(arm: str, start: int, stop: int) -> str:
    return boundary_prompt(prompt_arm(arm), start, stop)


def parse_axis_response(payload, observed, start: int, stop: int, arm: str):
    return parse_boundary_response(payload, observed, start, stop, prompt_arm(arm))


def image_ticks(start: int, stop: int, phase: int) -> np.ndarray:
    if phase not in (0, 15) or not 0 <= start < stop:
        raise ValueError("Expected a valid plot range and tick phase 0 or 15")
    return np.unique(np.r_[np.arange(start + phase, stop, 30), stop - 1])


def plot_geometry(panels) -> list[dict]:
    """Record data and panel bounds independently of ticks and tick labels."""
    geometry = []
    for panel in panels:
        line = panel.lines[0]
        coordinates = np.column_stack([line.get_xdata(), line.get_ydata()])
        geometry.append({
            "data_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "xlim": list(panel.get_xlim()), "ylim": list(panel.get_ylim()),
            "position": list(panel.get_position().bounds),
            "linewidth": line.get_linewidth(), "color": line.get_color(),
        })
    return geometry


def render_tick_pair(case, noise, start, stop, paths, config):
    # Reproduce the frozen raw-only renderer first. Lay out once with its original
    # ticks; changing ticks afterwards must not alter panel positions or scales.
    values = np.array([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ])
    fig, panels = plt.subplots(3, 1, figsize=config["single_plot_inches"], sharex=True)
    try:
        for axis, name in enumerate(("N", "E", "U")):
            panel = panels[axis]
            series = values[start:stop, axis]
            panel.plot(
                np.arange(start, stop), series, linewidth=0.8,
                color=("#147d72", "#cc6d4b", "#a38a32")[axis],
            )
            finite = series[np.isfinite(series)]
            if len(finite):
                low, high = float(finite.min()), float(finite.max())
                center = (low + high) / 2
                span = max(high - low, 6 * noise[axis], 10)
                panel.set_ylim(center - 0.75 * span, center + 0.75 * span)
            panel.set_xlim(start, stop - 1)
            panel.set_ylabel(name + " displacement (mm)")
            panel.grid(alpha=0.2)
        panels[-1].set_xticks(image_ticks(start, stop, 0))
        panels[-1].set_xlabel("Original day index; missing observations remain gaps")
        fig.suptitle(f"{case.case_id} | Days {start}..{stop - 1} | Raw only; NO labels")
        fig.tight_layout()
        original = plot_geometry(panels)
        audit = {"plot_start": start, "plot_stop": stop, "geometry": original, "ticks": {}}
        for phase in (0, 15):
            ticks = image_ticks(start, stop, phase)
            panels[-1].set_xticks(ticks)
            fig.savefig(paths[phase], dpi=config["single_plot_dpi"])
            if plot_geometry(panels) != original:
                raise ValueError("Changing ticks altered the data, layout or axis bounds")
            audit["ticks"][str(phase)] = ticks.tolist()
        audit["geometry_unchanged"] = True
        return audit
    finally:
        plt.close(fig)
