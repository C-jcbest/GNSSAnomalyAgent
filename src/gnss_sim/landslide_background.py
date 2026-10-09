"""New observation backgrounds and fixed review figures; historical runs stay frozen."""
from __future__ import annotations

import hashlib
import io
from collections import Counter

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnss_sim.landslide import LandslideInput, LandslideRequest, simulate_case


def profile_schedule(config: dict, split: str) -> list[str]:
    profiles = []
    for profile, count in config["profiles_per_split"].items():
        profiles.extend([profile] * count)
    rng = np.random.default_rng(np.random.SeedSequence([config["split_seeds"][split], 0xBACC]))
    rng.shuffle(profiles)
    return profiles


def mechanism_episodes(phases) -> list[tuple[int, int]]:
    """Group designed motion phases, solely to apply the declared intervention."""
    episodes: list[tuple[int, int]] = []
    for phase in phases:
        if phase.kind in {"stable", "dormant"}:
            continue
        start, stop = phase.start_index, phase.end_index + 1
        if episodes and episodes[-1][1] == start:
            episodes[-1] = (episodes[-1][0], stop)
        else:
            episodes.append((start, stop))
    return episodes


def generate_background(config: dict, split: str, index: int, profile: str):
    """Do not resample cases or use visual outcomes to select a motion episode."""
    if profile not in {"mixed", "weak_embedded", "noise_control", "nuisance"}:
        raise ValueError("Unknown background profile")
    request = LandslideRequest(
        seed=config["split_seeds"][split], count=sum(config["profiles_per_split"].values()),
        days=config["days"], start_date=config["start_date"],
    )
    scenario = "mixed_stages" if profile in {"mixed", "weak_embedded"} else "stable"
    case, truth = simulate_case(request, index, scenario)
    latent = np.asarray(truth.latent_displacement_mm)
    noise = np.asarray(truth.measurement_noise_mm)
    artifacts = np.asarray(truth.observation_artifact_mm)
    intervention_rng = np.random.default_rng(
        np.random.SeedSequence([request.seed, index, 0xA11E]),
    )
    design = {"profile": profile, "split": split, "base_seed": truth.seed}
    if profile == "weak_embedded":
        episodes = mechanism_episodes(truth.phases)
        # Longest designed episode, with earliest tie-break, chosen without reading a plot.
        start, stop = max(episodes, key=lambda bounds: bounds[1] - bounds[0])
        rate = np.asarray(truth.daily_rate_mm)
        peak = float(intervention_rng.uniform(*config["weak_peak_mm_day"]))
        gain = peak / float(np.max(rate[start:stop]))
        increments = np.diff(latent, axis=0, prepend=latent[:1])
        increments[start:stop] *= gain
        rate[start:stop] *= gain
        latent = np.cumsum(increments, axis=0)
        truth.daily_rate_mm = rate.tolist()
        design.update(selected_episode=[start, stop - 1], gain=gain, target_peak_mm_day=peak,
                      episode_count=len(episodes))
    elif profile == "nuisance":
        days = np.arange(request.days)
        start = int(intervention_rng.integers(request.days // 5, request.days // 2))
        duration = int(intervention_rng.integers(*config["drift_duration_days"]))
        progress = np.clip((days - start) / duration, 0, 1)
        envelope = progress ** 2 * (3 - 2 * progress)
        variant = profile_schedule(config, split)[:index].count("nuisance") % 3
        if variant == 1:
            recovery = np.clip((days - start - duration - 60) / (duration + 30), 0, 1)
            envelope -= recovery ** 2 * (3 - 2 * recovery)
        direction = intervention_rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        amplitude = float(intervention_rng.uniform(*config["drift_amplitude_mm"]))
        artifacts += envelope[:, None] * amplitude * direction
        if variant == 2:
            step_day = min(start + duration + 70, request.days - 1)
            artifacts[step_day:] += intervention_rng.normal(0, 3, 3)
            noise[start:start + duration] *= 1.8
        design.update(drift_start=start, drift_duration=duration, amplitude_mm=amplitude,
                      variant=variant, attribution="instrumental mechanism; not an observed label")
    truth.latent_displacement_mm = [tuple(row) for row in latent.tolist()]
    truth.measurement_noise_mm = [tuple(row) for row in noise.tolist()]
    truth.observation_artifact_mm = [tuple(row) for row in artifacts.tolist()]
    truth.parameters["background_profile"] = profile
    observed = latent + noise + artifacts
    missing = set(truth.missing_indices)
    case.displacement_mm = [None if day in missing else tuple(row.tolist())
                            for day, row in enumerate(observed)]
    return case, truth, design


def local_review_windows(days: int, target_days: int, context_days: int):
    windows = []
    for start in range(0, days, target_days):
        stop = min(start + target_days, days)
        windows.append({"target": [start, stop - 1],
                        "view": [max(0, start - context_days), min(days, stop + context_days) - 1]})
    return windows


def observation_noise(values: np.ndarray) -> np.ndarray:
    differences = np.diff(values, axis=0)
    centered = differences - np.nanmedian(differences, axis=0)
    return np.maximum(1.4826 * np.nanmedian(np.abs(centered), axis=0) / np.sqrt(2), 0.1)


def initial_ticks(start: int, stop: int, step: int) -> list[int]:
    if not 0 <= start < stop or step < 1:
        raise ValueError("Invalid plot range or tick interval")
    first_multiple = ((start + step - 1) // step) * step
    return sorted({start, stop - 1, *range(first_multiple, stop, step)})


def collision_free_ticks(figure, panel, ticks: list[int], minimum_gap: float) -> dict:
    """Keep both range endpoints; prune interior labels using rendered pixel bounds."""
    kept = ticks.copy()
    removed = []
    while True:
        panel.set_xticks(kept)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = [label.get_window_extent(renderer) for label in panel.get_xticklabels()]
        collision = next((i for i in range(len(bounds) - 1)
                          if bounds[i + 1].x0 - bounds[i].x1 < minimum_gap), None)
        if collision is None:
            return {"ticks": kept, "removed_ticks": removed,
                    "label_bounds_pixels": [[box.x0, box.x1] for box in bounds],
                    "minimum_gap_pixels": minimum_gap}
        if len(kept) <= 2:
            raise ValueError("Endpoint labels collide at this figure size")
        remove_at = collision + 1 if collision + 1 < len(kept) - 1 else collision
        removed.append(kept.pop(remove_at))


def render_raw_review(case: LandslideInput, start: int, stop: int, plot: dict,
                      tick_days: int) -> tuple[bytes, dict]:
    values = np.array([row if row is not None else [np.nan] * 3
                       for row in case.displacement_mm])
    noise = observation_noise(values)
    figure = Figure(figsize=plot["figure_inches"], dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(3, 1, sharex=True)
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.91, hspace=0.17)
    audit = {"case_id": case.case_id, "view": [start, stop - 1], "panels": []}
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        series = values[start:stop, axis]
        panel.plot(np.arange(start, stop), series, linewidth=0.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        finite = series[np.isfinite(series)]
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
            center = (low + high) / 2
            span = max(high - low, plot["noise_span_multiplier"] * noise[axis],
                       plot["minimum_span_mm"]) * plot["total_span_multiplier"]
            panel.set_ylim(center - span / 2, center + span / 2)
        panel.set_xlim(start, stop - 1)
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=0.18)
        coordinates = np.column_stack([np.arange(start, stop), series])
        audit["panels"].append({"xlim": list(panel.get_xlim()), "ylim": list(panel.get_ylim()),
                                "data_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
                                "missing_days": np.flatnonzero(~np.isfinite(series)).tolist()})
    panels[-1].set_xlabel("Original day index; missing observations remain gaps")
    figure.suptitle(f"{case.case_id} | Days {start}..{stop - 1} | Raw only; NO labels")
    audit["tick_audit"] = collision_free_ticks(
        figure, panels[-1], initial_ticks(start, stop, tick_days), plot["tick_label_gap_pixels"],
    )
    output = io.BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue(), audit


def schedule_summary(config: dict) -> dict:
    return {split: dict(Counter(profile_schedule(config, split)))
            for split in config["split_seeds"]}
