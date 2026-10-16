"""Generate diverse target deformation separately from known observation artifacts.

Reuse the existing GNSS measurement-noise simulator; explicit event profiles
replace its latent motion and artifacts. These are controlled references, not
field labels or independently established visually observable boundaries.
"""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide import LandslideInput, simulate_case
from gnss_sim.landslide_evaluation import STAGES

FAMILIES = (
    "quiet",
    "step_only",
    "spike_only",
    "deformation_single",
    "deformation_fast",
    "deformation_recurrent",
    "deformation_step",
    "deformation_spike",
)


def motion_profile(rng, duration, peak):
    """Allow constant, rising, falling, hump and irregular positive motion.

    A/S/D is optional process information and never determines the event extent.
    No event is required to traverse a fixed acceleration/steady/deceleration
    sequence. Positive samples, rather than a derivative threshold, define the
    designed motion support.
    """
    shape = str(rng.choice(["constant", "rising", "falling", "hump", "irregular"]))
    time = (np.arange(duration) + 0.5) / duration
    if shape == "constant":
        speed = np.full(duration, peak)
    elif shape == "rising":
        speed = peak * (0.15 + 0.85 * time ** rng.uniform(0.7, 2.2))
    elif shape == "falling":
        speed = peak * (0.15 + 0.85 * (1 - time) ** rng.uniform(0.7, 2.2))
    elif shape == "hump":
        speed = peak * (0.1 + 0.9 * np.sin(np.pi * time) ** rng.uniform(0.7, 2.2))
    else:
        knots = int(rng.integers(3, 7))
        amplitudes = rng.uniform(0.15, 1, knots)
        amplitudes /= amplitudes.max()
        speed = peak * np.interp(time, np.linspace(0, 1, knots), amplitudes)
    changes = np.gradient(speed)
    stage = np.full(duration, STAGES["steady_motion"], dtype=int)
    stage[changes > peak * 1e-8] = STAGES["acceleration"]
    stage[changes < -peak * 1e-8] = STAGES["deceleration"]
    return speed, stage, shape


def diverse_case(request, stream_index, family, case_id):
    """Produce independent noise/motion/artifact streams, with no curve selection."""
    if family not in FAMILIES:
        raise ValueError("Unknown deformation-diversity family")
    original, background = simulate_case(request, stream_index, "stable")
    motion_rng = np.random.default_rng(np.random.SeedSequence([request.seed, stream_index, 0xDEFA]))
    quality_rng = np.random.default_rng(
        np.random.SeedSequence([request.seed, stream_index, 0xA117])
    )
    days = request.days
    velocity = np.zeros(days)
    stage = np.zeros(days, dtype=int)
    episode = np.full(days, "", dtype=object)
    schedule = []
    if family.startswith("deformation"):
        if family == "deformation_recurrent":
            first_start = int(motion_rng.integers(40, 140))
            first_duration = int(motion_rng.integers(35, 150))
            second_start = int(
                motion_rng.integers(max(first_start + first_duration + 50, 350), 480)
            )
            periods = [
                (first_start, first_duration),
                (second_start, int(motion_rng.integers(35, 180))),
            ]
        elif family == "deformation_fast":
            periods = [(int(motion_rng.integers(80, days - 80)), int(motion_rng.integers(8, 36)))]
        else:
            duration = int(motion_rng.integers(35, 260))
            periods = [(int(motion_rng.integers(30, days - duration - 30)), duration)]
        for number, (start, duration) in enumerate(periods, 1):
            stop = min(start + duration, days)
            limits = (1.0, 3.5) if family == "deformation_fast" else (0.035, 0.8)
            peak = float(np.exp(motion_rng.uniform(np.log(limits[0]), np.log(limits[1]))))
            speed, phases, shape = motion_profile(motion_rng, stop - start, peak)
            velocity[start:stop] = speed
            stage[start:stop] = phases
            episode[start:stop] = f"episode-{number}"
            schedule.append({"start": start, "stop": stop, "profile": shape, "peak_mm_day": peak})
    azimuth = np.deg2rad(float(background.parameters["azimuth_degrees"]))
    dip = np.deg2rad(float(background.parameters["dip_degrees"]))
    direction = np.array(
        [np.cos(azimuth) * np.cos(dip), np.sin(azimuth) * np.cos(dip), -np.sin(dip)]
    )
    latent = np.cumsum(velocity[:, None] * direction, axis=0)
    artifact = np.zeros((days, 3))
    quality_event = np.zeros(days, dtype=bool)
    quality_kinds = np.full(days, "", dtype=object)
    quality_events = []
    sigma = np.array(background.parameters["white_noise_sigma_mm"])
    if family in {"step_only", "deformation_step"}:
        count = int(quality_rng.integers(1, 3))
        for index in range(count):
            if family == "deformation_step" and index == 0:
                period = schedule[0]
                day = int(quality_rng.integers(period["start"], period["stop"]))
            else:
                day = int(quality_rng.integers(80, days - 80))
            orientation = quality_rng.normal(size=3)
            orientation /= np.linalg.norm(orientation)
            offset = orientation * quality_rng.uniform(8, 40)
            artifact[day:] += offset
            quality_event[day] = True
            quality_kinds[day] = "instrument_offset"
            quality_events.append(
                {"kind": "instrument_offset", "day": day, "offset_mm": offset.tolist()}
            )
    if family in {"spike_only", "deformation_spike"}:
        count = int(quality_rng.integers(2, 7))
        for index in range(count):
            if family == "deformation_spike" and index == 0:
                period = schedule[0]
                day = int(quality_rng.integers(period["start"], period["stop"]))
            else:
                day = int(quality_rng.integers(20, days - 20))
            amplitude = quality_rng.normal(size=3) * sigma * quality_rng.uniform(6, 20)
            artifact[day] += amplitude
            quality_event[day] = True
            quality_kinds[day] = "spike"
            quality_events.append({"kind": "spike", "day": day, "amplitude_mm": amplitude.tolist()})
    # Reuse measurement noise and raw missing calendar, but exclude the original
    # simulator's unlabelled artifacts. Every added artifact now has provenance.
    noise = np.asarray(background.measurement_noise_mm)
    observed = np.array([row is not None for row in original.displacement_mm])
    displacement = latent + noise + artifact
    case = LandslideInput(
        case_id=case_id,
        dates=original.dates,
        reference_coordinate_mm=original.reference_coordinate_mm,
        displacement_mm=[
            row.tolist() if available else None
            for row, available in zip(displacement, observed, strict=True)
        ],
    )
    inverse_stages = {code: name for name, code in STAGES.items()}
    rows = []
    for day in range(days):
        activity = int(velocity[day] > 0) if observed[day] else -1
        rows.append(
            {
                "case_id": case_id,
                "day_index": day,
                "date": str(case.dates[day]),
                "activity_label": activity,
                "feature": inverse_stages[int(stage[day])] if activity == 1 else "",
                "episode_id": str(episode[day]) if activity == 1 else "",
                "weak_signal": int(observed[day] and 0 < velocity[day] < 0.1),
                "quality_event_label": int(quality_event[day]) if observed[day] else -1,
                "quality_kind": str(quality_kinds[day]) if observed[day] else "",
            }
        )
    truth = {
        "scope": "Controlled mechanism reference, not independent visually observable or field truth",
        "family": family,
        "stream_index": stream_index,
        "noise_seed": background.seed,
        "schedule": schedule,
        "quality_events": quality_events,
        "speed_mm_day": velocity.tolist(),
        "latent_displacement_mm": latent.tolist(),
        "measurement_noise_mm": noise.tolist(),
        "artifact_mm": artifact.tolist(),
        "quality_event_indices": np.flatnonzero(quality_event).tolist(),
        "original_missing_indices": np.flatnonzero(~observed).tolist(),
        "quality_definition": "Offset transition/spike at its original day; a permanent shifted platform is not a daily new quality event",
    }
    return case, rows, truth
