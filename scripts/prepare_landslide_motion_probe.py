"""Create grouped analytic motion probes using the existing GNSS observation errors.

These are controlled mechanism-recovery records, not expert observable labels or
field data. Exact A/S/D refers to the known latent speed, without hydrologic
modulation. Existing sealed test records are never opened.
"""
from __future__ import annotations

import argparse
import csv
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from run_landslide_comparison import (
    ROOT,
    STAGES,
    fit_baselines,
    observation_features,
    predict_baselines,
    save_json,
    sha,
)

from gnss_sim.landslide import LandslideInput, LandslideRequest, simulate_case
from gnss_sim.landslide_diagnostics import derive_motion

FAMILIES = ("quiet", "instrument_step", "weak_constant", "asd",
            "reactivation", "weak_after_strong")
PROTOCOL = {
    "schema_version": "landslide-motion-probe-v1", "date": "2026-10-09",
    "seed": 2026100937, "days": 730, "count": 48,
    "split": {"train": [f"case_{index:04d}" for index in range(1, 25)],
              "validation": [f"case_{index:04d}" for index in range(25, 37)],
              "mechanism_test": [f"case_{index:04d}" for index in range(37, 49)]},
    "families": list(FAMILIES),
    "assignment": "Cycle the six families, four train and two validation/test records per family",
    "reference": "Exact controlled latent activity and speed dynamics; missing dates unscored. "
                 "Not independent expert observable-motion reference. Weak is a separate "
                 "known scenario attribute, not a replacement for A/S/D.",
    "noise": "Reuse stable-case simulation measurement errors, spikes and gaps at a new seed; "
             "replace latent motion only. Fixed directions from the sampled slope orientation.",
    "activity_start": "First nonzero designed latent velocity; no derivative-based boundaries",
    "boundary_scores": "Exact mechanism endpoint errors conditional on matched events; "
                       "unmatched events reported separately",
    "selection": "Agent policy frozen from pilot-v3 before generating these observations. "
                 "No further selection using mechanism-test labels or outputs.",
    "baseline_calibration": "Reuse the complete existing robust/PELT/XGBoost validation grids, "
                            "same train/validation/test split; no reuse of old fitted models",
    "scope": "Independent random streams in a controlled analytic probe family; "
             "not field generalization or independence across real sites",
}


def controlled_case(request, index):
    case, original = simulate_case(request, index, "stable")
    rng = np.random.default_rng(np.random.SeedSequence([request.seed, index, 0xAD09]))
    family = FAMILIES[index % len(FAMILIES)]
    days = request.days
    speed = np.zeros(days)
    stage = np.zeros(days, dtype=int)
    weak = np.zeros(days, dtype=bool)
    episodes = np.full(days, "", dtype=object)
    peak = float(rng.uniform(0.4, 0.9))
    schedule = []

    def add_phase(start, length, code, first, last, episode, is_weak=False):
        stop = min(start + length, days)
        speed[start:stop] = np.linspace(first, last, stop - start)
        stage[start:stop] = code
        episodes[start:stop] = episode
        weak[start:stop] = is_weak
        schedule.append({"start": start, "stop": stop, "stage": code,
                         "initial_speed": first, "final_speed": last,
                         "episode": episode, "weak": is_weak})

    def add_asd(start, lengths, maximum, episode):
        acceleration, steady, deceleration = lengths
        add_phase(start, acceleration, 1, 0.02, maximum, episode)
        start += acceleration
        add_phase(start, steady, 2, maximum, maximum, episode)
        start += steady
        add_phase(start, deceleration, 3, maximum, 0.02, episode)

    jitter = int(rng.integers(-15, 16))
    weak_rate = float(rng.uniform(0.04, 0.09))
    if family == "weak_constant":
        add_phase(200 + jitter, 220, 2, weak_rate, weak_rate, "episode-1", True)
    elif family == "asd":
        add_asd(110 + jitter, (70, 90, 70), peak, "episode-1")
    elif family == "reactivation":
        add_asd(70 + jitter, (55, 60, 55), peak, "episode-1")
        add_asd(460 + jitter, (55, 60, 55), peak * 0.8, "episode-2")
    elif family == "weak_after_strong":
        add_asd(50 + jitter, (60, 80, 60), peak, "episode-1")
        add_phase(460 + jitter, 210, 2, weak_rate, weak_rate, "episode-2", True)
    azimuth = np.deg2rad(original.parameters["azimuth_degrees"])
    dip = np.deg2rad(original.parameters["dip_degrees"])
    direction = np.array([np.cos(azimuth) * np.cos(dip),
                          np.sin(azimuth) * np.cos(dip), -np.sin(dip)])
    latent = np.cumsum(speed[:, None] * direction, axis=0)
    # Preserve the noise/artifact/gap realization. Stable original latent motion is zero.
    if family == "instrument_step":
        step_day = int(rng.integers(200, 500))
        offset = float(rng.uniform(12, 30)) * direction
        latent_artifact = np.zeros_like(latent)
        latent_artifact[step_day:] = offset
    else:
        step_day = None
        latent_artifact = np.zeros_like(latent)
    observed = []
    labels = []
    inverse_stages = {code: name for name, code in STAGES.items()}
    for day, raw in enumerate(case.displacement_mm):
        value = None if raw is None else (np.array(raw) + latent[day] + latent_artifact[day]).tolist()
        observed.append(value)
        available = value is not None
        active = int(speed[day] > 0) if available else -1
        labels.append({"case_id": case.case_id, "day_index": day, "date": str(case.dates[day]),
                       "activity_label": active,
                       "feature": inverse_stages[stage[day]] if active == 1 else "",
                       "episode_id": str(episodes[day]) if active == 1 else "",
                       "weak_signal": int(weak[day] and available)})
    case = LandslideInput(case_id=case.case_id, dates=case.dates,
                         reference_coordinate_mm=case.reference_coordinate_mm,
                         displacement_mm=observed)
    truth = {"scope": "Known controlled mechanism, not observable expert labels",
             "family": family, "schedule": schedule, "speed_mm_day": speed.tolist(),
             "latent_displacement_mm": latent.tolist(), "instrument_step_day": step_day,
             "observation_error_source_seed": original.seed}
    return case, labels, truth


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    # Protocol precedes generation and any access to newly sampled observations.
    save_json(out / "protocol.json", PROTOCOL)
    (out / "source").mkdir()
    for path in (Path(__file__), ROOT / "src/gnss_sim/landslide.py",
                 ROOT / "src/gnss_sim/landslide_diagnostics.py",
                 ROOT / "src/gnss_sim/landslide_baselines.py",
                 ROOT / "scripts/run_landslide_comparison.py"):
        shutil.copyfile(path, out / "source" / path.name)
    request = LandslideRequest(seed=PROTOCOL["seed"], count=PROTOCOL["count"],
                               days=PROTOCOL["days"])
    features, training_reference, all_reference = {}, {}, []
    started = time.monotonic()
    for index in range(PROTOCOL["count"]):
        case, rows, truth = controlled_case(request, index)
        cid = case.case_id
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        reference_folder = out / "mechanism-reference" / cid
        reference_folder.mkdir(parents=True)
        save_json(reference_folder / "truth.json", truth)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2),
                                                         encoding="utf-8")
        features[cid] = observation_features(case, motions)
        all_reference.extend(rows)
        if cid in PROTOCOL["split"]["train"] + PROTOCOL["split"]["validation"]:
            training_reference[cid] = rows
        print(f"Prepared input {index + 1}/{PROTOCOL['count']}", flush=True)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_reference[0]))
        writer.writeheader()
        writer.writerows(all_reference)
    # Selection functions receive only train/validation references. Test references
    # are already saved but are never passed into fitting or parameter selection.
    choices, activity_model, stage_model = fit_baselines(
        features, training_reference, PROTOCOL["split"], out)
    methods = predict_baselines(features, PROTOCOL["split"]["mechanism_test"], choices,
                                activity_model, stage_model)
    save_json(out / "numeric-predictions.json", {
        name: {cid: value.to_dict() for cid, value in records.items()}
        for name, records in methods.items()})
    save_json(out / "manifest.json", {
        "status": "complete", "seconds": time.monotonic() - started,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "split": PROTOCOL["split"],
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
                  if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
