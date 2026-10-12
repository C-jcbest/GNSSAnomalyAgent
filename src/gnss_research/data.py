"""Simple integrated motion plus independent observation artifacts and white noise."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from gnss_research.files import sha256, source_hashes, write_json

SCENARIOS = ("normal", "spike", "step", "transient", "deformation", "mixed")


def simulate(seed: int, station_id: str, scenario: str, days: int = 730,
             background_days: int = 180) -> tuple[dict, dict]:
    if scenario not in SCENARIOS or days < 365 or not 90 <= background_days < days - 120:
        raise ValueError("unsupported scenario or insufficient background/event horizon")
    rng = np.random.default_rng(seed)
    time = np.arange(days)
    noise_scale = rng.uniform(0.5, 1.5, 3) * np.array([1, 1, 2])
    background_rate = rng.uniform(-0.015, 0.015, 3)
    seasonal_amplitude = rng.uniform(0, 2, 3) * noise_scale
    seasonal_phase = rng.uniform(0, 2 * np.pi, 3)
    background = time[:, None] * background_rate
    background += seasonal_amplitude * np.sin(2 * np.pi * time[:, None] / 365.25
                                              + seasonal_phase)
    direction = rng.normal(size=3)
    direction[2] = -abs(direction[2])
    direction /= np.linalg.norm(direction)
    motion = np.zeros((days, 3))
    artifacts = np.zeros((days, 3))
    events = []
    start = int(rng.integers(background_days + 45, days - 100))
    duration = int(rng.integers(60, min(180, days - start) + 1))
    end = min(days - 1, start + duration - 1)
    if scenario in ("deformation", "mixed"):
        # Integrate a speed profile: the residual plateau after arrest is not active motion.
        speed = rng.uniform(0.04, 0.5)
        length = end - start + 1
        first = length // 3
        second = 2 * length // 3
        rate = np.concatenate([
            np.linspace(speed * 0.1, speed, first),
            np.full(second - first, speed),
            np.linspace(speed, speed * 0.1, length - second),
        ])
        increments = np.zeros(days)
        increments[start:end + 1] = rate
        motion = np.cumsum(increments)[:, None] * direction
        phases = [
            {"kind": "acceleration", "start": start, "end": start + first - 1},
            {"kind": "steady", "start": start + first, "end": start + second - 1},
            {"kind": "deceleration", "start": start + second, "end": end},
        ]
        events.append({"kind": "deformation", "start": start, "end": end,
                       "status": "confirmed", "evidence": "simulation mechanism",
                       "phases": phases})
    for kind in ("spike", "step", "transient"):
        if scenario != kind and scenario != "mixed":
            continue
        onset = start
        if scenario == "mixed":
            onset = int(rng.integers(start, end + 1))
        shift = rng.normal(size=3) * noise_scale * rng.uniform(4, 12)
        last = onset
        if kind == "spike":
            artifacts[onset] += shift
        elif kind == "step":
            artifacts[onset:] += shift
        else:
            last = min(days - 1, onset + int(rng.integers(10, 65)))
            artifacts[onset:last + 1] += shift
        events.append({"kind": kind, "start": onset, "end": last, "status": "confirmed",
                       "evidence": "simulation mechanism", "phases": []})
    values = background + motion + artifacts + rng.normal(size=(days, 3)) * noise_scale
    # Missingness is drawn independently of event type; it never supplies an anomaly label.
    missing = rng.random(days) < 0.015
    gap_start = int(rng.integers(background_days, days - 8))
    missing[gap_start:gap_start + int(rng.integers(2, 8))] = True
    observations = [None if missing[i] else row.tolist() for i, row in enumerate(values)]
    observation = {
        "station_id": station_id,
        "dates": [(date(2022, 1, 1) + timedelta(days=i)).isoformat() for i in range(days)],
        "displacement_mm": observations,
        "background_end": background_days - 1,
    }
    truth = {"station_id": station_id, "seed": seed, "scenario": scenario, "events": events,
             "latent_motion_mm": motion.tolist(), "artifact_mm": artifacts.tolist(),
             "missing_indices": np.flatnonzero(missing).tolist(),
             "parameters": {"noise_scale_mm": noise_scale.tolist(),
                            "background_rate_mm_day": background_rate.tolist()}}
    return observation, truth


def generate(out: Path, seed: int, per_split: int, days: int) -> dict:
    if per_split < 6 or per_split % 6:
        raise ValueError("per-split must be a positive multiple of six")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    out.mkdir(parents=True, exist_ok=False)
    streams = np.random.SeedSequence(seed).spawn(per_split * 3)
    records = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for index in range(per_split):
            station_id = f"s{split_index * per_split + index:04d}"
            case_seed = int(streams[split_index * per_split + index].generate_state(1)[0])
            observation, truth = simulate(case_seed, station_id, SCENARIOS[index % 6], days,
                                          background_days=min(365, days // 2))
            observation_path = out / "observations" / f"{station_id}.json"
            truth_path = out / "truth" / f"{station_id}.json"
            write_json(observation_path, observation)
            write_json(truth_path, truth)
            records.append({"station_id": station_id, "split": split,
                            "observation": str(observation_path.relative_to(out)),
                            "observation_sha256": sha256(observation_path),
                            "truth": str(truth_path.relative_to(out)),
                            "truth_sha256": sha256(truth_path)})
    manifest = {"schema": "gnss-research-dataset-v1", "seed": seed, "days": days,
                "origin": "mechanism-inspired simulation, not field measurements",
                "per_split": per_split, "sources": source_hashes(), "records": records,
                "design": "independent stations, simple white noise, normal historical prefix",
                "background_days": min(365, days // 2)}
    write_json(out / "manifest.json", manifest)
    return {"dataset": str(out.resolve()), "stations": len(records)}
