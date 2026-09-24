"""P3 scenario templates; event profiles remain defined in events.py."""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np

from gnss_sim import events
from gnss_sim.schemas import EventSeeds, ScenarioType

AXES = ("N", "E", "U")
TYPE_PRIORITY = {
    "slow_trend": 0, "acceleration": 1, "step": 2,
    "transient_shift": 3, "spike": 4,
}
MAGNITUDES_MM = {
    "spike": (4.5, 4.5, 9.0),
    "step": (3.75, 3.75, 7.5),
    "slow_trend": (4.5, 4.5, 9.0),
    "acceleration": (4.5, 4.5, 9.0),
    "transient_shift": (3.75, 3.75, 7.5),
}
MAKERS = {
    "spike": events.make_spike,
    "step": events.make_step,
    "slow_trend": events.make_slow_trend,
    "acceleration": events.make_acceleration,
    "transient_shift": events.make_transient_shift,
}
DURATIONS = {"spike": 1, "step": 1, "slow_trend": 90, "acceleration": 90, "transient_shift": 14}


def _template(scenario: ScenarioType, rng: np.random.Generator) -> list[str]:
    longterm = "slow_trend" if rng.integers(0, 2) else "acceleration"
    if scenario == "multi_spike":
        return ["spike"] * int(rng.integers(2, 5))
    if scenario == "change_with_local":
        return ["step"] * int(rng.integers(1, 3)) + ["spike"] * int(rng.integers(1, 4))
    if scenario == "temporary_with_local":
        return ["transient_shift"] * int(rng.integers(1, 3)) + ["spike"] * int(rng.integers(1, 4))
    if scenario == "longterm_with_local":
        return [longterm] + ["spike"] * int(rng.integers(1, 4))
    if scenario == "longterm_with_change":
        return [longterm] + ["step"] * int(rng.integers(1, 3)) + ["spike"] * int(rng.integers(1, 3))
    if scenario == "complex_multiaxis":
        return ([longterm] + ["step"] * int(rng.integers(1, 3))
                + ["transient_shift"] * int(rng.integers(1, 3))
                + ["spike"] * int(rng.integers(0, 2)))
    raise ValueError(f"Unknown scenario: {scenario}")


def generate_scenario_events(dates: list[date], scenario: ScenarioType, seeds: EventSeeds):
    position_rng = np.random.default_rng(seeds.position)
    shape_rng = np.random.default_rng(seeds.shape)
    sign_rng = np.random.default_rng(seeds.sign)
    types = _template(scenario, shape_rng)
    counts = Counter(types)
    if not (2 <= len(types) <= 6 and counts["spike"] <= 4 and counts["step"] <= 2
            and counts["transient_shift"] <= 2
            and counts["slow_trend"] + counts["acceleration"] <= 1):
        raise ValueError("P3 template violates cardinality")

    placed: list[tuple[np.ndarray, object]] = []
    long_interval: tuple[int, int] | None = None
    for index, kind in enumerate(types):
        axis = AXES[int(shape_rng.integers(0, 3))]
        if scenario == "complex_multiaxis" and index == 1:
            axis = AXES[(AXES.index(placed[0][1].axis) + 1 + int(shape_rng.integers(0, 2))) % 3]
        duration = DURATIONS[kind]
        earliest = events.SAFE_START
        latest = events.SAFE_END - duration + 1
        if long_interval and kind == "spike" and not any(item.type == "spike" for _, item in placed):
            earliest, latest = long_interval
        if long_interval and scenario == "complex_multiaxis" and kind == "transient_shift" and not any(item.type == "transient_shift" for _, item in placed):
            earliest = long_interval[0]
            latest = min(long_interval[1] - duration + 1, latest)
        candidates = list(range(earliest, latest + 1))
        position_rng.shuffle(candidates)
        start = next((candidate for candidate in candidates if all(
            (kind != "spike" or prior.type != "spike" or abs(candidate - prior.start_index) >= 7)
            and (kind != "step" or prior.type != "step" or abs(candidate - prior.start_index) >= 45)
            and (kind != "transient_shift" or prior.type != "transient_shift"
                 or candidate + duration - 1 < prior.start_index or candidate > prior.end_index)
            for _, prior in placed
        )), None)
        if start is None:
            raise ValueError("No valid event position for P3 scenario")
        magnitude = MAGNITUDES_MM[kind][AXES.index(axis)]
        sign = 1 if sign_rng.integers(0, 2) else -1
        contribution, event = MAKERS[kind](dates, axis, start, sign * magnitude)
        placed.append((contribution, event))
        if kind in ("slow_trend", "acceleration"):
            long_interval = (event.start_index, event.end_index)

    placed.sort(key=lambda item: (item[1].start_index, TYPE_PRIORITY[item[1].type]))
    for index, (_, event) in enumerate(placed, start=1):
        event.event_id = f"event_{index:03d}"
    return placed
