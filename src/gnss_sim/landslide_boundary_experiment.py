"""Frozen-coordinate checks and grid-phase perturbation for visual activity review."""
from __future__ import annotations

from gnss_sim.landslide_confirmation_prompt_experiment import (
    confirmation_prompt,
    parse_confirmation,
    target_blocks,
)
from gnss_sim.landslide_window_experiment import parse_window_activity

ARMS = ("b0_blocks", "b1_endpoints", "b2_phase15")


def boundary_blocks(start: int, stop: int, arm: str) -> list[tuple[int, int]]:
    original = target_blocks(start, stop)
    if arm in ARMS[:2]:
        return original
    if arm != ARMS[2]:
        raise ValueError("Unknown boundary experiment arm")
    edges = [start, *range(start + 15, stop, 30), stop]
    return list(zip(edges[:-1], edges[1:]))


def boundary_prompt(arm: str, start: int, stop: int) -> str:
    blocks = boundary_blocks(start, stop, arm)
    prompt = confirmation_prompt("r2_blocks", start, stop)
    if arm == ARMS[0]:
        return prompt
    if arm == ARMS[2]:
        original = str([[left, right - 1] for left, right in target_blocks(start, stop)])
        shifted = str([[left, right - 1] for left, right in blocks])
        if prompt.count(original) != 1:
            raise ValueError("Frozen block-list anchor changed")
        prompt = prompt.replace(original, shifted)
    # This changes model instructions; the parser still rejects invalid coordinates.
    return prompt + f"""FINAL COORDINATE CHECK FOR THIS TARGET:
The minimum permitted endpoint is {start}. The maximum permitted endpoint is {stop - 1}.
The number {stop} is outside this target and must not occur as an interval endpoint.
All activity and uncertain endpoints use inclusive ORIGINAL day indices, not block
numbers, offsets, or exclusive stop indices. Use the printed axis to locate movement.
If visibly continuing motion reaches the target's last day, its target-limited interval
can end at {stop - 1}; this is an observation-window boundary, not a claim that motion
physically stopped there. Never extend the answer into visible context outside target.
An uncertain boundary remains uncertain; do not stretch motion merely to fill a block.
Before returning JSON, check each endpoint against {start} <= start <= end <= {stop - 1},
check exact block endpoints, and check first_activity_day equals the first activity start.
Activity may start/end INSIDE any block. Choose visible boundaries, not the nearest
block edge. Keep stationary and uncertain days out of confirmed activity. Do not add
JSON fields or report this check as a separate reasoning trace.
"""


def parse_boundary_response(payload, observed, start: int, stop: int, arm: str):
    if arm in ARMS[:2]:
        return parse_confirmation(payload, observed, start, stop, "r2_blocks")
    expected_blocks = boundary_blocks(start, stop, arm)
    expected_keys = {"activity", "uncertain", "evidence", "blocks", "first_activity_day"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Expected activity, uncertain, evidence, blocks and first_activity_day")
    activity = parse_window_activity(
        {key: payload[key] for key in ("activity", "uncertain", "evidence")},
        observed, start, stop,
    )
    first = payload["first_activity_day"]
    expected_first = payload["activity"][0][0] if payload["activity"] else None
    if first != expected_first or (first is not None and type(first) is not int):
        raise ValueError("First activity day contradicts final activity")
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or len(blocks) != len(expected_blocks):
        raise ValueError("Block observations do not cover every fixed target block")
    # The earlier validator is frozen with the phase-0 experiment. Keep the same
    # checks here for the new grid rather than changing historical execution code.
    for block, (left, right) in zip(blocks, expected_blocks):
        if not isinstance(block, dict) or set(block) != {"start", "end", "state", "observation"}:
            raise ValueError("Invalid block observation fields")
        if (type(block["start"]) is not int or type(block["end"]) is not int
                or block["start"] != left or block["end"] != right - 1):
            raise ValueError("Block observation calendar differs from frozen target")
        state = block["state"]
        if not isinstance(state, str) or state not in {"motion", "stationary", "mixed", "uncertain"}:
            raise ValueError("Invalid raw block state")
        observation = block["observation"]
        if not isinstance(observation, str) or not 1 <= len(observation.strip()) <= 400:
            raise ValueError("A short visible block observation is required")
        if state in {"stationary", "uncertain"} and (activity[left:right] == 1).any():
            raise ValueError("Non-motion block contradicts final confirmed activity")
    return activity
