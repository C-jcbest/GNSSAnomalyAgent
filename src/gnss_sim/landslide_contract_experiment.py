"""Compare redundant activity fields with a single authoritative interval output."""
from __future__ import annotations

from gnss_sim.landslide_boundary_experiment import boundary_prompt, parse_boundary_response
from gnss_sim.landslide_confirmation_prompt_experiment import target_blocks
from gnss_sim.landslide_window_experiment import parse_window_activity

ARMS = ("c0_redundant", "c1_intervals")
CONTRACT_CHANGES = (
    (
        "For every block provide one short, checkable observation of raw displacement and\n"
        "state motion, stationary, mixed, or uncertain.",
        "For every block provide one short, checkable observation of raw displacement,\n"
        "describing motion, stationary, mixed, or uncertain observations in plain text.\n"
        "Do not return a separate block state.",
    ),
    (
        "Stationary blocks must not overlap final activity; uncertain blocks must not overlap\n"
        "final activity.",
        "Final activity must exclude stationary or uncertain dates, including portions\n"
        "inside mixed blocks.",
    ),
    (
        'Return ONLY JSON with exactly these five keys:\n'
        '{"blocks":[{"start":0,"end":0,"state":"motion|stationary|mixed|uncertain",'
        '"observation":"brief visible observation"}],\n'
        '"first_activity_day":null,"activity":[[start,end]],"uncertain":[[start,end]],'
        '"evidence":"brief overall raw observations"}.',
        'Return ONLY JSON with exactly these four keys:\n'
        '{"blocks":[{"start":0,"end":0,"observation":"brief visible observation"}],\n'
        '"activity":[[start,end]],"uncertain":[[start,end]],'
        '"evidence":"brief overall raw observations"}.\n'
        'The final activity list is the only activity-range output. Block observations\n'
        'are evidence summaries; do not output a separate first-activity day.',
    ),
    (
        "first_activity_day is the first final activity start, or null if activity is empty.\n",
        "",
    ),
    (
        "check exact block endpoints, and check first_activity_day equals the first activity start.",
        "and check exact block endpoints.",
    ),
)


def contract_prompt(arm: str, start: int, stop: int) -> str:
    prompt = boundary_prompt("b1_endpoints", start, stop)
    if arm == ARMS[0]:
        return prompt
    if arm != ARMS[1]:
        raise ValueError("Unknown contract experiment arm")
    for previous, replacement in CONTRACT_CHANGES:
        if prompt.count(previous) != 1:
            raise ValueError("Frozen output-contract anchor changed")
        prompt = prompt.replace(previous, replacement)
    return prompt


def parse_contract_response(payload, observed, start: int, stop: int, arm: str):
    if arm == ARMS[0]:
        return parse_boundary_response(payload, observed, start, stop, "b1_endpoints")
    if arm != ARMS[1]:
        raise ValueError("Unknown contract experiment arm")
    if not isinstance(payload, dict) or set(payload) != {"activity", "uncertain", "evidence", "blocks"}:
        raise ValueError("Expected activity, uncertain, evidence and block observations")
    activity = parse_window_activity(
        {key: payload[key] for key in ("activity", "uncertain", "evidence")},
        observed, start, stop,
    )
    blocks = payload["blocks"]
    expected = target_blocks(start, stop)
    if not isinstance(blocks, list) or len(blocks) != len(expected):
        raise ValueError("Block observations do not cover every fixed target block")
    for block, (left, right) in zip(blocks, expected):
        if not isinstance(block, dict) or set(block) != {"start", "end", "observation"}:
            raise ValueError("Invalid block observation fields")
        if (type(block["start"]) is not int or type(block["end"]) is not int
                or block["start"] != left or block["end"] != right - 1):
            raise ValueError("Block observation calendar differs from frozen target")
        observation = block["observation"]
        if not isinstance(observation, str) or not 1 <= len(observation.strip()) <= 400:
            raise ValueError("A short visible block observation is required")
    # Free-text consistency is reviewed separately; no heuristic text parser may
    # silently revise the final ranges or present fewer failures as visual accuracy.
    return activity
