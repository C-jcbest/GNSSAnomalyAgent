"""Controlled target wording and block observations for raw-displacement confirmation."""
from __future__ import annotations

from gnss_sim.landslide_window_experiment import REVIEW_TASK, parse_window_activity

ARMS = ("r0_candidate", "r1_neutral", "r2_blocks")
WORDING_CHANGES = (
    ("This is a separate confirmation review of a visual candidate.",
     "This is an independent activity detection task for a target interval."),
    ("First check whether the candidate contains", "First check whether the target interval contains"),
    ("Revise candidate boundaries", "Locate activity boundaries"),
)


def target_blocks(start, stop):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("Invalid target calendar")
    return [(left, min(left + 30, stop)) for left in range(start, stop, 30)]


def confirmation_prompt(arm, start, stop):
    original = REVIEW_TASK.format(start=start, end=stop - 1)
    if arm == "r0_candidate":
        return original
    if arm not in ARMS:
        raise ValueError("Unknown confirmation arm")
    neutral = original
    for previous, replacement in WORDING_CHANGES:
        if neutral.count(previous) != 1:
            raise ValueError("Source confirmation wording changed")
        neutral = neutral.replace(previous, replacement)
    if arm == "r1_neutral":
        return neutral
    instructions = neutral.split("Return ONLY JSON with exactly:")[0]
    blocks = [[left, right - 1] for left, right in target_blocks(start, stop)]
    return instructions + f"""Inspect these fixed target blocks in order, using the SAME raw images:
{blocks}
For every block provide one short, checkable observation of raw displacement and
state motion, stationary, mixed, or uncertain. Motion means continuing directional
accumulation within the block, not merely different levels at the target endpoints.
Mixed means the block contains motion and a plateau or uncertain boundary.
Stationary means the observed block is a plateau/noise; uncertain means activity
cannot be distinguished from drift/noise. Missing observations alone are not motion.
Short final blocks use the available raw context; if insufficient, keep uncertain.
Explicitly locate the earliest confirmed motion INSIDE the target, allowing it to
start after the target begins. Do not carry a late movement backward over an earlier
plateau. Final activity may start/end inside blocks; do not round to block edges.
Keep uncertain boundaries unknown. Inspect all axes, without requiring the same sign.
Do not output stages, derivatives, hidden reasoning, or claims of physical failure.
Stationary blocks must not overlap final activity; uncertain blocks must not overlap
final activity. Block observations are visible evidence summaries, not a thinking trace.
Return ONLY JSON with exactly these five keys:
{{"blocks":[{{"start":0,"end":0,"state":"motion|stationary|mixed|uncertain","observation":"brief visible observation"}}],
"first_activity_day":null,"activity":[[start,end]],"uncertain":[[start,end]],"evidence":"brief overall raw observations"}}.
Use the exact block endpoints listed above, one object per block, in that order.
first_activity_day is the first final activity start, or null if activity is empty.
All interval endpoints are inclusive original integers. Empty interval lists are valid.
No Markdown, extra keys, or invented observations.
"""


def parse_confirmation(payload, observed, start, stop, arm):
    if arm in ARMS[:2]:
        return parse_window_activity(payload, observed, start, stop)
    if arm != "r2_blocks":
        raise ValueError("Unknown confirmation arm")
    expected_keys = {"activity", "uncertain", "evidence", "blocks", "first_activity_day"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Expected activity, uncertain, evidence, blocks and first_activity_day")
    activity = parse_window_activity({key: payload[key] for key in ("activity", "uncertain", "evidence")},
                                     observed, start, stop)
    first = payload["first_activity_day"]
    expected_first = payload["activity"][0][0] if payload["activity"] else None
    if first != expected_first or (first is not None and type(first) is not int):
        raise ValueError("First activity day contradicts final activity")
    blocks = payload["blocks"]
    expected = target_blocks(start, stop)
    if not isinstance(blocks, list) or len(blocks) != len(expected):
        raise ValueError("Block observations do not cover every fixed target block")
    for block, (left, right) in zip(blocks, expected):
        if not isinstance(block, dict) or set(block) != {"start", "end", "state", "observation"}:
            raise ValueError("Invalid block observation fields")
        if (type(block["start"]) is not int or type(block["end"]) is not int
                or block["start"] != left or block["end"] != right - 1):
            raise ValueError("Block observation calendar differs from frozen target")
        state, observation = block["state"], block["observation"]
        if not isinstance(state, str) or state not in {"motion", "stationary", "mixed", "uncertain"}:
            raise ValueError("Invalid raw block state")
        if not isinstance(observation, str) or not 1 <= len(observation.strip()) <= 400:
            raise ValueError("A short visible block observation is required")
        if state in {"stationary", "uncertain"} and (activity[left:right] == 1).any():
            raise ValueError("Non-motion block contradicts final confirmed activity")
    return activity
