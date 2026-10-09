"""Frozen scheduling and shared contracts for the local-view evidence experiment."""
from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_visual import parse_activity, parse_stages

ACTIVITY_METHODS = {"activity_global", "activity_local"}
STAGE_METHODS = {"stage_global", "stage_local"}


def request_schedule(case_ids):
    """Keep case order fixed; alternate the first arm within each experiment."""
    schedule = []
    for experiment, stem in (("activity", "locate"), ("stage", "stage")):
        for index, cid in enumerate(sorted(case_ids)):
            arms = ("global", "local") if index % 2 == 0 else ("local", "global")
            for arm in arms:
                schedule.append({"case_id": cid, "method": experiment + "_" + arm,
                                 "request_id": cid + "-" + stem + "-" + arm})
    return schedule


def activity_prediction(payload, observed):
    activity = parse_activity(payload, observed)
    stage = np.where(activity == 0, 0, -1)
    return ActivityPrediction(activity, stage)


def learned_stages(stage_codes, activity, supported):
    """Preserve confirmed movement even where stage features have no support."""
    stage = np.array(stage_codes, dtype=int, copy=True)
    stage[(activity != 1) | ~supported] = -1
    stage[activity == 0] = 0
    prediction = ActivityPrediction(activity.copy(), stage)
    prediction.validate(len(activity))
    return prediction


def execute_packet(transport, packet, observed, images, failures):
    method = packet["method"]
    gate = np.array(packet.get("activity", []), dtype=int)
    try:
        payload = transport.ask(packet["request_id"], packet["prompt"], images)
        if method in ACTIVITY_METHODS:
            return activity_prediction(payload, observed)
        if method not in STAGE_METHODS:
            raise ValueError("Unknown local-view experiment arm")
        return parse_stages(payload, gate, observed)
    except (ValueError, RuntimeError) as error:
        failures[packet["request_id"]] = str(error)
        unknown = np.full(len(observed), -1, dtype=int)
        if method in ACTIVITY_METHODS:
            return ActivityPrediction(unknown.copy(), unknown, status="failed")
        return ActivityPrediction(gate, unknown, stage_status="failed")
