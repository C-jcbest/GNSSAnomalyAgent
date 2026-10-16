"""Direct two bounded observations to missed motion and uncertain visual endpoints."""

from itertools import combinations

import numpy as np

from gnss_sim.landslide_evaluation import event_masks, runs


def repair_role_support(initial, conflicts, observed, footprint_days=49):
    """Keep missed-motion cues and restrict negative cues to visual event edges.

    Below-threshold raw motion does not prove a plateau. Its role here is to
    request endpoint inspection, not to erase the interior of a visual event.
    Interior plateaus in an overmerged initial event remain a known limitation.
    No raw cue assigns an activity state; subsequent VLM observations still do.
    """
    if initial.shape != observed.shape or conflicts.shape != observed.shape:
        raise ValueError("Role support must use the original observation calendar")
    if not np.isin(initial, [-1, 0, 1]).all() or np.any(initial[~observed] != -1):
        raise ValueError("Invalid activity state or missing observation decision")
    if conflicts.dtype != bool or observed.dtype != bool or np.any(conflicts & ~observed):
        raise ValueError("Conflicts must contain observed days only")
    if type(footprint_days) is not int or footprint_days < 0:
        raise ValueError("Observation footprint must be a nonnegative integer")
    edges = np.zeros(len(initial), dtype=bool)
    for event in event_masks(initial):
        indices = np.flatnonzero(event & (initial == 1))
        for endpoint in (int(indices[0]), int(indices[-1])):
            left = max(0, endpoint - footprint_days)
            right = min(len(initial), endpoint + footprint_days + 1)
            edges[left:right] = True
    missed = conflicts & (initial != 1)
    boundary_negative = conflicts & (initial == 1) & edges
    return {"missed_motion": missed, "boundary_negative": boundary_negative,
            "visual_edge_neighbourhood": edges & observed,
            "repair_support": missed | boundary_negative}


def select_role_targets(priorities, original, repair_support, minimum_conflict_days, mode):
    """Compare role support with untyped joint placement at identical call budget.

    Joint placement is evaluated separately because the original greedy choice
    can differ from the best disjoint pair even without the new repair roles.
    All windows retain the original admission threshold. This does not create
    new views or allocate more calls; tie breaks prefer the original locations.
    """
    if mode not in {"repair_roles", "untyped_joint"}:
        raise ValueError("Unknown bounded placement policy")
    count = len(original)
    if count == 0:
        return [], []
    if count > 2:
        raise ValueError("Only the original zero, one or two calls are permitted")
    rows = []
    for row in priorities:
        rows.append({**row, "repair_role_days": int(repair_support[row["start"]:row["stop"]].sum())})
    eligible = [row for row in rows if row["conflict_days"] >= minimum_conflict_days]
    original_indices = {row["index"] for row in original}
    candidates = []
    for group in combinations(eligible, count):
        if count == 2 and group[0]["start"] < group[1]["stop"] and group[1]["start"] < group[0]["stop"]:
            continue
        metric = "repair_role_days" if mode == "repair_roles" else "conflict_days"
        score = sum(row[metric] for row in group)
        preserved = sum(row["index"] in original_indices for row in group)
        candidates.append(((-score, -preserved, tuple(row["start"] for row in group)), group))
    if not candidates:
        raise ValueError("Original disjoint budget must remain feasible")
    _, selected = min(candidates, key=lambda value: value[0])
    return rows, [dict(row) for row in selected]


def role_neighbourhood(repair_support, observed, footprint_days=49):
    """Use the existing raw-evidence footprint for the role-specific authority."""
    if repair_support.shape != observed.shape or repair_support.dtype != bool:
        raise ValueError("Repair support must be a boolean observation calendar")
    if np.any(repair_support & ~observed):
        raise ValueError("Missing days cannot support a repair")
    if type(footprint_days) is not int or footprint_days < 0:
        raise ValueError("Observation footprint must be a nonnegative integer")
    neighbourhood = np.zeros(len(observed), dtype=bool)
    for start, stop in runs(repair_support):
        neighbourhood[max(0, start - footprint_days):min(len(observed), stop + footprint_days)] = True
    return neighbourhood & observed
