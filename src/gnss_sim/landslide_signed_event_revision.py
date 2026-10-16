"""Authorize local additions by candidate-event evidence and withdrawals by daily conflict."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import event_masks
from gnss_sim.landslide_event_motion_protection import event_motion_support


def candidate_event_authority(case, noise, initial, reviewed, permission, observed, targets):
    """Reuse existing ordered-motion evidence on local proposals, not prior events.

    A supported event must also contain a proposed addition at an original
    conflict date. Evidence authorizes a local answer; it never creates an
    activity decision where that answer is negative, unknown or missing.
    """
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Event authority must use the original activity calendar")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing observations must retain unknown decisions")
    if permission.dtype != bool or permission.shape != observed.shape or np.any(permission & ~observed):
        raise ValueError("Daily conflict permission must contain observed dates only")
    coverage = np.zeros(len(initial), dtype=bool)
    for target in targets:
        start, stop = target["start"], target["stop"]
        if not 0 <= start < stop <= len(initial) or coverage[start:stop].any():
            raise ValueError("Local event targets must be valid and disjoint")
        coverage[start:stop] = True
    proposal = np.where(observed, 0, -1)
    proposal[coverage & observed] = reviewed[coverage & observed]
    ordered_support, details = event_motion_support(case, noise, proposal)
    seeds = permission & (initial != 1) & (proposal == 1)
    seeded = np.zeros(len(initial), dtype=bool)
    for event in event_masks(proposal):
        if np.any(event & seeds):
            seeded |= event
    novel = coverage & observed & (initial != 1) & (reviewed == 1)
    additions = novel & ordered_support & seeded
    unfiltered = novel
    centered = np.zeros(len(initial), dtype=bool)
    for target in targets:
        start, stop = target["start"], target["stop"]
        available = np.flatnonzero(novel[start:stop]) + start
        budget = int(additions[start:stop].sum())
        midpoint = (start + stop - 1) / 2
        order = np.lexsort((available, np.abs(available - midpoint)))
        centered[available[order[:budget]]] = True
    diagnostics = dict(
        proposed_addition_days=int(novel.sum()),
        seeded_event_days=int((novel & seeded).sum()),
        supported_addition_days=int(additions.sum()),
        centered_addition_days=int(centered.sum()),
        event_checks=details,
    )
    return additions, unfiltered, centered, coverage & observed, diagnostics


def revise_with_signed_authority(initial, reviewed, observed, withdrawal_permission, addition_permission):
    """Keep a prior unless its directional authority permits the local revision.

    A failed local answer may change a permitted prior to unknown. Permission
    to add never turns an unknown local answer into activity. Successful
    abstention retention is handled by the caller's existing partial compiler.
    """
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Signed revision needs aligned activity states")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing observations cannot gain definite revisions")
    for permission in (withdrawal_permission, addition_permission):
        if permission.shape != observed.shape or permission.dtype != bool or np.any(permission & ~observed):
            raise ValueError("Revision permissions must contain observed dates only")
    activity = initial.copy()
    withdraw = withdrawal_permission & (initial == 1) & (reviewed != 1)
    add = addition_permission & (initial != 1) & (reviewed == 1)
    unresolved = withdrawal_permission & (initial != 1) & (reviewed == -1)
    activity[withdraw | unresolved] = reviewed[withdraw | unresolved]
    activity[add] = 1
    return activity, withdraw | add | unresolved
