"""Use a window-censored visual endpoint to choose one bounded continuation view."""

import numpy as np


def centered_target(center, days, width=98):
    if type(width) is not int or width < 2 or width > days:
        raise ValueError("Follow-up width must fit the original observation calendar")
    start = max(0, min(days - width, center - width // 2))
    return {"start": int(start), "stop": int(start + width)}


def continuation_choice(first, first_activity, conflicts, observed, minimum_days=14):
    """A right-edge activity answer is censored, not a physical cessation claim.

    Look beyond that edge only when the same original raw-conflict mechanism
    supplies enough observed continuation evidence. Negative raw conflicts are
    requests for inspection, never plateau labels. Left-boundary errors are
    outside this forward-continuation mechanism's claimed scope.
    """
    if first_activity.shape != observed.shape or conflicts.shape != observed.shape:
        raise ValueError("Continuation evidence must use the original calendar")
    if not np.isin(first_activity, [-1, 0, 1]).all() or np.any(first_activity[~observed] != -1):
        raise ValueError("Invalid activity state or missing observation decision")
    if observed.dtype != bool or conflicts.dtype != bool or np.any(conflicts & ~observed):
        raise ValueError("Only observed days may support continuation")
    start, stop = first["start"], first["stop"]
    if not 0 <= start < stop <= len(observed) or minimum_days < 1:
        raise ValueError("Invalid first target or continuation admission")
    censored = stop < len(observed) and first_activity[stop - 1] == 1
    outside_stop = min(len(observed), stop + 49)
    support = int(conflicts[stop:outside_stop].sum())
    return {"right_censored": bool(censored), "continuation_conflict_days": support,
            "numeric_admitted": bool(censored and support >= minimum_days),
            "edge": stop, "support_start": stop, "support_stop": outside_stop}


def followup_targets(first, remaining, choice, days):
    """Keep the first view and original count; isolate location, width and gate.

    Fallback here is the predeclared experimental no-trigger branch, not an old
    API compatibility path. The matched fixed short control shares the trigger
    and width, but centers on the other originally selected window.
    """
    if remaining is None:
        return {name: None for name in ("censored", "fixed_short", "wide", "no_numeric")}
    result = {name: dict(remaining) for name in ("censored", "fixed_short", "wide", "no_numeric")}
    if choice["numeric_admitted"]:
        result["censored"] = centered_target(first["stop"], days)
        result["wide"] = centered_target(first["stop"], days, 180)
        result["fixed_short"] = centered_target((remaining["start"] + remaining["stop"]) // 2, days)
    if choice["right_censored"]:
        result["no_numeric"] = centered_target(first["stop"], days)
    return result
