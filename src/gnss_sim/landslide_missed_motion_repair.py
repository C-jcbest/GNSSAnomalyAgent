"""Allow bounded visual additions only where raw sustained-motion evidence agrees."""

import numpy as np


def sustained_motion_support(features, long_evidence, policy):
    """Reuse the frozen21/42 raw support and thresholds, without new model scores."""
    short_available = features.observed & np.isfinite(features.progressive_rate)
    long_available = features.observed & np.isfinite(long_evidence["rate"])
    short_motion = (short_available & (features.progressive_rate >= policy.minimum_rate)
                    & (features.signal_to_noise >= policy.minimum_signal_to_noise))
    long_motion = (long_available & (long_evidence["rate"] >= policy.minimum_rate)
                   & (long_evidence["signal_to_noise"] >= policy.minimum_signal_to_noise))
    return short_motion | long_motion


def repair_missed_motion(initial, reviewed, permission, observed):
    """Preserve initial activity and require a local visual confirmation to add.

    Permission is computed before local responses. Numerical support alone never
    labels activity. Within permitted non-active dates, a negative or unknown
    visual reply leaves an unresolved conflict; it is not a confident negative.
    Failed replies are already represented as unknown by the strict compiler.
    """
    initial, reviewed = np.asarray(initial), np.asarray(reviewed)
    observed, permission = np.asarray(observed), np.asarray(permission)
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Motion repair must retain the original calendar and states")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing observations must remain unknown")
    if permission.shape != observed.shape or permission.dtype != bool:
        raise ValueError("Motion repair requires an aligned boolean permission mask")
    if np.any(permission & ~observed):
        raise ValueError("Missing dates cannot be permitted for activity repair")
    allowed = permission & (initial != 1)
    activity = initial.copy()
    activity[allowed] = -1
    activity[allowed & (reviewed == 1)] = 1
    return activity
