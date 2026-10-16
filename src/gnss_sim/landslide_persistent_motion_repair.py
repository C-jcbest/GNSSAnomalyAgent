"""Require the existing numerical baseline duration for novel visual activity."""

import numpy as np

from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_missed_motion_repair import repair_missed_motion


def repair_persistent_motion(initial, reviewed, permission, observed):
    """Keep initial activity, but reject isolated patches of novel agreement.

    The existing baseline requires 14 consecutive observed supporting days.
    Apply exactly that convention to new raw/visual agreement, without filling
    negative dates, bridging missing observations or moving proposal boundaries.
    Rejected numerical/visual conflicts remain unknown.
    """
    activity = repair_missed_motion(initial, reviewed, permission, observed)
    additions = (np.asarray(initial) != 1) & (activity == 1)
    sustained_additions = activity_from_mask(additions, observed) == 1
    activity[additions & ~sustained_additions] = -1
    return activity
