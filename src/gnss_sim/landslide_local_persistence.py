"""Training-only motion/offset pairs that retain surrounding deformation."""

from __future__ import annotations

import numpy as np


def local_offset_counterpart(observations, latent, first, last, step_day):
    """Replace only the specified latent interval by an equal-displacement step.

    Motion before and after this interval remains in both pair members. This is
    a controlled training intervention, requiring known simulated components;
    it must never preprocess an observation supplied to the detector.
    """
    observations = np.asarray(observations, dtype=float)
    latent = np.asarray(latent, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 3:
        raise ValueError("Training intervention requires an original NEU calendar")
    if latent.shape != observations.shape or not np.isfinite(latent).all():
        raise ValueError("Training latent displacement must be finite on the same calendar")
    if not 0 <= first < step_day <= last < len(latent):
        raise ValueError("The offset must lie inside the selected training interval")
    if not np.isfinite(observations[[first, last]]).all():
        raise ValueError("Equal observed endpoints require original endpoint observations")
    if np.linalg.norm(latent[last] - latent[first]) <= 1e-10:
        raise ValueError("Local persistence needs a nonzero interval displacement")

    replacement = latent.copy()
    replacement[first:step_day] = latent[first]
    replacement[step_day : last + 1] = latent[last]
    negative = observations.copy()
    negative[first : last + 1] = (
        observations[first : last + 1] - latent[first : last + 1] + replacement[first : last + 1]
    )
    # Exact endpoint and outer observation bytes avoid unnecessary roundoff.
    negative[[first, last]] = observations[[first, last]]
    return negative, replacement
