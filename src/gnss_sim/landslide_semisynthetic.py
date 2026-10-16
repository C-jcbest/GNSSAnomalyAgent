"""Inject known motion into observed GNSS residual backgrounds, preserving gaps."""
from __future__ import annotations

import numpy as np


def residual_background(coordinates_m):
    """Remove only an observed-date linear trend, before adding any injection.

    The residual is an experimental nuisance background, not a zero-motion field
    label. Seasonal motion and instrument changes can remain indistinguishable
    from weak injected activity. All components and original dates are retained.
    """
    coordinates = np.asarray(coordinates_m, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("A background needs calendar-aligned N/E/U coordinates")
    observed = np.isfinite(coordinates).all(axis=1)
    if observed.sum() < 60:
        raise ValueError("At least 60 complete observations are required")
    relative = (coordinates - coordinates[observed][0]) * 1000
    days = np.arange(len(coordinates), dtype=float)
    center = float(days[observed].mean())
    design = np.column_stack((np.ones(len(days)), days - center))
    coefficients = np.linalg.lstsq(design[observed], relative[observed], rcond=None)[0]
    residual = relative - design @ coefficients
    residual[~observed] = np.nan
    return residual, {"offset_mm": coefficients[0].tolist(),
                      "rate_NEU_mm_day": coefficients[1].tolist(),
                      "time_center_day": center, "observed_days": int(observed.sum())}


def add_injection(background_mm, displacement_mm, artifact_mm):
    """Add finite, known components without filling missing background dates."""
    background = np.asarray(background_mm, dtype=float)
    displacement = np.asarray(displacement_mm, dtype=float)
    artifact = np.asarray(artifact_mm, dtype=float)
    if background.shape != displacement.shape or artifact.shape != background.shape:
        raise ValueError("Background and injection calendars must agree")
    if not np.isfinite(displacement).all() or not np.isfinite(artifact).all():
        raise ValueError("Known injection components must be finite")
    injected = background + displacement + artifact
    observed = np.isfinite(background).all(axis=1)
    return [row.tolist() if available else None for row, available in zip(injected, observed)]
