"""A training-only annual reference for already detrended GNSS backgrounds."""
from __future__ import annotations

from datetime import date

import numpy as np

from gnss_sim.landslide import LandslideInput

PERIOD_DAYS = 365.25
PHASE_ORIGIN = date(2000, 1, 1)


def observed_values(case):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    return values, np.isfinite(values).all(axis=1)


def projected_annual_basis(dates, observed):
    """Match the benchmark's linear detrending using dates and gaps only.

    Projection coefficients never use the current displacement or injection.
    This does not remove a fitted current trend, and is not a field-motion label.
    """
    days = np.array([(day - PHASE_ORIGIN).days for day in dates], dtype=float)
    if len(days) < 365 or np.count_nonzero(observed) < 60:
        raise ValueError("An annual reference needs a year and 60 complete observations")
    if observed.shape != days.shape or not np.all(np.diff(days) == 1):
        raise ValueError("Reference dates must retain a complete daily calendar")
    phase = 2 * np.pi * days / PERIOD_DAYS
    basis = np.column_stack((np.sin(phase), np.cos(phase)))
    centered_days = days - days[observed].mean()
    linear = np.column_stack((np.ones(len(days)), centered_days))
    coefficients = np.linalg.lstsq(linear[observed], basis[observed], rcond=None)[0]
    return basis - linear @ coefficients


def fit_annual_reference(training_cases):
    """Fit one annual sine/cosine per N/E/U from supplied training references.

    Callers own the train-only population. Reference records are assumed to
    contain no target injection; this is a benchmark normal-reference condition,
    not an independently established claim of stationary field observations.
    """
    if len(training_cases) < 2:
        raise ValueError("Use at least two separate training reference records")
    matrices, responses = [], []
    for case in training_cases:
        values, observed = observed_values(case)
        basis = projected_annual_basis(case.dates, observed)
        matrices.append(basis[observed])
        responses.append(values[observed])
    coefficients = np.linalg.lstsq(np.concatenate(matrices), np.concatenate(responses), rcond=None)[0]
    return {"schema": "gnss-annual-reference-v1", "period_days": PERIOD_DAYS,
            "phase_origin": PHASE_ORIGIN.isoformat(), "coefficients_sin_cos_NEU_mm": coefficients.tolist(),
            "training_case_ids": [case.case_id for case in training_cases],
            "scope": "Training-normal annual reference in already linearly detrended backgrounds; not field ground truth"}


def apply_annual_reference(case, model):
    """Subtract only the frozen historical reference; preserve every gap."""
    if model["schema"] != "gnss-annual-reference-v1" or model["period_days"] != PERIOD_DAYS:
        raise ValueError("Unexpected annual reference specification")
    if model["phase_origin"] != PHASE_ORIGIN.isoformat():
        raise ValueError("Annual phase origin differs")
    coefficients = np.asarray(model["coefficients_sin_cos_NEU_mm"], dtype=float)
    if coefficients.shape != (2, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Annual coefficients must be finite sine/cosine by N/E/U")
    values, observed = observed_values(case)
    prediction = projected_annual_basis(case.dates, observed) @ coefficients
    residual = values - prediction
    corrected = LandslideInput(case_id=case.case_id, dates=case.dates,
        displacement_mm=[row.tolist() if available else None for row, available in zip(residual, observed)])
    return corrected, prediction
