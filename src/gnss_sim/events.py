"""Pure, single-axis P2 event profiles. Indices are zero-based and inclusive."""

from __future__ import annotations

from datetime import date

import numpy as np

from gnss_sim.schemas import (
    AccelerationEvent,
    AccelerationParameters,
    Axis,
    SlowTrendEvent,
    SpikeEvent,
    SpikeParameters,
    StepEvent,
    StepParameters,
    TransientParameters,
    TransientShiftEvent,
    TrendParameters,
)

AXIS_INDEX = {"N": 0, "E": 1, "U": 2}
SAFE_START = 60
SAFE_END = 304
TREND_DURATION = 90
TRANSIENT_DURATION = 14


def _base(dates: list[date], axis: Axis, start: int, end: int) -> tuple[np.ndarray, dict]:
    if len(dates) != 365 or not (SAFE_START <= start <= end <= SAFE_END):
        raise ValueError("P2 event must fit the 365-day interior [60, 304]")
    contribution = np.zeros((len(dates), 3), dtype=float)
    fields = dict(
        event_id="event_001",
        axis=axis,
        start_index=start,
        end_index=end,
        start_date=dates[start],
        end_date=dates[end],
    )
    return contribution, fields


def make_spike(dates: list[date], axis: Axis, start: int, amplitude_mm: float):
    contribution, fields = _base(dates, axis, start, start)
    contribution[start, AXIS_INDEX[axis]] = amplitude_mm
    return contribution, SpikeEvent(
        **fields, type="spike", source="observation_artifact", persistent=False,
        parameters=SpikeParameters(duration_days=1, amplitude_mm=amplitude_mm, sigma_multiplier=6),
    )


def make_step(dates: list[date], axis: Axis, start: int, amplitude_mm: float):
    contribution, fields = _base(dates, axis, start, start)
    contribution[start:, AXIS_INDEX[axis]] = amplitude_mm
    return contribution, StepEvent(
        **fields, type="step", source="injected_deformation", persistent=True,
        parameters=StepParameters(duration_days=1, amplitude_mm=amplitude_mm),
    )


def make_slow_trend(dates: list[date], axis: Axis, start: int, final_offset_mm: float):
    end = start + TREND_DURATION - 1
    contribution, fields = _base(dates, axis, start, end)
    slope = final_offset_mm / (TREND_DURATION - 1)
    elapsed = np.minimum(np.arange(len(dates) - start), TREND_DURATION - 1)
    contribution[start:, AXIS_INDEX[axis]] = elapsed * slope
    return contribution, SlowTrendEvent(
        **fields,
        type="slow_trend", source="injected_deformation", persistent=True,
        parameters=TrendParameters(
            duration_days=TREND_DURATION,
            final_offset_mm=final_offset_mm,
            slope_mm_per_day=slope,
        ),
    )


def make_acceleration(dates: list[date], axis: Axis, start: int, final_offset_mm: float):
    end = start + TREND_DURATION - 1
    contribution, fields = _base(dates, axis, start, end)
    u = np.minimum(np.arange(len(dates) - start) / (TREND_DURATION - 1), 1.0)
    contribution[start:, AXIS_INDEX[axis]] = final_offset_mm * u**2
    return contribution, AccelerationEvent(
        **fields, type="acceleration", source="injected_deformation", persistent=True,
        parameters=AccelerationParameters(
            duration_days=TREND_DURATION, final_offset_mm=final_offset_mm
        ),
    )


def make_transient_shift(dates: list[date], axis: Axis, start: int, amplitude_mm: float):
    end = start + TRANSIENT_DURATION - 1
    contribution, fields = _base(dates, axis, start, end)
    contribution[start : end + 1, AXIS_INDEX[axis]] = amplitude_mm
    return contribution, TransientShiftEvent(
        **fields, type="transient_shift", source="observation_artifact", persistent=False,
        parameters=TransientParameters(
            duration_days=TRANSIENT_DURATION, amplitude_mm=amplitude_mm
        ),
    )
