"""Observation and runtime diagnostics; never alter the evaluation mask."""

import numpy as np


def runtime_diagnostics(prediction, tools, calls):
    numerical = [t["diagnostics"] for t in tools if "scorable_mask" in t.get("diagnostics", {})]
    scorable = (
        np.logical_or.reduce([np.array(d["scorable_mask"], dtype=bool) for d in numerical])
        if numerical
        else None
    )
    first = [c for c in calls if c.get("attempt", 1) == 1]
    return {
        "version": "runtime-diagnostics-v1",
        "numerical_scorable_cells": int(scorable.sum()) if numerical else None,
        "numerical_observed_cells": numerical[0]["observed_cells"] if numerical else None,
        "first_attempt_completed": all(
            c["status"] == "ok" and c.get("prediction_status", prediction.status) == "ok"
            for c in first
        )
        if first
        else None,
        "logical_model_calls": len(first),
        "model_attempts": len(calls),
    }
