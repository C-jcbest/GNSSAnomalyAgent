import numpy as np

from .contracts import Prediction, Window

METRIC_VERSION = "observed-cells-v1.1"


def counts(window: Window, truth: Prediction, prediction: Prediction) -> dict:
    observed = np.isfinite(window.array())
    target = truth.mask(len(observed))
    guess = prediction.mask(len(observed))
    tp = int((guess & target & observed).sum())
    fp = int((guess & ~target & observed).sum())
    fn = int((~guess & target & observed).sum())
    tn = int((~guess & ~target & observed).sum()) if prediction.status == "ok" else 0
    hidden = sum(
        not (observed & e_mask).any()
        for e_mask in (Prediction(events=[e]).mask(len(observed)) for e in truth.events)
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "observed_cells": int(observed.sum()),
        "hidden_events": int(hidden),
        "unobserved_positive_cells": int((guess & ~observed).sum()),
        "has_truth": bool(target.any()),
        "has_observed_truth": bool((target & observed).any()),
        "false_alarm_window": bool((guess & observed).any()) if not target.any() else None,
        "completed": prediction.status == "ok",
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        # Project convention: groups with no observed positive truth use normal-window FAR.
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
    }


def aggregate(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r["metrics"][k] for r in rows) for k in ("tp", "fp", "fn"))
    normals = [r for r in rows if not r["metrics"]["has_truth"]]
    completed_normals = [r for r in normals if r["metrics"]["completed"]]
    diagnostics = [r.get("diagnostics", {}) for r in rows]
    first = [
        d["first_attempt_completed"]
        for d in diagnostics
        if d.get("first_attempt_completed") is not None
    ]
    covered = [d for d in diagnostics if d.get("numerical_observed_cells") is not None]
    denominator = sum(d["numerical_observed_cells"] for d in covered)
    logical = [d["logical_model_calls"] for d in diagnostics if "logical_model_calls" in d]
    return {
        "cases": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **prf(tp, fp, fn),
        "completion_rate": np.mean([r["metrics"]["completed"] for r in rows]).item(),
        "first_attempt_completion_rate": float(np.mean(first)) if first else None,
        "numerical_coverage": sum(d["numerical_scorable_cells"] for d in covered) / denominator
        if denominator
        else None,
        "mean_logical_model_calls": float(np.mean(logical)) if logical else None,
        "normal_windows": len(normals),
        "normal_failures": len(normals) - len(completed_normals),
        "normal_false_alarm_rate_completed": (
            sum(r["metrics"]["false_alarm_window"] for r in completed_normals)
            / len(completed_normals)
            if completed_normals
            else None
        ),
        "hidden_events": sum(r["metrics"]["hidden_events"] for r in rows),
        "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
        "mean_model_requests": float(np.mean([r["model_requests"] for r in rows])),
        "mean_tool_calls": float(np.mean([len(r["tools"]) for r in rows])),
        "cost": (
            sum(r["cost"] for r in rows) if all(r["cost"] is not None for r in rows) else None
        ),
    }
