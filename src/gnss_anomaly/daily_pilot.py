"""Provisional daily injection study; native changes remain unlabelled."""

from pathlib import Path

import numpy as np

from .contracts import Window, mask_events
from .daily import verify_daily
from .datasets import KINDS, missing_variant
from .storage import child, digest, file_hash, read_json, write_json

PROTOCOL = "daily-injection-pilot-v1"


def inject_daily(window, kind, rng):
    if window.sampling_hours != 24 or len(window.values) != 180 or kind not in KINDS:
        raise ValueError("daily pilot requires 180 daily samples and a supported kind")
    x = window.array().copy()
    mask = np.zeros_like(x, dtype=bool)
    params = {"protocol": PROTOCOL, "label_rule": "affected displacement including retained offset"}
    if kind == "normal":
        return x, mask, params
    axis = int(rng.integers(3))
    delta = np.diff(x[:, axis])
    delta = delta[np.isfinite(delta)]
    if len(delta) < 30:
        raise ValueError("insufficient adjacent-day observations for noise scale")
    sigma = max(float(1.4826 * np.median(np.abs(delta - np.median(delta))) / np.sqrt(2)), 0.1)
    start = int(rng.integers(45, 76))
    sign = int(rng.choice([-1, 1]))
    duration = {"spike": 1, "step": 180 - start, "drift": 60, "acceleration": 60, "variance": 20}[
        kind
    ]
    amp = max(
        {"spike": 8, "step": 10, "drift": 12, "acceleration": 20, "variance": 2}[kind],
        sigma * (12 if kind == "acceleration" else 8),
    )
    if kind == "spike":
        x[start, axis] += sign * amp
        mask[start, axis] = True
    elif kind == "step":
        x[start:, axis] += sign * amp
        mask[start:, axis] = True
    elif kind in ("drift", "acceleration"):
        growth = np.clip((np.arange(180) - start + 1) / duration, 0, 1)
        if kind == "acceleration":
            growth = growth**2
        x[:, axis] += sign * amp * growth
        mask[start:, axis] = True
    else:
        x[start : start + duration, axis] += rng.normal(0, amp, duration)
        mask[start : start + duration, axis] = True
    params.update(
        axis=axis,
        start=start,
        duration_days=duration,
        amplitude_mm=amp,
        sign=sign,
        daily_difference_sigma_mm=sigma,
    )
    return x, mask, params


def build_daily_pilot(source: Path, review: Path, out: Path, seed=20260922):
    if out.exists():
        raise FileExistsError(out)
    verify_daily(source)
    decisions = read_json(review)
    if decisions.get("scope") != "development_daily_pilot_only":
        raise ValueError("daily pilot requires a development-only review")
    selected = [
        r for r in read_json(source / "long/windows.json")["items"] if r["split"] == "development"
    ]
    approved = {r["id"]: r for r in decisions["items"]}
    if not selected or set(approved) != {r["id"] for r in selected}:
        raise ValueError("review must cover all selected development backgrounds")
    rng = np.random.default_rng(seed)
    records, labels, files = [], {}, {}
    for item in selected:
        path = child(source / "long", item["file"])
        decision = approved[item["id"]]
        if decision.get("sha256") != file_hash(path) or decision.get("decision") != "pilot_only":
            raise ValueError("review/hash mismatch")
        w = Window.model_validate(read_json(path))
        for kind in KINDS:
            values, truth, params = inject_daily(w, kind, rng)
            base = digest(
                {"protocol": PROTOCOL, "source": item["sha256"], "kind": kind, "seed": seed}
            )[:24]
            for state in ("native", "random10", "random25", "block25"):
                observed = (
                    values.copy() if state == "native" else missing_variant(values, state, rng)
                )
                key = digest({"base": base, "state": state})[:24]
                payload = Window(
                    case_id=key,
                    sampling_hours=24,
                    timestamps=w.timestamps,
                    values=[
                        [float(v) if np.isfinite(v) else None for v in row] for row in observed
                    ],
                )
                relative = f"inputs/{key}.json"
                write_json(out / relative, payload.model_dump(mode="json"))
                files[relative] = file_hash(out / relative)
                records.append(
                    {
                        "case_id": key,
                        "file": relative,
                        "split": "development",
                        "group": item["group"],
                        "source_group_id": item["id"],
                        "base_id": base,
                        "kind": kind,
                        "state": state,
                        "semantic": {
                            "spike": "quality",
                            "variance": "quality",
                            "drift": "deformation",
                            "acceleration": "deformation",
                            "normal": "normal",
                            "step": "undetermined",
                        }[kind],
                    }
                )
                labels[key] = {
                    "events": [e.model_dump() for e in mask_events(truth, kind)],
                    "injection": params,
                    "native_missing_cells": int(np.isnan(w.array()).sum()),
                }
    write_json(out / "labels.json", labels)
    files["labels.json"] = file_hash(out / "labels.json")
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "kind": "platform_pilot_provisional",
            "sampling_hours": 24,
            "protocol": PROTOCOL,
            "seed": seed,
            "review_sha256": file_hash(review),
            "records": records,
            "files": files,
            "limitations": "Two same-group development backgrounds; native changes unlabelled. Scores describe injection-mask agreement, not field accuracy.",
        },
    )
    return len(records)
