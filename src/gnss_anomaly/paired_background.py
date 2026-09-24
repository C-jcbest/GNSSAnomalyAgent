"""Development-only matched background controls and response diagnostics.

These controls contain no added injection; their native state is unknown.
Pair metadata and injection labels are evaluation-only, never Window fields.
"""

from pathlib import Path

import numpy as np

from .contracts import Prediction, Window
from .daily import verify_daily
from .datasets import load_dataset
from .storage import child, digest, file_hash, read_json, write_json

PROTOCOL = "paired-background-response-v1"


def matched_control(source: Window, injected: Window) -> Window:
    if (
        source.sampling_hours != 24
        or injected.sampling_hours != 24
        or len(source.values) != 180
        or source.timestamps != injected.timestamps
    ):
        raise ValueError("matched controls require the same 180-day grid")
    observed = np.isfinite(injected.array())
    original = source.array()
    if (observed & ~np.isfinite(original)).any():
        raise ValueError("injected input has observations absent from its source")
    values = np.where(observed, original, np.nan)
    rows = [[float(v) if np.isfinite(v) else None for v in row] for row in values]
    identity = {"protocol": PROTOCOL, "timestamps": source.model_dump(mode="json")["timestamps"]}
    return Window(
        case_id=digest({**identity, "values": rows})[:24],
        sampling_hours=24,
        timestamps=source.timestamps,
        values=rows,
    )


def build_controls(source: Path, injected: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    verify_daily(source)
    manifest = load_dataset(injected)
    if manifest.get("protocol") != "daily-injection-pilot-v1":
        raise ValueError("only the frozen daily development pilot is supported")
    sources = {
        r["id"]: r
        for r in read_json(source / "long/windows.json")["items"]
        if r["split"] == "development"
    }
    records, labels, files, pairs, seen = [], {}, {}, [], {}
    for item in manifest["records"]:
        if item["split"] != "development":
            raise ValueError("paired diagnosis may not access held-out inputs")
        if item["kind"] == "normal":
            continue
        origin = sources[item["source_group_id"]]
        path = child(source / "long", origin["file"])
        if file_hash(path) != origin["sha256"] or origin["group"] != item["group"]:
            raise ValueError("source identity mismatch")
        raw = Window.model_validate(read_json(path))
        w = Window.model_validate(read_json(child(injected, item["file"])))
        control = matched_control(raw, w)
        # Never merge identical values from different sources into one independent origin.
        control.case_id = digest({"source": origin["sha256"], "content": control.case_id})[:24]
        key = control.case_id
        pairs.append(
            {
                "injected_case_id": item["case_id"],
                "control_case_id": key,
                "kind": item["kind"],
                "state": item["state"],
                "source_id": origin["id"],
                "source_sha256": origin["sha256"],
            }
        )
        if key in seen:
            if seen[key] != item["state"]:
                raise ValueError("identical control unexpectedly has multiple states")
            continue
        seen[key] = item["state"]
        relative = f"inputs/{key}.json"
        write_json(out / relative, control.model_dump(mode="json"))
        files[relative] = file_hash(out / relative)
        records.append(
            {
                "case_id": key,
                "file": relative,
                "split": "development",
                "group": origin["group"],
                "source_group_id": origin["id"],
                "base_id": digest({"protocol": PROTOCOL, "source": origin["sha256"]})[:24],
                "kind": "normal",  # Legacy strata name means zero injection, not field normal.
                "state": item["state"],
                "semantic": "undetermined",
            }
        )
        labels[key] = {"events": [], "label_scope": "no_added_injection_not_field_truth"}
    if not pairs:
        raise ValueError("no injected development cases")
    write_json(out / "labels.json", labels)
    files["labels.json"] = file_hash(out / "labels.json")
    write_json(out / "pairs.json", {"protocol": PROTOCOL, "pairs": pairs})
    files["pairs.json"] = file_hash(out / "pairs.json")
    result = {
        "schema_version": 1,
        "kind": "platform_pilot_provisional",
        "sampling_hours": 24,
        "protocol": PROTOCOL,
        "injected_dataset_sha256": file_hash(injected / "manifest.json"),
        "source_bundle_sha256": file_hash(source / "bundle.json"),
        "records": records,
        "files": files,
        "limitations": "Matched no-injection controls; native state unknown. No field truth or capability card. The legacy normal stratum denotes no added injection only.",
    }
    write_json(out / "manifest.json", result)
    return {"pairs": len(pairs), "unique_controls": len(records)}


def response_counts(
    injected: Window,
    control: Window,
    truth: Prediction,
    injected_prediction: Prediction,
    control_prediction: Prediction,
) -> dict:
    observed = np.isfinite(injected.array())
    if injected.timestamps != control.timestamps or not np.array_equal(
        observed, np.isfinite(control.array())
    ):
        raise ValueError("paired inputs must share time grid and observed mask")
    target = truth.mask(len(observed)) & observed
    complete = injected_prediction.status == control_prediction.status == "ok"
    result = {"completed": complete, "expected_target_cells": int(target.sum())}
    if not complete:
        # An unavailable background prediction is not an empty successful control.
        return result
    pi = injected_prediction.mask(len(observed)) & observed
    p0 = control_prediction.mask(len(observed)) & observed
    result.update(
        target_cells=int(target.sum()),
        outside_cells=int((observed & ~target).sum()),
        observed_cells=int(observed.sum()),
        injected_target=int((pi & target).sum()),
        baseline_target=int((p0 & target).sum()),
        new_target=int((pi & ~p0 & target).sum()),
        inherited_target=int((pi & p0 & target).sum()),
        lost_target=int((~pi & p0 & target).sum()),
        injected_outside=int((pi & ~target).sum()),
        baseline_outside=int((p0 & ~target).sum()),
        new_outside=int((pi & ~p0 & ~target).sum()),
        inherited_outside=int((pi & p0 & ~target).sum()),
        baseline_positive=int(p0.sum()),
        changed_cells=int((pi ^ p0).sum()),
    )
    return result
