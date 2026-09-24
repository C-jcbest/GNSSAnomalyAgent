"""Split-preserving, injection-only benchmark from the frozen Beijing-15 daily snapshot."""

from pathlib import Path

import numpy as np

from .contracts import Window, mask_events
from .daily import verify_daily
from .daily_pilot import inject_daily
from .datasets import KINDS, missing_variant
from .storage import child, digest, file_hash, read_json, write_json

PROTOCOL = "daily15-injection-v1"
STATES = ("native", "random10", "random25", "block25")


def build_daily_dataset(
    source: Path, out: Path, seed: int = 20260923, native_only: bool = False
) -> int:
    if out.exists():
        raise FileExistsError(out)
    report = verify_daily(source)
    if report["window_days"] != 180:
        raise ValueError("requires the fixed 180-day snapshot")
    catalog = read_json(source / "long/windows.json")
    selected = catalog["items"]
    if not selected or not {"development", "validation", "test"} <= {
        item["split"] for item in selected
    }:
        raise ValueError("all three original source splits are required")
    records, labels, files = [], {}, {}
    for item in selected:
        if item["split"] != report["group_splits"][item["group"]]:
            raise ValueError("source split differs from frozen group assignment")
        path = child(source / "long", item["file"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("source checksum mismatch")
        window = Window.model_validate(read_json(path))
        if len(window.values) != 180 or window.sampling_hours != 24:
            raise ValueError("only 180-day windows are supported")
        if np.isfinite(window.array()).all(axis=1).sum() < 144:
            raise ValueError("source does not meet 80% three-axis completeness")
        for kind in KINDS:
            base = digest([PROTOCOL, seed, item["sha256"], kind])[:24]
            injection_rng = np.random.default_rng(int(digest([base, "injection"])[:16], 16))
            values, truth, parameters = inject_daily(window, kind, injection_rng)
            for state in ("native",) if native_only else STATES:
                variant_rng = np.random.default_rng(int(digest([base, state])[:16], 16))
                observed = (
                    values.copy()
                    if state == "native"
                    else missing_variant(values, state, variant_rng)
                )
                case_id = digest([base, state])[:24]
                relative = f"inputs/{case_id}.json"
                payload = Window(
                    case_id=case_id,
                    sampling_hours=24,
                    timestamps=window.timestamps,
                    values=[
                        [float(value) if np.isfinite(value) else None for value in row]
                        for row in observed
                    ],
                )
                write_json(out / relative, payload.model_dump(mode="json"))
                files[relative] = file_hash(out / relative)
                records.append(
                    {
                        "case_id": case_id,
                        "file": relative,
                        "split": item["split"],
                        "group": item["group"],
                        "source_group_id": item["id"],
                        "base_id": base,
                        "kind": kind,
                        "state": state,
                        "semantic": "quality" if kind in ("spike", "variance") else "undetermined",
                    }
                )
                labels[case_id] = {
                    "events": [event.model_dump() for event in mask_events(truth, kind)],
                    "injection": parameters,
                }
    write_json(out / "labels.json", labels)
    files["labels.json"] = file_hash(out / "labels.json")
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "kind": "daily_injection_unverified_v1",
            "protocol": "daily15-injection-native-v1" if native_only else PROTOCOL,
            "sampling_hours": 24,
            "seed": seed,
            "source_catalog_sha256": file_hash(source / "long/windows.json"),
            "source_bundle_sha256": file_hash(source / "bundle.json"),
            "records": records,
            "files": files,
            "limitations": "Labels mark only controlled injections; native changes are unknown, including no-injection cases. Source windows are not certified normal."
            + (
                " Only source missingness is retained; no missing values are injected."
                if native_only
                else ""
            ),
        },
    )
    return len(records)
