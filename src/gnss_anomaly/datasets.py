"""Snapshot backgrounds, reviewed injection sets, and a clearly marked demo fixture."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Window, mask_events
from .storage import child, digest, file_hash, read_json, verify_files, write_json

KINDS = ("spike", "step", "drift", "acceleration", "variance", "normal")
STATES = ("complete", "random10", "random25", "block25")


def missing_variant(values, state, rng):
    if state not in STATES:
        raise ValueError("unknown missingness state")
    observed = values.copy()
    count = int(len(values) * (0.1 if state == "random10" else 0.25))
    if state.startswith("random"):
        observed[rng.choice(len(values), count, replace=False)] = np.nan
    elif state == "block25":
        gap = int(rng.integers(0, len(values) - count + 1))
        observed[gap : gap + count] = np.nan
    return observed


def build_pilot(background: Path, review: Path, out: Path, seed=20260922):
    """One development background, 24 provisional inputs; never a formal evaluation set."""
    if out.exists():
        raise FileExistsError(out)
    decision = read_json(review)
    if decision.get("split") != "development" or decision.get("scope") != "pilot_only":
        raise ValueError("pilot requires an explicit development-only review")
    if decision.get("decision") != "usable_for_pilot" or not decision.get("review_note"):
        raise ValueError("pilot background requires a documented visual review")
    if decision.get("window_sha256") != file_hash(background):
        raise ValueError("pilot background checksum mismatch")
    source = Window.model_validate(read_json(background))
    rng = np.random.default_rng(seed)
    records, labels, files = [], {}, {}
    for kind in KINDS:
        values, truth, params = inject(source, kind, rng)
        base_id = digest({"source": file_hash(background), "kind": kind, "seed": seed})[:24]
        for state in STATES:
            observed = missing_variant(values, state, rng)
            key = digest({"base": base_id, "state": state})[:24]
            window = Window(
                case_id=key,
                timestamps=source.timestamps,
                values=[[float(v) if np.isfinite(v) else None for v in row] for row in observed],
            )
            relative = f"inputs/{key}.json"
            write_json(out / relative, window.model_dump(mode="json"))
            files[relative] = file_hash(out / relative)
            records.append(
                {
                    "case_id": key,
                    "file": relative,
                    "split": "development",
                    "group": decision["group"],
                    "source_group_id": source.case_id,
                    "base_id": base_id,
                    "kind": kind,
                    "state": state,
                    "semantic": {
                        "spike": "quality",
                        "variance": "quality",
                        "acceleration": "deformation",
                        "normal": "normal",
                    }.get(kind, "undetermined"),
                }
            )
            labels[key] = {
                "events": [e.model_dump() for e in mask_events(truth, kind)],
                "injection": params,
            }
    write_json(out / "labels.json", labels)
    files["labels.json"] = file_hash(out / "labels.json")
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "kind": "platform_pilot_provisional",
            "seed": seed,
            "review_sha256": file_hash(review),
            "background_sha256": file_hash(background),
            "review": decision,
            "records": records,
            "files": files,
            "limitations": "Single visually screened development background; labels cover injection only. Not field-normal truth, not generalization or capability evidence.",
        },
    )
    return len(records)


def extract_window(snapshot: Path, out: Path, start, end):
    """Local unlabelled case extraction; does not approve backgrounds or create dense truth."""
    if out.exists():
        raise FileExistsError(out)
    manifest = read_json(snapshot / "manifest.json")
    verify_files(snapshot, manifest["files"])
    if start >= end or any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("extraction must use increasing whole-hour boundaries")
    if start < pd.Timestamp(manifest["start"]) or end > pd.Timestamp(manifest["end_exclusive"]):
        raise ValueError("requested range is outside snapshot")
    frame = pd.read_csv(snapshot / "observations.csv")
    times = pd.to_datetime(frame.timestamp)
    selected = (times >= start) & (times < end)
    values = frame.loc[selected, ["n_mm", "e_mm", "u_mm"]].to_numpy(float)
    key = digest(
        {
            "snapshot": file_hash(snapshot / "manifest.json"),
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    )[:24]
    window = Window(
        case_id=key,
        sampling_hours=manifest.get("sampling_hours", 1),
        timestamps=list(times[selected]),
        values=[[float(v) if np.isfinite(v) else None for v in row] for row in values],
    )
    write_json(out / "window.json", window.model_dump(mode="json"))
    write_json(
        out / "provenance.json",
        {
            "kind": "unlabelled_real_window",
            "station_alias": manifest["station_alias"],
            "snapshot_sha256": file_hash(snapshot / "manifest.json"),
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "sampling_hours": window.sampling_hours,
            "samples": len(values),
            "hours": len(values) * window.sampling_hours,
            "window_sha256": file_hash(out / "window.json"),
            "has_dense_ground_truth": False,
        },
    )
    return len(values)


def split_groups(groups: list[str], seed: int) -> dict[str, str]:
    groups = sorted(set(groups))
    if len(groups) < 5:
        raise ValueError("need at least 5 source groups for development/validation/test")
    np.random.default_rng(seed).shuffle(groups)
    ndev, nval = max(1, int(len(groups) * 0.6)), max(1, int(len(groups) * 0.2))
    return {
        g: ("development" if i < ndev else "validation" if i < ndev + nval else "test")
        for i, g in enumerate(groups)
    }


def prepare_backgrounds(snapshots: list[Path], out: Path, hours: int = 336):
    if hours < 168:
        raise ValueError("background window must have at least 168 hours")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    items = []
    for snapshot in snapshots:
        manifest = read_json(snapshot / "manifest.json")
        verify_files(snapshot, manifest["files"])
        frame = pd.read_csv(snapshot / "observations.csv")
        for start in range(0, len(frame) - hours + 1, hours):
            part = frame.iloc[start : start + hours]
            values = part[["n_mm", "e_mm", "u_mm"]].to_numpy(float)
            if not np.isfinite(values).all():
                continue
            key = digest({"snapshot": file_hash(snapshot / "manifest.json"), "start": start})[:24]
            window = Window(
                case_id=key, timestamps=list(pd.to_datetime(part.timestamp)), values=values.tolist()
            )
            path = out / "windows" / f"{key}.json"
            write_json(path, window.model_dump(mode="json"))
            items.append(
                {
                    "id": key,
                    "file": path.relative_to(out).as_posix(),
                    "sha256": file_hash(path),
                    "group": manifest.get("site_alias") or manifest["station_alias"],
                    "station_alias": manifest["station_alias"],
                    "approved": False,
                    "review_note": "",
                    "source_units_verified": manifest.get("units_verified", False),
                    "source_manifest_sha256": file_hash(snapshot / "manifest.json"),
                }
            )
    write_json(out / "backgrounds.json", {"kind": "platform_backgrounds", "items": items})
    return len(items)


def demo_backgrounds(out: Path, seed: int = 42):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rng, items = np.random.default_rng(seed), []
    for i in range(15):
        t = np.arange(336)
        noise = rng.normal(size=(336, 3)) * [1, 1, 2]
        values = noise + t[:, None] * np.array([0.025, 0.01, -0.006])
        key = digest({"demo": seed, "group": i})[:24]
        window = Window(
            case_id=key,
            timestamps=[
                datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(h)) for h in t
            ],
            values=values.tolist(),
        )
        path = out / "windows" / f"{key}.json"
        write_json(path, window.model_dump(mode="json"))
        items.append(
            {
                "id": key,
                "file": path.relative_to(out).as_posix(),
                "sha256": file_hash(path),
                "group": f"demo-{i:02d}",
                "approved": True,
            }
        )
    write_json(out / "backgrounds.json", {"kind": "synthetic_demo_only", "items": items})


def inject(window: Window, kind: str, rng: np.random.Generator):
    if window.sampling_hours != 1:
        raise ValueError("daily injection protocol is not frozen; do not reuse hourly parameters")
    values = window.array().copy()
    if not np.isfinite(values).all() or len(values) < 168:
        raise ValueError("injection requires a complete background of at least 168 hours")
    truth = np.zeros_like(values, dtype=bool)
    params = {}
    if kind != "normal":
        n = len(values)
        axis = int(rng.integers(3))
        scale = max(
            float(np.median(np.abs(np.diff(values[:, axis]) - np.median(np.diff(values[:, axis])))))
            * 1.4826
            / np.sqrt(2),
            0.05,
        )
        strength = float(rng.uniform(2, 6))
        start = int(rng.integers(24, n - 96))
        duration = int(rng.integers(24, 73))
        end = start + duration
        sign = int(rng.choice([-1, 1]))
        amplitude = strength * scale * sign
        if kind == "spike":
            end = start + int(rng.integers(1, 4))
            values[start:end, axis] += amplitude
        elif kind == "step":
            end = n
            values[start:, axis] += amplitude
        elif kind in ("drift", "acceleration"):
            # Hold the accumulated offset; resetting would inject an unintended reverse step.
            progression = np.minimum(np.arange(1, n - start + 1) / duration, 1)
            values[start:, axis] += amplitude * progression ** (2 if kind == "acceleration" else 1)
            end = n
        elif kind == "variance":
            values[start:end, axis] += rng.normal(0, scale * np.sqrt(strength**2 - 1), end - start)
        else:
            raise ValueError(f"unknown injection: {kind}")
        truth[start:end, axis] = True
        params = {
            "start": start,
            "end_exclusive": end,
            "axis": axis,
            "duration": duration,
            "strength": strength,
            "amplitude_mm": amplitude,
        }
    return values, truth, params


def build_dataset(backgrounds: Path, out: Path, per_kind: int, seed: int = 20260922):
    if out.exists():
        raise FileExistsError(out)
    if per_kind < 5:
        raise ValueError("per-kind must be >= 5")
    catalog = read_json(backgrounds / "backgrounds.json")
    items = [r for r in catalog["items"] if r["approved"] is True]
    if "group_splits" in catalog:
        groups = catalog["group_splits"]
        for item in items:
            if groups.get(item["group"]) != item.get("split"):
                raise ValueError("background split does not match frozen source groups")
        if {groups[r["group"]] for r in items} != {"development", "validation", "test"}:
            raise ValueError("reviewed backgrounds must cover all frozen splits")
        # Retain the original assignment even when review removes every window in one group.
        groups = {r["group"]: groups[r["group"]] for r in items}
    else:
        groups = split_groups([r["group"] for r in items], seed)
    if per_kind < len(groups):
        raise ValueError("per-kind must cover every available group")
    rng = np.random.default_rng(seed)
    items = sorted(items, key=lambda r: (r["group"], r["id"]))
    # Round-robin across groups avoids many windows from one station dominating selection.
    grouped = {g: [r for r in items if r["group"] == g] for g in sorted(groups)}
    group_order = list(grouped)
    rng.shuffle(group_order)
    records, labels, files = [], {}, {}
    out.mkdir(parents=True)
    for kind in KINDS:
        for i in range(per_kind):
            group = group_order[i % len(group_order)]
            candidates = grouped[group]
            item = candidates[(i // len(group_order)) % len(candidates)]
            path = child(backgrounds, item["file"])
            if file_hash(path) != item["sha256"]:
                raise ValueError("background checksum mismatch")
            source = Window.model_validate(read_json(path))
            values, truth, params = inject(source, kind, rng)
            base_id = digest({"source": item["sha256"], "kind": kind, "rep": i, "seed": seed})[:24]
            semantic = {
                "spike": "quality",
                "variance": "quality",
                "acceleration": "deformation",
                "step": "undetermined",
                "drift": "undetermined",
                "normal": "normal",
            }[kind]
            for state in STATES:
                observed = missing_variant(values, state, rng)
                case_id = digest({"base": base_id, "state": state})[:24]
                rows = [[float(v) if np.isfinite(v) else None for v in row] for row in observed]
                window = Window(case_id=case_id, timestamps=source.timestamps, values=rows)
                relative = f"inputs/{case_id}.json"
                write_json(out / relative, window.model_dump(mode="json"))
                files[relative] = file_hash(out / relative)
                records.append(
                    {
                        "case_id": case_id,
                        "file": relative,
                        "split": groups[group],
                        "group": group,
                        "source_group_id": item["id"],
                        "base_id": base_id,
                        "kind": kind,
                        "state": state,
                        "semantic": semantic,
                    }
                )
                labels[case_id] = {
                    "events": [e.model_dump() for e in mask_events(truth, kind)],
                    "injection": params,
                }
    write_json(out / "labels.json", labels)
    files["labels.json"] = file_hash(out / "labels.json")
    manifest = {
        "schema_version": 1,
        "kind": catalog["kind"],
        "seed": seed,
        "background_catalog_sha256": file_hash(backgrounds / "backgrounds.json"),
        "records": records,
        "files": files,
    }
    write_json(out / "manifest.json", manifest)
    return len(records)


def load_dataset(root: Path):
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    for key in ("group", "source_group_id"):
        seen = {}
        for record in manifest["records"]:
            if seen.setdefault(record[key], record["split"]) != record["split"]:
                raise ValueError("source leakage across splits")
    return manifest
