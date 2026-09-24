"""Offline, lossless edge trimming and source-grouped unlabelled window preparation."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Window
from .storage import child, digest, file_hash, read_json, verify_files, write_json

VERSION = "offline-preparation-v1"
COLUMNS = ["n_mm", "e_mm", "u_mm"]


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def quality(frame):
    values = frame[COLUMNS].to_numpy(float)
    valid = np.isfinite(values)
    return {
        "expected_hours": len(frame),
        "observed_timestamps": int(valid.any(axis=1).sum()),
        "complete_hours": int(valid.all(axis=1).sum()),
        "missing_fraction": float(1 - valid.mean()),
        "longest_empty_gap_hours": int(
            max((b - a for a, b in runs(~valid.any(axis=1))), default=0)
        ),
        "longest_complete_run_hours": int(
            max((b - a for a, b in runs(valid.all(axis=1))), default=0)
        ),
    }


def as_window(frame, key):
    values = frame[COLUMNS].to_numpy(float)
    return Window(
        case_id=key,
        timestamps=list(pd.to_datetime(frame.timestamp, utc=True)),
        values=[[float(v) if np.isfinite(v) else None for v in row] for row in values],
    )


def local_iso(value):
    return pd.Timestamp(value).tz_convert("Asia/Shanghai").isoformat()


def trim_snapshot(source: Path, out: Path, group: str):
    """A partially observed boundary is retained; only all-missing edge hours are removed."""
    if out.exists():
        raise FileExistsError(out)
    meta = read_json(source / "manifest.json")
    verify_files(source, meta["files"])
    frame = pd.read_csv(source / "observations.csv", float_precision="round_trip")
    as_window(frame, meta["station_alias"])  # Validate the original hourly grid before slicing.
    valid = np.isfinite(frame[COLUMNS].to_numpy(float)).any(axis=1)
    found = np.flatnonzero(valid)
    row = {
        "station_alias": meta["station_alias"],
        "group": group,
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "original_start": meta["start"],
        "original_end_exclusive": meta["end_exclusive"],
        "original_hours": len(frame),
        "status": "trimmed" if len(found) else "excluded_empty",
    }
    if not len(found):
        return {**row, "removed_hours": len(frame), "audit": None}, None
    first, stop = int(found[0]), int(found[-1]) + 1
    part = frame.iloc[first:stop].copy()
    start = local_iso(part.timestamp.iloc[0])
    end = local_iso(pd.Timestamp(part.timestamp.iloc[-1]).to_pydatetime() + timedelta(hours=1))
    row.update(
        start=start,
        end_exclusive=end,
        leading_removed_hours=first,
        trailing_removed_hours=len(frame) - stop,
        removed_hours=len(frame) - len(part),
        audit=quality(part),
    )
    out.mkdir(parents=True)
    # Keep original CSV text, including finite values and blank cells, exactly unchanged.
    lines = (source / "observations.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != len(frame) + 1:
        raise ValueError("unexpected multiline observation CSV")
    (out / "observations.csv").write_text(
        "".join([lines[0], *lines[first + 1 : stop + 1]]), encoding="utf-8", newline=""
    )
    derived = {
        "schema_version": 1,
        "kind": "trimmed_platform_snapshot",
        "version": VERSION,
        "station_alias": meta["station_alias"],
        "site_alias": group,
        "start": start,
        "end_exclusive": end,
        "audit": row["audit"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units_verified": meta.get("units_verified", False),
        "reference_m": meta.get("reference_m"),
        "reference_id": meta.get("reference_id"),
        "unit_assumption": meta.get("unit_assumption"),
        "source_timezone": "Asia/Shanghai",
        "provenance": row,
        "files": {"observations.csv": file_hash(out / "observations.csv")},
    }
    write_json(out / "manifest.json", derived)
    return row, part


def group_splits(groups, development_group, seed):
    """Known-case group stays in development; holdouts contain whole other groups."""
    remaining = sorted(set(groups) - {development_group})
    if len(remaining) < 2:
        raise ValueError("need two independent groups outside the known-case group")
    np.random.default_rng(seed).shuffle(remaining)
    assignment = {g: "development" for g in groups}
    assignment[remaining[0]] = "validation"
    assignment[remaining[1]] = "test"
    return assignment


def prepare_offline(source: Path, stations: Path, references: Path, out: Path, seed=20260922):
    if out.exists():
        raise FileExistsError(out)
    metadata = read_json(stations)
    mapping = {"S-" + digest(r["StationUUID"])[:8]: r for r in metadata}
    refs = read_json(references).get("cases", [])
    known = {r["station_alias"] for r in refs}
    rows, parts, manifests = [], {}, {}
    for path in sorted(source.glob("*/manifest.json")):
        meta = read_json(path)
        alias = meta["station_alias"]
        station = mapping[alias]
        if not station.get("StationGroupUUID"):
            raise ValueError("station has no platform group; supply a reviewed source mapping")
        group = "G-" + digest(station["StationGroupUUID"])[:8]
        row, part = trim_snapshot(path.parent, out / "snapshots" / alias, group)
        rows.append(row)
        if part is not None:
            parts[alias] = part
            manifests[alias] = read_json(out / "snapshots" / alias / "manifest.json")
    if not rows:
        raise ValueError("no source snapshots")
    known_groups = {r["group"] for r in rows if r["station_alias"] in known}
    if len(known_groups) != 1:
        raise ValueError("expected one known-case source group")
    # Use groups with at least one complete short background, not empty/very sparse sites.
    eligible = {
        r["group"]
        for r in rows
        if r["station_alias"] not in known
        and r["audit"]
        and r["audit"]["longest_complete_run_hours"] >= 336
    }
    assignments = group_splits(eligible | known_groups, next(iter(known_groups)), seed)
    short, long, cases = [], [], []

    def save_window(alias, part, scale, start_index):
        m = manifests[alias]
        key = digest(
            {
                "version": VERSION,
                "source": m["provenance"]["source_manifest_sha256"],
                "start": str(part.timestamp.iloc[0]),
                "hours": len(part),
            }
        )[:24]
        relative = f"windows/{key}.json"
        write_json(out / scale / relative, as_window(part, key).model_dump(mode="json"))
        return {
            "id": key,
            "file": relative,
            "sha256": file_hash(out / scale / relative),
            "group": m["site_alias"],
            "split": assignments.get(m["site_alias"], "excluded"),
            "station_alias": alias,
            "start": local_iso(part.timestamp.iloc[0]),
            "end_exclusive": local_iso(
                pd.Timestamp(part.timestamp.iloc[-1]).to_pydatetime() + timedelta(hours=1)
            ),
            "offset_in_trimmed_hours": start_index,
            "audit": quality(part),
            "source_manifest_sha256": m["provenance"]["source_manifest_sha256"],
            "trimmed_manifest_sha256": file_hash(out / "snapshots" / alias / "manifest.json"),
            "source_units_verified": m["units_verified"],
            "approved": False,
            "review_note": "完整性筛选不等于正常背景审核；未生成真值。",
        }

    for alias, frame in parts.items():
        if alias in known:
            for ref in (r for r in refs if r["station_alias"] == alias):
                times = pd.to_datetime(frame.timestamp, utc=True)
                selected = (times >= pd.Timestamp(ref["start"])) & (
                    times < pd.Timestamp(ref["end_exclusive"])
                )
                part = frame.loc[selected]
                if len(part):
                    item = save_window(alias, part, "real-cases", int(np.flatnonzero(selected)[0]))
                    item.update(split="auxiliary", reference=ref, has_dense_ground_truth=False)
                    cases.append(item)
            continue  # Entire known station is withheld from presumed-normal backgrounds.
        full = np.isfinite(frame[COLUMNS].to_numpy(float)).all(axis=1)
        for first, stop in runs(full):
            for i in range(int(first), int(stop) - 336 + 1, 336):
                short.append(save_window(alias, frame.iloc[i : i + 336], "short", i))
        for i in range(0, len(frame) - 2880 + 1, 2880):
            part = frame.iloc[i : i + 2880]
            item_quality = quality(part)
            if item_quality["complete_hours"] / 2880 >= 0.8:
                observed = np.flatnonzero(np.isfinite(part[COLUMNS].to_numpy(float)).any(axis=1))
                left, right = int(observed[0]), int(observed[-1]) + 1
                item = save_window(alias, part.iloc[left:right], "long", i + left)
                item["target_hours"] = 2880
                item["edge_removed_hours"] = 2880 - right + left
                item["usage"] = "unlabelled_long_observation_only"
                long.append(item)
    common = {
        "version": VERSION,
        "seed": seed,
        "group_splits": assignments,
        "has_dense_ground_truth": False,
    }
    write_json(
        out / "short/backgrounds.json", {**common, "kind": "platform_backgrounds", "items": short}
    )
    write_json(
        out / "long/windows.json", {**common, "kind": "unlabelled_long_windows", "items": long}
    )
    write_json(
        out / "real-cases/windows.json", {**common, "kind": "known_real_cases", "items": cases}
    )

    def counts(items):
        return dict(Counter(r["split"] for r in items))

    report = {
        **common,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_unlabelled_review_pending",
        "stations_sha256": file_hash(stations),
        "references_sha256": file_hash(references),
        "group_basis": "platform StationGroupUUID; conservative source isolation",
        "stations": rows,
        "source_stations": len(rows),
        "retained_stations": len(parts),
        "excluded_empty_stations": len(rows) - len(parts),
        "original_hours": sum(r["original_hours"] for r in rows),
        "retained_hours": sum(r["audit"]["expected_hours"] for r in rows if r["audit"]),
        "observed_hours": sum(r["audit"]["observed_timestamps"] for r in rows if r["audit"]),
        "removed_hours": sum(r["removed_hours"] for r in rows),
        "short_windows": len(short),
        "short_splits": counts(short),
        "long_windows": len(long),
        "long_splits": counts(long),
        "auxiliary_cases": len(cases),
        "rules": {
            "edge": "any finite N/E/U retains hour",
            "short_hours": 336,
            "long_hours": 2880,
            "long_min_complete_fraction": 0.8,
            "imputation": False,
            "automatic_normal_approval": False,
        },
    }
    write_json(out / "preparation-report.json", report)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "audit"} | (r["audit"] or {}) for r in rows]
    ).to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    for scale, items in (("short", short), ("long", long), ("real-cases", cases)):
        pd.DataFrame(
            [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "station_alias",
                        "group",
                        "split",
                        "start",
                        "end_exclusive",
                        "file",
                        "approved",
                    )
                }
                | r["audit"]
                for r in items
            ]
        ).to_csv(out / scale / "index.csv", index=False, encoding="utf-8-sig")
    files = {
        p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()
    }
    write_json(out / "bundle.json", {"version": VERSION, "files": files})
    return report


def verify_preparation(root: Path):
    verify_files(root, read_json(root / "bundle.json")["files"])
    seen = {}
    for catalog in ("short/backgrounds.json", "long/windows.json", "real-cases/windows.json"):
        for item in read_json(root / catalog)["items"]:
            Window.model_validate(read_json(child((root / catalog).parent, item["file"])))
            if item["split"] == "auxiliary":
                continue
            if seen.setdefault(item["group"], item["split"]) != item["split"]:
                raise ValueError("source group leakage across scales")
    return read_json(root / "preparation-report.json")
