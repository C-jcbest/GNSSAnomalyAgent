"""Exact Beijing 15:00 extraction; no aggregation, imputation or truth creation."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Window
from .preparation import COLUMNS, as_window, runs
from .storage import child, digest, file_hash, read_json, verify_files, write_json

VERSION = "daily15-v1"
TZ = "Asia/Shanghai"


def daily_quality(frame):
    valid = np.isfinite(frame[COLUMNS].to_numpy(float))
    return {
        "expected_days": len(frame),
        "observed_timestamps": int(valid.any(axis=1).sum()),
        "complete_days": int(valid.all(axis=1).sum()),
        "missing_fraction": float(1 - valid.mean()),
        "longest_empty_gap_days": int(max((b - a for a, b in runs(~valid.any(axis=1))), default=0)),
    }


def select_daily(frame):
    as_window(frame, "source-validation")
    times = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(TZ)
    selected = frame.loc[times.dt.hour == 15].copy()
    valid = np.flatnonzero(np.isfinite(selected[COLUMNS].to_numpy(float)).any(axis=1))
    if not len(valid):
        return selected.iloc[:0]
    return selected.iloc[int(valid[0]) : int(valid[-1]) + 1].copy()


def daily_window(frame, key):
    return Window(
        case_id=key,
        sampling_hours=24,
        timestamps=list(pd.to_datetime(frame.timestamp, utc=True)),
        values=[
            [float(v) if np.isfinite(v) else None for v in row]
            for row in frame[COLUMNS].to_numpy(float)
        ],
    )


def prepare_daily(source: Path, out: Path, days=180):
    if days < 120:
        raise ValueError("long-term daily windows require at least 120 calendar days")
    if out.exists():
        raise FileExistsError(out)
    verify_files(source, read_json(source / "bundle.json")["files"])
    original = read_json(source / "preparation-report.json")
    if original.get("sampling_hours", 1) != 1:
        raise ValueError("daily extraction requires the original hourly preparation")
    groups = original["group_splits"]
    references = read_json(source / "real-cases/windows.json")["items"]
    known = {r["station_alias"] for r in references}
    rows, windows, auxiliary = [], [], []
    created = datetime.now(timezone.utc).isoformat()
    for manifest in sorted((source / "snapshots").glob("*/manifest.json")):
        meta = read_json(manifest)
        alias, group = meta["station_alias"], meta["site_alias"]
        frame = pd.read_csv(manifest.parent / "observations.csv", float_precision="round_trip")
        part = select_daily(frame)
        split = "auxiliary" if alias in known else groups.get(group, "excluded")
        row = {"station_alias": alias, "group": group, "split": split}
        if part.empty:
            rows.append({**row, "status": "no_valid_15h", "expected_days": 0})
            continue
        folder = out / "snapshots" / alias
        folder.mkdir(parents=True)
        # Preserve selected CSV lines verbatim, including all raw coordinate columns.
        lines = (manifest.parent / "observations.csv").read_text(encoding="utf-8").splitlines(True)
        if len(lines) != len(frame) + 1:
            raise ValueError("unexpected multiline CSV")
        (folder / "observations.csv").write_text(
            lines[0] + "".join(lines[i + 1] for i in part.index), encoding="utf-8", newline=""
        )
        times = pd.to_datetime(part.timestamp, utc=True).dt.tz_convert(TZ)
        audit = daily_quality(part)
        daily_window(part, alias)  # Validate gaps remain on the daily grid.
        derived = {
            "kind": "daily15_platform_snapshot",
            "version": VERSION,
            "sampling_hours": 24,
            "source_timezone": TZ,
            "selection_hour": 15,
            "station_alias": alias,
            "site_alias": group,
            "split": split,
            "start": times.iloc[0].isoformat(),
            "end_exclusive": (times.iloc[-1].to_pydatetime() + timedelta(days=1)).isoformat(),
            "last_observation": times.iloc[-1].isoformat(),
            "created_at": created,
            "audit": audit,
            "units_verified": meta.get("units_verified", False),
            "provenance": {
                "source_manifest_sha256": file_hash(manifest),
                "source_csv_sha256": file_hash(manifest.parent / "observations.csv"),
            },
            "files": {"observations.csv": file_hash(folder / "observations.csv")},
        }
        write_json(folder / "manifest.json", derived)
        row.update(start=derived["start"], end_exclusive=derived["end_exclusive"], **audit)
        row["status"] = "ready_for_review" if len(part) >= days else "short_history"
        rows.append(row)
        # Exactly one latest full-length candidate per station, chosen without inspecting values.
        # The auxiliary case uses all history, including the pre-June period.
        if alias not in known and (len(part) < days or split == "excluded"):
            continue
        selected = part if alias in known else part.iloc[-days:]
        if alias not in known and daily_quality(selected)["complete_days"] / days < 0.8:
            row["status"] = "low_window_coverage"
            continue
        key = digest({"source": file_hash(folder / "manifest.json"), "days": len(selected)})[:24]
        scale = "real-cases" if alias in known else "long"
        target = out / scale / "windows" / f"{key}.json"
        write_json(target, daily_window(selected, key).model_dump(mode="json"))
        window_times = pd.to_datetime(selected.timestamp, utc=True).dt.tz_convert(TZ)
        item = {
            "id": key,
            "station_alias": alias,
            "group": group,
            "split": split,
            "file": f"windows/{key}.json",
            "sha256": file_hash(target),
            "sampling_hours": 24,
            "target_days": days,
            "start": window_times.iloc[0].isoformat(),
            "end_exclusive": (
                window_times.iloc[-1].to_pydatetime() + timedelta(days=1)
            ).isoformat(),
            "audit": daily_quality(selected),
            "approved": False,
            "has_dense_ground_truth": False,
            "source_manifest_sha256": file_hash(folder / "manifest.json"),
            "usage": "unlabelled_daily_long_observation_only",
        }
        if alias in known:
            item["references"] = [
                r.get("reference", {}) for r in references if r["station_alias"] == alias
            ]
            auxiliary.append(item)
        else:
            windows.append(item)
    common = {
        "version": VERSION,
        "sampling_hours": 24,
        "group_splits": groups,
        "has_dense_ground_truth": False,
    }
    for scale, items in (("long", windows), ("real-cases", auxiliary)):
        write_json(out / scale / "windows.json", {**common, "items": items})
        pd.DataFrame(
            [
                {k: v for k, v in r.items() if k not in ("audit", "references")} | r["audit"]
                for r in items
            ]
        ).to_csv(out / scale / "index.csv", index=False)
    report = {
        **common,
        "created_at": created,
        "status": "daily_prepared_unlabelled_review_pending",
        "source_bundle_sha256": file_hash(source / "bundle.json"),
        "retained_stations": sum(r["status"] != "no_valid_15h" for r in rows),
        "excluded_empty_stations": sum(r["status"] == "no_valid_15h" for r in rows),
        "retained_days": sum(r["expected_days"] for r in rows),
        "observed_days": sum(r.get("observed_timestamps", 0) for r in rows),
        "window_days": days,
        "long_windows": len(windows),
        "long_splits": dict(Counter(r["split"] for r in windows)),
        "auxiliary_cases": len(auxiliary),
        "stations": rows,
        "rules": {
            "timezone": TZ,
            "hour": 15,
            "aggregation": False,
            "imputation": False,
            "window_selection": "latest calendar window per station",
            "automatic_normal_approval": False,
            "long_min_complete_fraction": 0.8,
        },
    }
    write_json(out / "preparation-report.json", report)
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    write_json(
        out / "bundle.json",
        {
            "version": VERSION,
            "files": {
                p.relative_to(out).as_posix(): file_hash(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        },
    )
    return report


def verify_daily(root):
    verify_files(root, read_json(root / "bundle.json")["files"])
    report = read_json(root / "preparation-report.json")
    for scale in ("long", "real-cases"):
        for item in read_json(root / scale / "windows.json")["items"]:
            window = Window.model_validate(read_json(child(root / scale, item["file"])))
            if window.sampling_hours != 24 or any(t.hour != 7 for t in window.timestamps):
                raise ValueError("expected Beijing 15:00 daily observations")
            if scale == "long" and (
                len(window.values) != report["window_days"]
                or item["split"] != report["group_splits"][item["group"]]
            ):
                raise ValueError("daily length or source split mismatch")
    return report
