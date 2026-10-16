"""Profile the published Yutang CSVs without converting units or deriving labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = ["time", "displacement_XY", "elevation", "rainfall", "reservoir_level"]


def read_station(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, observations = rows[0], rows[1:]
    if header[:5] != FIELDS or any(header[5:]):
        raise ValueError(f"Unexpected published columns: {path}")
    for row in observations:
        if len(row) != len(header) or any(row[5:]):
            raise ValueError(f"Unexpected row width or populated extra field: {path}")
    dates = [datetime.strptime(row[0], "%Y/%m/%d").date() for row in observations]
    values = {}
    for column, name in enumerate(FIELDS[1:], start=1):
        values[name] = [float(row[column]) if row[column].strip() else None
                        for row in observations]
        if any(value is not None and not math.isfinite(value) for value in values[name]):
            raise ValueError(f"Nonfinite source value: {path}, {name}")
    return header, dates, values


def summarize_values(values):
    present = [value for value in values if value is not None]
    differences = [right - left for left, right in zip(values, values[1:])
                   if left is not None and right is not None]
    longest_run = 0
    current_run = 0
    previous = None
    for value in values:
        if value is None:
            current_run = 0
        elif current_run and value == previous:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
        previous = value
    return {
        "missing_rows": len(values) - len(present),
        "range_source_units": [min(present), max(present)] if present else None,
        "adjacent_increases": sum(value > 0 for value in differences),
        "adjacent_decreases": sum(value < 0 for value in differences),
        "adjacent_exact_repeats": sum(value == 0 for value in differences),
        "longest_exact_repeat_run_rows": longest_run,
        "interpretation": "File statistics only; repeated values are not inactivity labels.",
    }


def audit(source: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    stations = {}
    profiles = {}
    for station in (2, 3, 4):
        path = source / f"data_{station}.csv"
        header, dates, values = read_station(path)
        profiles[station] = (dates, values)
        intervals = Counter((right - left).days for left, right in zip(dates, dates[1:]))
        stations[str(station)] = {
            "source_path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "header": header,
            "rows": len(dates),
            "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
            "duplicate_dates": len(dates) - len(set(dates)),
            "interval_days_counts": dict(sorted(intervals.items())),
            "columns": {name: summarize_values(values[name]) for name in FIELDS[1:]},
        }
    shared_environment = {}
    first_dates, first_values = profiles[2]
    for field in ("rainfall", "reservoir_level"):
        shared_environment[field] = all(
            dates == first_dates and values[field] == first_values[field]
            for dates, values in profiles.values()
        )
    report = {
        "source_doi": "10.5281/zenodo.21482916",
        "site_count": 1,
        "file_count": 3,
        "stations": stations,
        "shared_environment_exact_equal": shared_environment,
        "calendar_resolution": "Daily file rows; independent daily measurements not established.",
        "units": "Unresolved: paper reports mm; GitHub schema says metres; notebook mixes m/mm.",
        "elevation": "GitHub describes surface elevation; actual varying values need clarification.",
        "missingness": "No blank required cells does not establish no acquisition gaps or interpolation.",
        "activity_reference": "Not supplied in these CSVs or inspected code-tree inventory.",
        "stage_reference": "Not supplied; trend/periodic/random decomposition is a different target.",
        "decision": "Retain source data; no ENU conversion, model inference, training or accuracy scoring.",
        "new_model_requests": 0,
    }
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    figure, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for row, (station, (dates, values)) in enumerate(profiles.items()):
        for column, field in enumerate(("displacement_XY", "elevation")):
            axes[row, column].plot(dates, values[field], linewidth=0.65)
            axes[row, column].set_title(f"Published data_{station}.csv: {field}")
            axes[row, column].set_ylabel("Source values (unit unresolved)")
            axes[row, column].grid(alpha=0.2)
    figure.suptitle("Yutang published CSVs: no smoothing, conversion, labels or predictions")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(out / "source-values.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"stations": len(stations), "rows": [item["rows"] for item in stations.values()],
                      "new_model_requests": 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.out)


if __name__ == "__main__":
    main()
