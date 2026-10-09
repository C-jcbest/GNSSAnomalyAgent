"""Complete frozen-activity support audit, without stage labels or model calls."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from landslide_stage_evidence import load_case
from landslide_stage_grounding import RUN as SOURCE_RUN
from landslide_stage_grounding import verify_prepared
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_window_support_audit import WINDOWS, audit_record, summarize

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/landslide-window-support-audit-2026-10-09/audit-v2"
SOURCE_FILES = (
    "docs/landslide-window-support-audit-protocol.md",
    "src/gnss_sim/landslide_window_support_audit.py",
    "tests/test_landslide_window_support_audit.py",
    "scripts/audit_landslide_window_support.py",
    "scripts/landslide_window_support_page.html",
    "scripts/verify_landslide_window_support_audit.py",
    "scripts/check_landslide_window_support_ui.cjs",
    "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_stage_reference.py",
    "scripts/landslide_stage_evidence.py", "scripts/landslide_stage_grounding.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def contiguous_ranges(days):
    """Half-open runs never bridge a missing or excluded calendar day."""
    result = []
    for day in sorted(days):
        if result and result[-1][1] == day:
            result[-1][1] = day + 1
        else:
            result.append([day, day + 1])
    return result


def support_ranges(rows, pairs):
    selected = {(row["window_days"], row["day_index"]): row for row in rows if row["observed"]}
    days = sorted({row["day_index"] for row in rows if row["observed"]})
    return {
        "additional_31_over_61_velocity": contiguous_ranges(day for day in days
            if selected[31, day]["velocity_available"] and not selected[61, day]["velocity_available"]),
        "additional_31_over_61_tangential": contiguous_ranges(day for day in days
            if selected[31, day]["tangential_available"] and not selected[61, day]["tangential_available"]),
        "raw_opposite_sign_pairs": {f"{first}-{second}": contiguous_ranges(
            row["day_index"] for row in pairs if row["first_window"] == first
            and row["second_window"] == second and row["tangential_opposite_sign"])
            for first, second in ((31, 61), (31, 91), (61, 91))},
    }


def render_audit(case, rows, spans, path):
    """New audit figure; original model-input axes and images remain frozen."""
    figure = Figure(figsize=(15, 10), dpi=130, facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1.4, .7, 1, 1]})
    days = np.arange(len(case.dates))
    observed = np.array([value if value is not None else [np.nan] * 3 for value in case.displacement_mm])
    for axis, color in enumerate(("#147d72", "#cc6d4b", "#a38a32")):
        panels[0].plot(days, observed[:, axis], linewidth=.7, color=color, label=("N", "E", "U")[axis])
    panels[0].set_ylabel("Observed displacement (mm)")
    panels[0].margins(y=.15)
    panels[0].legend(loc="upper left", ncols=3)
    panels[0].set_title(f"{case.case_id} | observation-only audit | NOT sent to a model")
    support = np.zeros((3, len(days)), dtype=int)
    for index, window in enumerate(WINDOWS):
        current = [row for row in rows if row["window_days"] == window]
        for row in current:
            day = row["day_index"]
            if row["activity_state"] != "activity":
                support[index, day] = -1
            elif not row["observed"]:
                support[index, day] = 0
            elif not row["velocity_available"]:
                support[index, day] = 1
            elif not row["tangential_available"]:
                support[index, day] = 2
            elif not row["contained_in_same_interior"]:
                support[index, day] = 3
            else:
                support[index, day] = 4
        speed = [row["speed_mm_day"] if row["speed_mm_day"] is not None else np.nan for row in current]
        tangential = [row["tangential_mm_day2"] if row["tangential_mm_day2"] is not None else np.nan for row in current]
        panels[2].plot(days, speed, linewidth=.9, label=f"{window}d")
        panels[3].plot(days, tangential, linewidth=.9, label=f"{window}d")
    colors = ListedColormap(["#edf0f3", "#202b35", "#c7cdd4", "#b98a3c", "#669aba", "#177b69"])
    panels[1].imshow(support, aspect="auto", origin="upper", interpolation="nearest",
                     extent=(-.5, len(days)-.5, 2.5, -.5), cmap=colors,
                     norm=BoundaryNorm(np.arange(-1.5, 5), colors.N))
    panels[1].set_yticks([0, 1, 2], ["31d", "61d", "91d"])
    panels[1].set_ylabel("Activity support")
    panels[1].set_title("grey: outside / black: missing / light grey: no fit / gold: v only / blue: v+a crossing / green: v+a contained", fontsize=9)
    panels[2].set_ylabel("Speed (mm/day)")
    panels[3].set_ylabel("Tangential a (mm/day^2)")
    panels[3].axhline(0, color="#777", linewidth=.6)
    panels[3].set_xlabel("Day index; full centered windows use future observations")
    for panel in (panels[0], panels[2], panels[3]):
        for span in spans:
            panel.axvspan(span["start"]-.5, span["stop"]-.5, color="#177b69", alpha=.07)
        panel.grid(alpha=.15)
    panels[2].legend(ncols=3, loc="upper left")
    panels[3].legend(ncols=3, loc="upper left")
    panels[0].set_xlim(-.5, len(days)-.5)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, tuple)) else value
                             for key, value in row.items()})


def run():
    started = time.monotonic()
    _, source_public, records, entries = verify_prepared()
    RUN.mkdir(parents=True, exist_ok=False)
    public = RUN / "public"
    public.mkdir()
    source_hashes = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        source_hashes[name] = sha(path)
        destination = RUN / "source-code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    input_hashes = {}
    source_names = {"source/source-manifest.json", "source/activity-manifest.json", "source/activity-freeze.json"}
    for record in records:
        entry = entries[record["case_id"]]
        source_names.add("source/" + record["input"])
        source_names.update("source/" + record["diagnostics"][str(window)]["data"] for window in WINDOWS)
        source_names.update("source/activity-source/" + entry[key] for key in ("review", "daily"))
        source_names.update(view["image"] for view in record["views"])
        source_names.update(record["diagnostics"][str(window)]["image"] for window in WINDOWS)
    for name in sorted(source_names):
        source = source_public / name
        input_hashes[name] = sha(source)
        destination = public / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(RUN / "run-ledger-start.json", {
        "started_utc": utc_now(), "source_run": SOURCE_RUN.relative_to(ROOT).as_posix(),
        "source_inference_freeze_sha256": sha(SOURCE_RUN / "inference-freeze.json"),
        "source_ledger_end_sha256": sha(SOURCE_RUN / "run-ledger-end.json"),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "stage_reference_access": "none", "stage_prediction_access": "none",
        "generator_truth_access": "none", "sealed_test_access": "none", "model_calls": 0,
    })
    all_rows, all_pairs, cases, interiors = [], [], [], []
    for record in records:
        if time.monotonic() - started > 180:
            raise TimeoutError("Observation audit exceeded 180 seconds")
        case, review, _, _, motions = load_case(source_public, record, entries)
        rows, pairs, spans = audit_record(case, review, motions)
        all_rows.extend(rows)
        all_pairs.extend(pairs)
        case_interiors = []
        for span in spans:
            inside_rows = [row for row in rows if span["start"] <= row["day_index"] < span["stop"]]
            inside_pairs = [row for row in pairs if span["start"] <= row["day_index"] < span["stop"]]
            summary = {"case_id": case.case_id, "start": span["start"], "stop": span["stop"],
                       **summarize(inside_rows, inside_pairs), "ranges": support_ranges(inside_rows, inside_pairs)}
            case_interiors.append(summary)
            interiors.append(summary)
        plot = f"audit-images/{case.case_id}.png"
        render_audit(case, rows, spans, public / plot)
        cases.append({"case_id": case.case_id, "calendar_days": len(case.dates),
                      "start_date": case.dates[0].isoformat(), "stop_date_inclusive": case.dates[-1].isoformat(),
                      "observed_days": sum(value is not None for value in case.displacement_mm),
                      "summary": summarize(rows, pairs), "interiors": case_interiors, "plot": plot,
                      "raw_images": [view["image"] for view in record["views"]],
                      "aux_images": [record["diagnostics"][str(window)]["image"] for window in WINDOWS]})
        print(f"{len(cases)}/12 {case.case_id}: {len(spans)} activities audited", flush=True)
    analysis = {"schema_version": "landslide-window-support-audit-v1", "cases": cases,
                "interiors": interiors, "summary": summarize(all_rows, all_pairs),
                "calendar_days": sum(case["calendar_days"] for case in cases),
                "window_daily_rows": len(all_rows), "pair_daily_rows": len(all_pairs),
                "model_calls": 0, "stage_reference_access": "none", "performance_scores": None}
    write_json(public / "analysis.json", analysis)
    write_csv(public / "daily-window-support.csv", all_rows)
    write_csv(public / "daily-window-pairs.csv", all_pairs)
    template = (ROOT / "scripts/landslide_window_support_page.html").read_text(encoding="utf-8")
    payload = json.dumps(analysis, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (public / "index.html").write_text(template.replace("__AUDIT_DATA__", payload), encoding="utf-8")
    files = {path.relative_to(public).as_posix(): sha(path) for path in sorted(public.rglob("*")) if path.is_file()}
    write_json(RUN / "audit-freeze.json", {"completed_utc": utc_now(), "files_sha256": files,
        "ledger_start_sha256": sha(RUN / "run-ledger-start.json"), "duration_seconds": time.monotonic()-started,
        "model_calls": 0, "old_configuration_replaced": False})
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
