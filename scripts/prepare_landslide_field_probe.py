"""Prepare official NGL components for frozen, explicitly unlabelled field diagnostics."""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from check_landslide_field_dates import ngl_calendar
from fit_landslide_raw_persistence_baselines import raw_gate
from run_landslide_adaptive_review import PLOT
from run_landslide_comparison import ROOT, observation_features, predict_baselines, save_json, sha
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_adaptive_review import raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_local_experiment import learned_stages

START_DATE = date(2018, 3, 1)
DAYS = 730
ANCHOR_DAYS = 30
STATIONS = ("OREO", "ORE2", "P227")
COORDINATE_FIELDS = (("____n0(m)", "_north(m)"), ("_e0(m)", "__east(m)"), ("u0(m)", "____up(m)"))


def load_ngl_records(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = lines[0].split()
    required = {"site", "YYMMMDD", "__MJD", "reflon", "sig_n(m)", "sig_e(m)", "sig_u(m)"}
    required.update(name for pair in COORDINATE_FIELDS for name in pair)
    if not required.issubset(headers) or len(set(headers)) != len(headers):
        raise ValueError("Official NGL header is incomplete or duplicated")
    records = {}
    for ordinal, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != len(headers):
            raise ValueError(f"Malformed NGL row {ordinal}")
        row = dict(zip(headers, tokens))
        day = ngl_calendar(row["YYMMMDD"]).date()
        if row["site"] != path.stem or (day - date(1858, 11, 17)).days != int(row["__MJD"]):
            raise ValueError(f"Station/date provenance mismatch at row {ordinal}")
        if day in records:
            raise ValueError(f"Duplicate official observation date: {day}")
        coordinates = np.array([float(row[left]) + float(row[right]) for left, right in COORDINATE_FIELDS])
        sigma = np.array([float(row[name]) for name in ("sig_n(m)", "sig_e(m)", "sig_u(m)")])
        if not np.isfinite(coordinates).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError(f"Invalid coordinate or formal sigma at row {ordinal}")
        records[day] = {"coordinates_m": coordinates, "sigma_m": sigma,
                        "reference_meridian": float(row["reflon"]), "source_row": ordinal,
                        "source_date_code": row["YYMMMDD"]}
    return records


def daily_coordinates(records, calendar):
    coordinates = np.full((len(calendar), 3), np.nan)
    for index, day in enumerate(calendar):
        if day in records:
            coordinates[index] = records[day]["coordinates_m"]
    return coordinates


def horizontal_reference_rate(reference_coordinates):
    observed = np.isfinite(reference_coordinates).all(axis=1)
    if observed.sum() < 60:
        raise ValueError("Stable-reference rate needs at least 60 observed dates")
    time = np.arange(len(reference_coordinates), dtype=float)
    centered_time = time - time[observed].mean()
    matrix = np.column_stack((np.ones(observed.sum()), centered_time[observed]))
    relative = reference_coordinates[observed] - reference_coordinates[observed][0]
    coefficients = np.linalg.lstsq(matrix, relative, rcond=None)[0]
    # The paper removes horizontal reference motion. Vertical is deliberately retained.
    return np.array([coefficients[1, 0], coefficients[1, 1], 0.0])


def relative_displacement(coordinates, horizontal_rate_m_day=None):
    values = coordinates.copy()
    if horizontal_rate_m_day is not None:
        rate = np.asarray(horizontal_rate_m_day)
        if rate.shape != (3,) or rate[2] != 0 or not np.isfinite(rate).all():
            raise ValueError("Reference correction must be finite horizontal N/E only")
        values -= np.arange(len(values))[:, None] * rate
    anchor = values[:ANCHOR_DAYS]
    complete = anchor[np.isfinite(anchor).all(axis=1)]
    if len(complete) < 8:
        raise ValueError("First 30 calendar days need at least eight complete observations")
    origin = np.median(complete, axis=0)
    displacement = (values - origin) * 1000
    serialized = [row.tolist() if np.isfinite(row).all() else None for row in displacement]
    return serialized, origin.tolist()


def prepare(source, training, raw_baselines, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(ROOT / "configs/landslide-adaptive-review-confirmation-v1.json")
    case_ids = [f"case_{index:04d}" for index in range(101, 107)]
    protocol.update({
        "date": "2026-10-09", "case_ids": case_ids, "maximum_requests": 84,
        "hard_timeout_seconds": 1800,
        "scope": "Unlabelled real NGL field transfer diagnostics. No expert activity or ASD "
                 "reference, no precision/recall/F1 or correctness claim. Raw and horizontal "
                 "P227-rate-corrected records share observations. P227 is contextual stable "
                 "bedrock, not a zero-motion label; plate/seasonal/instrument signals can remain.",
        "source_bundle": str(source.resolve()),
        "start_date": str(START_DATE), "days": DAYS,
        "reference_policy": "Full-period observed-date OLS N/E P227 secular rate removed. "
                            "U unchanged; no daily reference subtraction or interpolation. "
                            "Not original paper IGS14 processing, not a field-calibrated model.",
        "anchor_policy": "Per record/component median of observed first 30 calendar days",
        "stage_policy": "Synthetic train24 stage head transferred without field tuning; "
                        "descriptive output only, never field stage accuracy",
        "primary_metrics": "Predicted intervals, repeat/representation state differences, "
                           "requests/tokens; no accuracy labels or precision/recall/F1",
        "secondary_metrics": "Transferred stage counts and masks only; no field ASD F1",
        "freeze": "All four arms, two repeats and policy unchanged from confirmation. "
                  "No method/repeat selection from field outputs. No reference-label file exists.",
    })
    save_json(out / "protocol.json", protocol)
    calendar = [START_DATE + timedelta(days=index) for index in range(DAYS)]
    retrieval = read_json(source / "current-retrieval-v2.json")
    expected = {row["file"]: row["sha256"] for row in retrieval if row.get("status") == 200}
    records, coordinates = {}, {}
    for station in STATIONS:
        path = source / f"{station}.tenv3"
        if sha(path) != expected[path.name]:
            raise ValueError("Official source differs from retained download")
        records[station] = load_ngl_records(path)
        coordinates[station] = daily_coordinates(records[station], calendar)
    meridians = {records[station][day]["reference_meridian"] for station in STATIONS
                 for day in calendar if day in records[station]}
    if len(meridians) != 1:
        raise ValueError("Do not subtract graticule rates across different reference meridians")
    reference_rate = horizontal_reference_rate(coordinates["P227"])
    save_json(out / "reference-correction.json", {
        "station": "P227", "rate_NEU_mm_day": (reference_rate * 1000).tolist(),
        "observed_fit_dates": int(np.isfinite(coordinates["P227"]).all(axis=1).sum()),
        "scope": "OLS secular rate, not labels or independent movement evidence",
    })
    (out / "images").mkdir()
    features, mapping, trace = {}, {}, []
    for ordinal, station in enumerate(STATIONS):
        for corrected in (False, True):
            cid = case_ids[ordinal * 2 + int(corrected)]
            displacement, origin = relative_displacement(
                coordinates[station], reference_rate if corrected else None)
            case = LandslideInput(case_id=cid, dates=calendar, displacement_mm=displacement)
            folder = out / "inputs" / cid
            folder.mkdir(parents=True)
            (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
            motions = {}
            for window in (31, 61, 91):
                motion = derive_motion(case, window)
                motions[window] = motion
                (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
            feature = observation_features(case, motions)
            features[cid] = feature
            mapping[cid] = {
                "station": station, "representation": "P227_horizontal_rate_removed" if corrected else "IGS20_raw",
                "source_sha256": sha(source / f"{station}.tenv3"),
                "source_coordinates_order": "northing,easting,up; integer plus fractional meters",
                "relative_origin_m": origin, "observed_days": int(feature.observed.sum()),
                "input_sha256": sha(folder / "input.json"), "independent_record": False,
            }
            render_raw(case, out / "images" / f"{cid}-raw.png")
            for index, (start, stop) in enumerate(regular_windows(DAYS, 180, 90)):
                render_window(case, feature.noise, start, stop,
                              out / "images" / f"{cid}-window-{index:02d}.png", PLOT)
            for index, day in enumerate(calendar):
                original = records[station].get(day)
                trace.append({"case_id": cid, "day_index": index, "date": str(day),
                              "station": station, "source_row": original["source_row"] if original else "",
                              "source_date_code": original["source_date_code"] if original else "",
                              "observed": int(original is not None)})
    save_json(out / "field-mapping.json", mapping)
    with (out / "source-calendar.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    choices = read_json(training / "frozen-selection.json")
    activity_model, stage_model, long_model = XGBClassifier(), XGBClassifier(), XGBClassifier()
    activity_model.load_model(training / "activity-xgboost.json")
    stage_model.load_model(training / "stage-xgboost.json")
    long_model.load_model(raw_baselines / "activity-xgboost-raw-persistence.json")
    methods = predict_baselines(features, case_ids, choices, activity_model, stage_model)
    del methods["ungated_rule"]
    long_choices = read_json(raw_baselines / "frozen-selection.json")
    methods["raw_multiscale_rule"], methods["xgboost_raw_persistence"] = {}, {}
    for cid in case_ids:
        case = LandslideInput.model_validate_json((out / "inputs" / cid / "input.json").read_text(encoding="utf-8"))
        feature = features[cid]
        evidence = raw_motion_evidence(case, feature)
        rule, learned = long_choices["raw_multiscale_rule"], long_choices["xgboost_raw_persistence"]
        matrix = np.column_stack((feature.matrix, evidence["rate"], evidence["signal_to_noise"], evidence["direction"]))
        gates = {
            "raw_multiscale_rule": raw_gate(feature, evidence, rule["rate"], rule["snr"]),
            "xgboost_raw_persistence": activity_from_mask(
                long_model.predict_proba(matrix)[:, 1] >= learned["threshold"], feature.supported),
        }
        codes = stage_model.predict(feature.matrix).astype(int) + 1
        for method, gate in gates.items():
            methods[method][cid] = learned_stages(codes, gate, feature.supported)
    save_json(out / "numeric-predictions.json", {
        method: {cid: value.to_dict() for cid, value in cases.items()} for method, cases in methods.items()})
    shutil.copyfile(training / "stage-xgboost.json", out / "stage-xgboost.json")
    (out / "source").mkdir()
    paths = [Path(__file__), ROOT / "scripts/check_landslide_field_dates.py"]
    original_sources = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/confirmation-v1/source"
    paths.extend(original_sources.glob("*.py"))
    for path in paths:
        shutil.copyfile(path, out / "source" / path.name)
    save_json(out / "training-bindings.json", {
        str(path.resolve()): sha(path) for path in (
            training / "frozen-selection.json", training / "stage-xgboost.json",
            training / "activity-xgboost.json", raw_baselines / "frozen-selection.json",
            raw_baselines / "activity-xgboost-raw-persistence.json")})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "scope": protocol["scope"]})
    print("Prepared six real representations without reference labels or field tuning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--raw-baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.training, args.raw_baselines, args.out)


if __name__ == "__main__":
    main()
