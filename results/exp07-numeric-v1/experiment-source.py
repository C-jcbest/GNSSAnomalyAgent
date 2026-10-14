"""Isolated development candidates; production SDR-VLM remains unchanged."""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
from summarize_exp06 import pooled_bootstrap

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import Prediction, event_masks, observed_values
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.segmentation import regression, segment_proposals
from gnss_research.vision import SYSTEM, call_visual, credentials

COEXISTENCE_REVIEW = """
Permanent and temporary offsets are NOT mutually exclusive. A transient may begin
on exactly the same supplied jump index as a permanent step. Inspect whether the
post-return trajectory still has a persistent level offset relative to the
pre-first-jump trajectory, allowing the continuous motion shown in the raw panels.
If the pair contains a temporary platform AND a remaining permanent offset, report
both transient and step with the corresponding supplied indices. Do not preserve
a step merely because a numerical jump exists; require visible persistent evidence.
If the observations cannot establish which edge owns a permanent component, omit
the step or use uncertain. Do not assume every transient contains a permanent step.
"""


def rebuild(observation: dict, result: dict, events: list[dict]) -> dict:
    prediction = Prediction.model_validate({"events": events})
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {**result, "prediction": prediction.model_dump(),
            "generic_mask": np.any(np.column_stack(list(masks.values())), axis=1).tolist(),
            "generic_score": None}


def isolated_competition(observation: dict, numerical: dict, evidence=None) -> dict:
    """Supplement RF spikes only when one sample beats persistent local structures."""
    if evidence is None:
        evidence = numerical_evidence(observation)
    time = np.arange(len(evidence.values))
    eligible = evidence.observed & (time > evidence.background_end)
    standardized = evidence.residual / evidence.noise
    search_penalty = float(2 * np.log(eligible.sum()))
    receipts = []
    for day in np.flatnonzero(eligible):
        if day == 0 or day == len(time) - 1:
            continue
        if not evidence.observed[day - 1:day + 2].all():
            continue
        keep = evidence.observed & (np.abs(time - day) <= 7)
        local_time, values = time[keep], standardized[keep]
        if (local_time < day).sum() < 5 or (local_time > day).sum() < 5:
            continue
        jump = (local_time >= day).astype(float)
        hinge = np.maximum(local_time - day, 0)
        impulse = (local_time == day).astype(float)
        fits = {"linear": regression(local_time, values, []),
                "step": regression(local_time, values, [jump]),
                "hinge": regression(local_time, values, [hinge]),
                "step_hinge": regression(local_time, values, [jump, hinge]),
                "spike": regression(local_time, values, [impulse])}
        criteria = {name: fit["criterion"] for name, fit in fits.items()}
        criteria["spike"] += search_penalty
        selected = min(criteria, key=criteria.get)
        if selected == "spike":
            receipts.append({"index": int(day), "criteria": criteria})
    events = copy.deepcopy(numerical["prediction"]["events"])
    existing = {event["start"] for event in events if event["kind"] == "spike"}
    for receipt in receipts:
        day = receipt["index"]
        if day in existing:
            continue
        events.append({"kind": "spike", "start": day, "end": day, "status": "confirmed",
                       "evidence": "Isolated-sample model beats persistent local structures with position penalty",
                       "phases": []})
    result = rebuild(observation, numerical, events)
    result.update(isolated_sample_receipts=receipts,
                  isolated_sample_parameters={"half_window_days": 7, "min_each_side": 5,
                                              "position_penalty": search_penalty,
                                              "criterion": "RSS + 3p log(3n_local) + 2 log(N_eligible) for impulse"})
    return result


def coexist_repair(observation: dict, visual: dict, candidates: dict) -> dict:
    result = repair_return_pairs(observation, visual, candidates)
    if visual["status"] != "success":
        return result
    repaired_edges = {edge for repair in result["semantic_repairs"] for edge in repair["edges"]}
    confirmed_steps = {event["start"] for event in visual["prediction"]["events"]
                       if event["kind"] == "step" and event["status"] == "confirmed"}
    events = result["prediction"]["events"]
    retained = []
    for event in candidates["numerical_prediction"]["prediction"]["events"]:
        if event["kind"] != "step":
            continue
        if event["start"] not in repaired_edges or event["start"] not in confirmed_steps:
            continue
        events.append({**event, "evidence": "Visual confirmation of a permanent component coexisting with transient"})
        retained.append(event["start"])
    result = rebuild(observation, result, events)
    result.update(coexisting_step_indices=retained,
                  derivation="same-response nonexclusive step/transient repair; numerical event presence retained")
    return result


def records_for(run: Path):
    registration = read_json(run / "registration.json")
    if registration["split"] != "validation":
        raise ValueError("EXP07 development requires validation observations")
    dataset = Path(registration["dataset"])
    records = [record for record in read_json(dataset / "manifest.json")["records"]
               if record["station_id"] in registration["station_ids"]]
    return dataset, records


def score(dataset: Path, records: list[dict], predictions: dict[str, dict]) -> tuple[dict, dict]:
    rows = {method: [] for method in predictions}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        for method, results in predictions.items():
            rows[method].append(evaluate_record(truth, results[station], observed,
                                               observation["background_end"]))
    return {method: aggregate(values) for method, values in rows.items()}, rows


def numeric_stage(run: Path, out: Path):
    dataset, records = records_for(run)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "experiment-source.py")
    predictions = {"segment_numeric": {}, "isc_numeric": {}, "rf_split_temporal": {}}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numerical = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
        evidence = numerical_evidence(observation)
        augmented = isolated_competition(observation, numerical, evidence)
        segmented = segment_proposals(observation, augmented, evidence)
        write_json(out / "predictions/isc_source" / station / "result.json", augmented)
        write_json(out / "predictions/isc_numeric" / station / "result.json", segmented)
        predictions["isc_numeric"][station] = segmented
        for method in ("segment_numeric", "rf_split_temporal"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
        print(station, "supplemental_spikes", len(augmented["isolated_sample_receipts"]), flush=True)
    summary, rows = score(dataset, records, predictions)
    write_json(out / "summary.json", {"methods": summary,
               "primary_difference": summary["isc_numeric"]["types_daily_macro_f1"]
                                     - summary["segment_numeric"]["types_daily_macro_f1"],
               "requests": 0, "dataset": str(dataset), "source_hashes": source_hashes(),
               "script_sha256": sha256(Path(__file__)),
               "source_registration_sha256": sha256(run / "registration.json")})
    write_json(out / "per-station.json", rows)
    print({name: values["types_daily_macro_f1"] for name, values in summary.items()})


def prepare_stage(run: Path, numeric: Path, out: Path, env: Path):
    dataset, records = records_for(run)
    out.mkdir(parents=True, exist_ok=False)
    settings = credentials(env)
    shutil.copyfile(__file__, out / "experiment-source.py")
    needed = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for method, source in (("stable", run / "predictions/rf_split_temporal"),
                               ("isc", numeric / "predictions/isc_source")):
            work = out / "predictions" / method / station
            numerical = read_json(source / station / "result.json")
            payload = prepare_pair_focus(observation, numerical, work, settings[2], SYSTEM)
            coexist_work = out / "predictions" / (method + "_coexist") / station
            shutil.copytree(work, coexist_work)
            if payload is not None:
                coexist_payload = copy.deepcopy(payload)
                coexist_payload["messages"][0]["content"] += COEXISTENCE_REVIEW
                write_json(coexist_work / "request.json", coexist_payload)
                write_json(coexist_work / "input-receipt.json", {
                    "original_input_receipt_sha256": sha256(work / "input-receipt.json"),
                    "request_sha256": sha256(coexist_work / "request.json"),
                    "change": "nonexclusive step/transient prompt appended; same input images"})
                needed.extend([{"station": station, "method": method},
                               {"station": station, "method": method + "_coexist"}])
    pilot = sorted({job["station"] for job in needed})[:6]
    registration = {"experiment": "EXP07", "model": settings[2], "dataset": str(dataset),
                    "station_ids": [record["station_id"] for record in records],
                    "pending": needed, "pilot_ids": pilot, "source_hashes": source_hashes(),
                    "script_sha256": sha256(Path(__file__)),
                    "protocol_sha256": sha256(Path("docs/experiment-exp07.md")),
                    "numeric_summary_sha256": sha256(numeric / "summary.json"),
                    "request_sha256": {job["method"] + "/" + job["station"]:
                        sha256(out / "predictions" / job["method"] / job["station"] / "request.json")
                        for job in needed}}
    write_json(out / "registration.json", registration)
    print({"pilot_ids": pilot, "request_upper_bound": len(needed),
           "pilot_requests": sum(job["station"] in pilot for job in needed)})


def visual_stage(out: Path, env: Path, phase: str):
    registration = read_json(out / "registration.json")
    if registration["source_hashes"] != source_hashes() or registration["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("registered source changed")
    if registration["protocol_sha256"] != sha256(Path("docs/experiment-exp07.md")):
        raise ValueError("registered protocol changed")
    sentinel = out / (phase + "-started.json")
    if sentinel.exists():
        raise ValueError("phase already started; no implicit retry")
    settings = credentials(env)
    if settings[2] != registration["model"]:
        raise ValueError("registered model changed")
    dataset = Path(registration["dataset"])
    records = {record["station_id"]: record for record in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in registration["pending"]
            if (job["station"] in registration["pilot_ids"]) == (phase == "pilot")]
    write_json(sentinel, {"requests": len(jobs), "phase": phase})
    for job in jobs:
        work = out / "predictions" / job["method"] / job["station"]
        expected = registration["request_sha256"][job["method"] + "/" + job["station"]]
        if sha256(work / "request.json") != expected:
            raise ValueError("prepared request changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        result = call_visual(observation, job["method"], work, settings)
        print(job["station"], job["method"], result["status"], flush=True)
    write_json(out / (phase + "-completed.json"), {"requests": len(jobs)})


def summarize_stage(run: Path, numeric: Path, visual: Path, out: Path):
    dataset, records = records_for(run)
    out.mkdir(parents=True, exist_ok=False)
    methods = {"pair_focus_repair": ("stable", False), "isc_repair": ("isc", False),
               "coexist_repair": ("stable_coexist", True),
               "isc_coexist_repair": ("isc_coexist", True)}
    predictions = {method: {} for method in (*methods, "segment_numeric", "isc_numeric")}
    costs = {method: {"requests": 0, "failures": 0} for method in methods}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for method, (parent, coexist) in methods.items():
            work = visual / "predictions" / parent / station
            result = read_json(work / "result.json")
            candidates = read_json(work / "candidates.json")
            repair = coexist_repair if coexist else repair_return_pairs
            derived = repair(observation, result, candidates)
            predictions[method][station] = derived
            write_json(out / "predictions" / method / station / "result.json", derived)
            costs[method]["requests"] += result.get("request_count", 0)
            costs[method]["failures"] += result["status"] != "success"
        for method, source in (("segment_numeric", run), ("isc_numeric", numeric)):
            predictions[method][station] = read_json(source / "predictions" / method / station / "result.json")
    summary, rows = score(dataset, records, predictions)
    baseline = summary["pair_focus_repair"]
    decisions = {}
    for method in methods:
        if method == "pair_focus_repair":
            continue
        difference = summary[method]["types_daily_macro_f1"] - baseline["types_daily_macro_f1"]
        worst = min(summary[method]["types"][kind]["daily_micro"]["f1"]
                    - baseline["types"][kind]["daily_micro"]["f1"] for kind in baseline["types"])
        decisions[method] = {"difference": difference, "worst_type_difference": worst,
                             "eligible_for_new_test": difference >= .02 and worst >= -.05,
                             "paired_bootstrap": pooled_bootstrap(rows[method], rows["pair_focus_repair"], 2026101407)}
    write_json(out / "summary.json", {"methods": summary, "decisions": decisions,
               "costs": costs, "scope": "24 new development validation stations; not formal test",
               "source_run": str(run), "numeric_results": str(numeric), "visual_run": str(visual),
               "source_hashes": source_hashes(), "script_sha256": sha256(Path(__file__))})
    write_json(out / "per-station.json", rows)
    print({method: values["types_daily_macro_f1"] for method, values in summary.items()})
    print(decisions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("numeric", "prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp07-development-numeric-v1"))
    parser.add_argument("--numeric", type=Path, default=Path("results/exp07-numeric-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp07-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp07-development-v1"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.stage == "numeric":
        numeric_stage(args.run, args.numeric)
    elif args.stage == "prepare":
        prepare_stage(args.run, args.numeric, args.visual, args.env_file)
    elif args.stage in ("pilot", "remaining"):
        visual_stage(args.visual, args.env_file, args.stage)
    else:
        summarize_stage(args.run, args.numeric, args.visual, args.out)


if __name__ == "__main__":
    main()
