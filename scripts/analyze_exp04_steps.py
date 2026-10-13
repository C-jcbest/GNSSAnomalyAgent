"""Describe frozen EXP04 step errors without fitting or changing predictions."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["exp04-main-v1", "exp04-seed2026101303-v1"]


def analyze_step_candidates():
    events = []
    for run_name in RUNS:
        run = ROOT / "runs" / run_name
        registration = json.loads((run / "registration.json").read_text(encoding="utf-8"))
        dataset = Path(registration["dataset"])
        for station in registration["station_ids"]:
            truth_path = dataset / "truth" / f"{station}.json"
            observation_path = dataset / "observations" / f"{station}.json"
            candidate_path = run / "predictions/guided_fusion" / station / "candidates.json"
            visual_path = run / "predictions/guided_fusion" / station / "result.json"
            final_path = run / "predictions/morphology_fusion" / station / "result.json"
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            visual = json.loads(visual_path.read_text(encoding="utf-8"))
            final = json.loads(final_path.read_text(encoding="utf-8"))
            windows = [
                window for window in candidate["windows"]
                if window["inspection_kind"] == "step"
            ]
            centers = [window["peak_index"] for window in windows]
            visual_days = [
                event["start"]
                for event in visual.get("prediction", {}).get("events", [])
                if event["kind"] == "step" and event["status"] == "confirmed"
            ]
            final_days = [
                event["start"] for event in final.get("prediction", {}).get("events", [])
                if event["kind"] == "step"
            ]
            for event in truth["events"]:
                day = event["start"]
                if event["kind"] != "step" or day <= observation["background_end"]:
                    continue
                values = observation["displacement_mm"][day]
                if values is None or any(value is None for value in values):
                    continue
                paths = [truth_path, observation_path, candidate_path, visual_path, final_path]
                events.append({
                    "run": run_name,
                    "station": station,
                    "truth_day": day,
                    "candidate_centers": centers,
                    "candidate_exact": day in centers,
                    "candidate_within_3": any(abs(center - day) <= 3 for center in centers),
                    "covered_by_step_window": any(
                        window["start"] <= day <= window["end"] for window in windows
                    ),
                    "visual_step_days": visual_days,
                    "visual_exact": day in visual_days,
                    "final_exact": day in final_days,
                    "source_hashes": {
                        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths
                    },
                })
    keys = ["candidate_exact", "candidate_within_3", "covered_by_step_window",
            "visual_exact", "final_exact"]
    counts = {key: sum(event[key] for event in events) for key in keys}
    counts["true_observable_steps"] = len(events)
    counts["missed_with_exact_candidate"] = sum(
        event["candidate_exact"] and not event["final_exact"] for event in events
    )
    counts["missed_without_exact_candidate"] = sum(
        not event["candidate_exact"] and not event["final_exact"] for event in events
    )
    return {
        "scope": "Post-test descriptive error analysis; no new inference, "
                 "no parameter fitting, no new experiment",
        "counts": counts,
        "events": events,
    }


if __name__ == "__main__":
    report = analyze_step_candidates()
    destination = ROOT / "results/exp04-followup-diagnosis"
    destination.mkdir(exist_ok=True)
    (destination / "step-candidate-diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], indent=2))
