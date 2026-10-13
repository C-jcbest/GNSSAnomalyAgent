"""Previously used test sets are diagnostic evidence, never EXP03 holdouts."""
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("exp02-main-v3", "exp02-seed2026101204-v3", "exp02-seed2026101205-v3")
METHODS = ("rf_split_temporal", "guided_fusion", "anchored_fusion")


def diagnose():
    points = {}
    activity = {method: dict(target_days=0, missed_days=0, early_days=0, plateau_days=0)
                for method in METHODS}
    for name in RUNS:
        run = ROOT / "runs" / name
        config = read_json(run / "registration.json")
        dataset = Path(config["dataset"])
        for record in read_json(dataset / "manifest.json")["records"]:
            if record["split"] != "test":
                continue
            station = record["station_id"]
            observation = read_json(dataset / record["observation"])
            truth = read_json(dataset / record["truth"])
            observed = np.isfinite(observed_values(observation)).all(axis=1)
            windows = read_json(run / "predictions/guided_fusion" / station
                                / "candidates.json")["windows"]
            masks = {}
            for method in METHODS:
                result = read_json(run / "predictions" / method / station / "result.json")
                masks[method] = event_masks(Prediction.model_validate(
                    result.get("prediction") or {"events": []}), observed)
            for event in truth["events"]:
                kind = event["kind"]
                start, end = event["start"], event["end"]
                if kind in ("spike", "step") and observed[start]:
                    key = f"{truth['scenario']}/{kind}"
                    counts = points.setdefault(key, {"observed_events": 0, "window_covered": 0,
                        "peak_exact": 0, **{m: 0 for m in METHODS}, "visual_near_3": 0})
                    counts["observed_events"] += 1
                    selected = [w for w in windows if w["inspection_kind"] == kind]
                    counts["window_covered"] += any(w["start"] <= start <= w["end"]
                                                    for w in selected)
                    counts["peak_exact"] += any(w["peak_index"] == start for w in selected)
                    for method in METHODS:
                        counts[method] += int(masks[method][kind][start])
                    counts["visual_near_3"] += int(masks["guided_fusion"][kind][
                        max(0, start - 3):start + 4].any())
                if kind == "deformation":
                    target = observed[start:end + 1]
                    for method in METHODS:
                        predicted = masks[method][kind]
                        counts = activity[method]
                        counts["target_days"] += int(target.sum())
                        counts["missed_days"] += int((target & ~predicted[start:end + 1]).sum())
                        counts["early_days"] += int(predicted[
                            observation["background_end"] + 1:start].sum())
                        counts["plateau_days"] += int(predicted[end + 1:].sum())
    out = ROOT / "results/exp03-diagnosis"
    out.mkdir(exist_ok=False)
    write_json(out / "exp02-errors.json", {"scope": "used tests, diagnostic only",
                                          "points": points, "activity": activity})
    print({"points": points, "activity": activity})


if __name__ == "__main__":
    diagnose()
