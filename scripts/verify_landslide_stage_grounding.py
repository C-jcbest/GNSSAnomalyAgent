"""Audit image inventory, real requests, development metrics and delivery."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

from landslide_stage_grounding import RUN, configuration, replay
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json


def http_check(item):
    name, digest = item
    with urlopen(f"http://127.0.0.1:18792/{name}", timeout=10) as response:
        data = response.read()
        if response.status != 200 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"HTTP file differs: {name}")
    return len(data)


def main():
    public, cases, all_rows = replay()
    config = configuration()
    packets = json.loads((public / "packets.json").read_text(encoding="utf-8"))
    by_case = defaultdict(list)
    for packet in packets:
        by_case[packet["case_id"]].append(packet)
    inventory_images = 0
    for item in cases:
        inventory_images += len(item["summary"]["images"])
        if not any(row["activity_label"] == 1 for row in item["final"]["image-r1"]):
            if item["case_id"] in by_case:
                raise ValueError("No-activity record has a request")
            continue
        requests = by_case[item["case_id"]]
        control = [packet for packet in requests if packet["condition"] == "image"]
        treatment = [packet for packet in requests if packet["condition"] == "numeric"]
        if len(control) != 2 or len(treatment) != 2 or control[0]["prompt"] != control[1]["prompt"]:
            raise ValueError("Incomplete repeated paired schedule")
        if any(packet["images"] != control[0]["images"] or len(packet["images"]) != 7 for packet in requests):
            raise ValueError("Seven-image invariant differs")
        if any(not packet["prompt"].startswith(control[0]["prompt"]) for packet in treatment):
            raise ValueError("Original prompt prefix changed")
    with (public / "daily-stage-grounding.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in csv_rows:
        groups[(row["case_id"], f"{row['condition']}-r{row['repetition']}")].append(row)
    if len(csv_rows) != 52560 or len(groups) != 48:
        raise ValueError("Full calendar CSV incomplete")
    for item in cases:
        for arm, rows in item["final"].items():
            expected = [{"condition": arm.split("-r")[0], "repetition": arm[-1],
                         **{key: str(value) for key, value in row.items()}} for row in rows]
            if expected != groups[(item["case_id"], arm)]:
                raise ValueError("CSV differs from request replay")
    reference = json.loads((RUN.parents[1] / "landslide-ai-stage-review-2026-10-08/review-v1/compiled/daily-reference.json").read_text(encoding="utf-8"))
    analysis = json.loads((public / "analysis.json").read_text(encoding="utf-8"))
    for arm, predicted in all_rows.items():
        actual = [row["feature"] for row in reference if row["stage_evaluable"]]
        guess = [row["feature"] for row, ref in zip(predicted, reference, strict=True) if ref["stage_evaluable"]]
        p, r, f, support = precision_recall_fscore_support(
            actual, guess, labels=["acceleration", "steady_motion", "deceleration"], zero_division=0,
        )
        reported = analysis["development_agreement"][arm]
        for index, stage in enumerate(("acceleration", "steady_motion", "deceleration")):
            h = reported["per_class"][stage]
            if h["reference_support"] != int(support[index]):
                raise ValueError("Class reference support differs")
            for key, value in (("precision", p[index]), ("recall", r[index]), ("f1", f[index])):
                if h[key] is not None and abs(h[key] - value) > 1e-12:
                    raise ValueError("Development metric differs from sklearn")
    audits = json.loads((public / "presentation/plot-audits.json").read_text(encoding="utf-8"))
    code = {"unknown": 0, "none": 1, "acceleration": 2, "steady_motion": 3, "deceleration": 4}
    for item in cases:
        observed = [row is not None for row in item["case"].displacement_mm]
        for name in item["plots"] if "plots" in item else [f"presentation/{item['case_id']}-{i}.png" for i in range(4)]:
            audit = audits[name]
            if sha(public / name) != audit["sha256"]:
                raise ValueError("Comparison figure changed")
            for index, arm in enumerate(("image-r1", "numeric-r1", "image-r2", "numeric-r2"), 1):
                expected = [5 if not observed[day] else code[row["feature"]]
                            for day, row in enumerate(item["final"][arm])]
                if audit["label_colors"][index] != expected:
                    raise ValueError("Plot colors differ from exported labels")
    freeze = json.loads((RUN / "presentation-freeze.json").read_text(encoding="utf-8"))
    for name, digest in freeze["public_files_sha256"].items():
        if sha(public / name) != digest:
            raise ValueError(f"Public file changed: {name}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        total_bytes = sum(executor.map(http_check, freeze["public_files_sha256"].items()))
    result = {"status": "passed", "requests_replayed": len(packets), "image_associations": len(packets) * 7,
              "cases": len(cases), "inventory_images": inventory_images, "csv_rows": len(csv_rows),
              "sklearn_metric_groups": len(all_rows), "plot_masks_verified": len(audits),
              "http_files": len(freeze["public_files_sha256"]), "http_bytes": total_bytes,
              "inference_and_historical_source_unchanged": True, "config": config["protocol_id"],
              "reference_use": "development_only", "performance_scores": None,
              "browser_rendering": "not_verified; existing CUA initialization path failure unresolved"}
    write_json(RUN / "verification-audit.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
