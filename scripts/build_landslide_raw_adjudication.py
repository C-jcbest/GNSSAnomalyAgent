"""Freeze and publish the twelve-case same-session author activity review."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_landslide_reference_interface import verify_previous_run  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_raw_adjudication import (  # noqa: E402
    compile_author_activity,
    raw_block_statistics,
    render_author_activity,
)
from gnss_sim.landslide_reference_interface import partition_sha256  # noqa: E402

SOURCE = ROOT / "artifacts/landslide-background-2026-10-08/batch-v1/public"
HISTORICAL = ROOT / "artifacts/landslide-reference-interface-2026-10-08/run-v2"
PROTOCOL = ROOT / "docs/landslide-raw-adjudication-protocol.md"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources(out):
    ledger = read_json(out / "run-ledger-start.json")
    if sha(PROTOCOL) != ledger["protocol_sha256"]:
        raise ValueError("The author review protocol changed after the review began")
    for name, expected in ledger["source_packet_sha256"].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"Source packet changed: {name}")
    historical = verify_previous_run(HISTORICAL)
    verify_previous_run(ROOT / historical["historical_run"])
    return ledger


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(out):
    if (out / "activity-freeze.json").exists():
        raise ValueError("This author review is frozen; use --verify or a new review directory")
    ledger = verify_sources(out)
    public = out / "public"
    source_manifest = read_json(SOURCE / "manifest.json")
    plot = read_json(SOURCE / "protocol.json")["plot"]
    answers_path = public / "reviews/author-answers.json"
    answers = read_json(answers_path)
    records = source_manifest["cases"]
    expected_ids = [f"case_{number:04d}" for number in range(1, 13)]
    if sorted(answers) != expected_ids or [record["case_id"] for record in records] != expected_ids:
        raise ValueError("Review the complete twelve-case development packet")
    for directory in ("observations", "images", "daily"):
        (public / directory).mkdir(exist_ok=True)
    totals = Counter()
    all_rows = []
    entries = []
    transitions = Counter()
    freezes = {}
    for record in records:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((SOURCE / record["input"]).read_bytes())
        block_path = public / f"blocks/{case_id}.json"
        if raw_block_statistics(case) != read_json(block_path):
            raise ValueError(f"Saved descriptive blocks differ from raw observations: {case_id}")
        review, rows = compile_author_activity(answers[case_id], case, record)
        review["block_statistics_sha256"] = sha(block_path)
        review["protocol_sha256"] = ledger["protocol_sha256"]
        review_path = public / f"reviews/{case_id}.review.json"
        daily_path = public / f"daily/{case_id}.json"
        write_json(review_path, review)
        write_json(daily_path, rows)
        write_csv(public / f"daily/{case_id}.csv", rows)
        freezes[case_id] = {"activity_sha256": partition_sha256(review),
                            "review_file_sha256": sha(review_path), "daily_sha256": sha(daily_path)}
        for name in (record["input"], record["csv"]):
            shutil.copyfile(SOURCE / name, public / name)
        views = []
        expected_colors = [row["activity_label"] + 1 if value is not None else 3
                           for row, value in zip(rows, case.displacement_mm, strict=True)]
        for number, view in enumerate(record["views"]):
            shutil.copyfile(SOURCE / view["image"], public / view["image"])
            image_name = f"images/{case_id}-author-{number}.png"
            audit = render_author_activity(case, view, rows, public / image_name, plot)
            if audit["label_colors"] != expected_colors:
                raise ValueError("The plot strip differs from the exported daily labels")
            views.append({**view, "review_image": image_name, "review_image_audit": audit})
        previous = read_json(HISTORICAL / f"public/drafts/{case_id}-partition_v2.daily.json")
        observed_transitions = Counter(
            f"{old['activity_label']}->{new['activity_label']}"
            for old, new, value in zip(previous, rows, case.displacement_mm, strict=True)
            if value is not None
        )
        transitions.update(observed_transitions)
        counts = Counter(str(row["activity_label"]) for row in rows)
        totals.update(counts)
        all_rows.extend(rows)
        missing_days = sum(value is None for value in case.displacement_mm)
        entries.append({"case_id": case_id, "input": record["input"],
                        "input_sha256": record["input_sha256"], "views": views,
                        "review": review_path.relative_to(public).as_posix(),
                        "daily": daily_path.relative_to(public).as_posix(),
                        "blocks": block_path.relative_to(public).as_posix(),
                        "activity_sha256": freezes[case_id]["activity_sha256"],
                        "counts": dict(counts), "missing_days": missing_days,
                        "observed_unknown_days": counts["-1"] - missing_days,
                        "nominal_activity_spans": sum(span["state"] == "activity" for span in review["spans"]),
                        "v2_to_author_transitions": dict(observed_transitions),
                        "disagreement_days": sum(count for key, count in observed_transitions.items()
                                                 if key.split("->")[0] != key.split("->")[1])})
    missing_days = sum(entry["missing_days"] for entry in entries)
    summary = {"status": "same-session AI author activity review; independent verification pending",
               "cases": 12, "raw_images_read": 48, "daily_rows": len(all_rows),
               "activity_days": totals["1"], "stationary_days": totals["0"],
               "unknown_days": totals["-1"], "missing_days": missing_days,
               "observed_unknown_days": totals["-1"] - missing_days,
               "nominal_activity_spans": sum(entry["nominal_activity_spans"] for entry in entries),
               "v2_to_author_transitions": dict(transitions),
               "disagreement_days": sum(entry["disagreement_days"] for entry in entries),
               "stage_status": "not_started", "new_external_calls": 0,
               "prior_outputs_seen": True, "independent_expert": False,
               "new_auxiliary_images_read": 0, "sealed_test_read": False,
               "performance_scores": None,
               "meaning": "Development review disagreements, NOT errors or accuracy improvement"}
    write_csv(public / "daily-activity-author.csv", all_rows)
    shutil.copyfile(PROTOCOL, public / "review-guide.md")
    shutil.copyfile(ROOT / "scripts/landslide_raw_adjudication_page.html", public / "index.html")
    write_json(public / "manifest.json", {"summary": summary, "cases": entries})
    write_json(out / "activity-freeze.json", {"reviews": freezes, "author_answers_sha256": sha(answers_path),
                                              "stage_status": "not_started"})
    source_names = ["src/gnss_sim/landslide_raw_adjudication.py",
                    "scripts/build_landslide_raw_adjudication.py",
                    "scripts/landslide_raw_adjudication_page.html",
                    "tests/test_landslide_raw_adjudication.py", "docs/landslide-raw-adjudication-protocol.md"]
    write_json(out / "manifest.json", {
        "source_packet": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": {name: sha(ROOT / name) for name in source_names},
        "evidence_sha256": {path.relative_to(out).as_posix(): sha(path)
                            for path in sorted(public.rglob("*")) if path.is_file()},
        "activity_freeze_sha256": sha(out / "activity-freeze.json"), "summary": summary})
    return verify(out)


def verify(out):
    ledger = verify_sources(out)
    manifest = read_json(out / "manifest.json")
    public = out / "public"
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Author review source changed: {name}")
    for name, expected in manifest["evidence_sha256"].items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen author evidence changed: {name}")
    if sha(out / "activity-freeze.json") != manifest["activity_freeze_sha256"]:
        raise ValueError("Activity freeze changed")
    freeze = read_json(out / "activity-freeze.json")
    answers_path = public / "reviews/author-answers.json"
    if sha(answers_path) != freeze["author_answers_sha256"]:
        raise ValueError("Author source answers changed")
    answers = read_json(answers_path)
    records = read_json(SOURCE / "manifest.json")["cases"]
    entries = read_json(public / "manifest.json")["cases"]
    all_rows = []
    for record, entry in zip(records, entries, strict=True):
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        review, rows = compile_author_activity(answers[case.case_id], case, record)
        block_path = public / entry["blocks"]
        if raw_block_statistics(case) != read_json(block_path):
            raise ValueError("Blocks do not replay")
        review["block_statistics_sha256"] = sha(block_path)
        review["protocol_sha256"] = ledger["protocol_sha256"]
        if review != read_json(public / entry["review"]) or rows != read_json(public / entry["daily"]):
            raise ValueError("Review/daily replay differs")
        if partition_sha256(review) != freeze["reviews"][case.case_id]["activity_sha256"]:
            raise ValueError("Activity content hash differs")
        colors = [row["activity_label"] + 1 if value is not None else 3
                  for row, value in zip(rows, case.displacement_mm, strict=True)]
        for old_view, view in zip(record["views"], entry["views"], strict=True):
            if sha(public / view["image"]) != old_view["image_sha256"]:
                raise ValueError("Raw image changed")
            if colors != view["review_image_audit"]["label_colors"]:
                raise ValueError("Plot mask does not match daily export")
            if view["review_image_audit"]["ylim_NEU"] != [panel["ylim"] for panel in old_view["audit"]["panels"]]:
                raise ValueError("Review chart raw axes changed")
        all_rows.extend(rows)
    with (public / "daily-activity-author.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    for row in csv_rows:
        row["day_index"] = int(row["day_index"])
        row["activity_label"] = int(row["activity_label"])
    if csv_rows != all_rows or len(all_rows) != 13140:
        raise ValueError("Combined CSV differs from the daily source")
    if any(row["feature"] != ("none" if row["activity_label"] == 0 else "unknown") for row in all_rows):
        raise ValueError("Definite stages must not be assigned in this activity-only pass")
    return manifest["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/landslide-raw-adjudication-2026-10-08/review-v3")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    summary = verify(args.output) if args.verify else build(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
