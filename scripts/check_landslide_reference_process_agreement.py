"""Check frozen author-AI references against sealed process labels, without relabeling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "ai-observable-holdout-data-v1"
REVIEW = DATA / "codex-review"
STAGES = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify_day(ai, process, observed):
    """Unknown and missing are distinct; phase comparison never extends activity."""
    if not observed:
        return "missing", "not_evaluable"
    actual = int(ai["activity_label"])
    latent = int(process["activity_label"])
    if actual == -1:
        return f"unknown_process_{latent}", "not_evaluable"
    if actual not in (0, 1) or latent not in (0, 1):
        raise ValueError("Unexpected activity code")
    activity = f"ai_{actual}_process_{latent}"
    if actual != 1 or ai["feature"] not in STAGES:
        return activity, "not_evaluable"
    process_stage = process["feature"] or "none"
    stage = "agree" if ai["feature"] == process_stage else "disagree"
    return activity, stage


def summarize(rows):
    activity = Counter(row["activity_comparison"] for row in rows)
    stage = Counter(row["stage_comparison"] for row in rows)
    definite = sum(activity[f"ai_{a}_process_{b}"] for a in (0, 1) for b in (0, 1))
    disagreement = activity["ai_0_process_1"] + activity["ai_1_process_0"]
    stage_n = stage["agree"] + stage["disagree"]
    jointly_active = [
        row
        for row in rows
        if row["activity_comparison"] == "ai_1_process_1"
        and row["stage_comparison"] != "not_evaluable"
    ]
    return {
        "calendar_days": len(rows),
        "missing": activity["missing"],
        "unknown_process_0": activity["unknown_process_0"],
        "unknown_process_1": activity["unknown_process_1"],
        **{f"ai_{a}_process_{b}": activity[f"ai_{a}_process_{b}"] for a in (0, 1) for b in (0, 1)},
        "definite_days": definite,
        "activity_disagreement_days": disagreement,
        "activity_agreement_fraction": (definite - disagreement) / definite if definite else None,
        "stage_definite_days": stage_n,
        "stage_agree": stage["agree"],
        "stage_disagree": stage["disagree"],
        "stage_agreement_fraction": stage["agree"] / stage_n if stage_n else None,
        "joint_active_stage_days": len(jointly_active),
        "joint_active_stage_agree": sum(
            row["stage_comparison"] == "agree" for row in jointly_active
        ),
        "confirmed_activity_unknown_stage": sum(
            row["observed"] and row["ai_activity"] == 1 and row["ai_feature"] not in STAGES
            for row in rows
        ),
    }


def disagreement_spans(rows, key, categories):
    spans = []
    start = 0
    while start < len(rows):
        category = rows[start][key]
        stop = start + 1
        while stop < len(rows) and rows[stop][key] == category:
            stop += 1
        if category in categories:
            spans.append(
                {
                    "case_id": rows[start]["case_id"],
                    "comparison": key,
                    "category": category,
                    "start": start,
                    "stop": stop,
                    "days": stop - start,
                    "start_date": rows[start]["date"],
                    "last_date": rows[stop - 1]["date"],
                }
            )
        start = stop
    return spans


def render(case, rows, path):
    values = np.array(
        [[np.nan] * 3 if value is None else value for value in case["displacement_mm"]], dtype=float
    )
    figure, axes = plt.subplots(
        7, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1, 1]}
    )
    for index, name in enumerate(("N", "E", "U")):
        axes[index].plot(values[:, index], linewidth=0.7)
        axes[index].set_ylabel(name + " mm")
        axes[index].grid(alpha=0.2)
        for span in disagreement_spans(
            rows, "activity_comparison", {"ai_0_process_1", "ai_1_process_0"}
        ):
            axes[index].axvspan(
                span["start"] - 0.5, span["stop"] - 0.5, color="#d64949", alpha=0.16
            )
    bands = []
    for source in ("ai", "process"):
        activity = [row[source + "_activity"] + 1 if row["observed"] else 3 for row in rows]
        stage = [STAGES.get(row[source + "_feature"], 0) if row["observed"] else 4 for row in rows]
        bands.extend(
            [
                (activity, ["#edc46a", "#e0e7e0", "#407ea0", "#777777"], source + " activity"),
                (stage, ["#edc46a", "#c96055", "#498774", "#6581b7", "#777777"], source + " stage"),
            ]
        )
    for axis, (codes, colors, label) in zip(axes[3:], bands, strict=True):
        axis.imshow(
            [codes],
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=len(colors) - 1,
            cmap=ListedColormap(colors),
            extent=(-0.5, len(rows) - 0.5, 0.5, -0.5),
        )
        axis.set_yticks([])
        axis.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel("Original day index; red shade = definite activity disagreement")
    figure.suptitle(case["case_id"] + " | frozen author-AI vs latent process; post hoc check")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.text(
        0.1,
        0.02,
        "Activity: blue=active, pale=stationary, amber=unknown, gray=missing.\n"
        "Stage: red=A, green=S, blue=D; amber includes none/unknown (CSV distinguishes both).",
        fontsize=9,
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run(out):
    protocol = read_json(out / "protocol.json")
    if (out / "summary.json").exists():
        raise FileExistsError("Keep the completed check; use a separate replay directory")
    for name, digest in protocol["bindings"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    for name, digest in read_json(DATA / "data-freeze.json").items():
        if sha(DATA / name) != digest:
            raise ValueError("Data freeze changed: " + name)
    for name, digest in read_json(REVIEW / "compiled/reference-freeze.json")["files"].items():
        if sha(REVIEW / "compiled" / name) != digest:
            raise ValueError("Author-AI reference changed: " + name)
    source = Path(__file__).resolve()
    write_json(
        out / "analysis-start.json",
        {
            "source_sha256": sha(source),
            "protocol_sha256": sha(out / "protocol.json"),
            "sealed_read_after_protocol": True,
        },
    )
    (out / "source.py").write_bytes(source.read_bytes())
    ai_rows = read_json(REVIEW / "compiled/daily-reference.json")
    with (DATA / "mechanism-sealed/reference-labels.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        process_rows = list(csv.DictReader(stream))
    ai = {(row["case_id"], int(row["day_index"])): row for row in ai_rows}
    process = {(row["case_id"], int(row["day_index"])): row for row in process_rows}
    if len(ai) != len(ai_rows) or len(process) != len(process_rows) or set(ai) != set(process):
        raise ValueError("Duplicate or unequal reference calendars")
    daily, per_case, spans = [], [], []
    (out / "figures").mkdir()
    for case_id in protocol["population"]:
        case = read_json(REVIEW / case_id / "input.json")
        record_rows = []
        for day, date in enumerate(case["dates"]):
            a, p = ai[(case_id, day)], process[(case_id, day)]
            if a["date"] != date or p["date"] != date:
                raise ValueError("Reference/input date mismatch")
            observed = case["displacement_mm"][day] is not None
            activity, stage = classify_day(a, p, observed)
            record_rows.append(
                {
                    "case_id": case_id,
                    "day_index": day,
                    "date": date,
                    "observed": observed,
                    "ai_activity": int(a["activity_label"]),
                    "process_activity": int(p["activity_label"]),
                    "ai_feature": a["feature"],
                    "process_feature": p["feature"] or "none",
                    "activity_comparison": activity,
                    "stage_comparison": stage,
                }
            )
        daily.extend(record_rows)
        per_case.append({"case_id": case_id, **summarize(record_rows)})
        spans.extend(
            disagreement_spans(
                record_rows, "activity_comparison", {"ai_0_process_1", "ai_1_process_0"}
            )
        )
        spans.extend(disagreement_spans(record_rows, "stage_comparison", {"disagree"}))
        render(case, record_rows, out / "figures" / (case_id + ".png"))
    if len(daily) != len(ai):
        raise ValueError("Unexpected case outside the frozen population")
    write_csv(out / "daily-agreement.csv", daily)
    write_csv(out / "per-record.csv", per_case)
    write_json(out / "disagreement-spans.json", spans)
    summary = summarize(daily)
    summary["stage_confusion"] = dict(
        Counter(
            row["ai_feature"] + " -> " + row["process_feature"]
            for row in daily
            if row["stage_comparison"] != "not_evaluable"
        )
    )
    summary["scope"] = protocol["inference_limit"]
    write_json(out / "summary.json", summary)
    write_json(
        out / "manifest.json",
        {
            str(path.relative_to(out)): sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
