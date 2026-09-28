"""Reproducible paper tables from immutable predictions; offline, never calls models."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gnss_sim.evaluation import AXES, load_results_jsonl
from gnss_sim.evaluation_views import PROTOCOL, evaluate_views
from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseTruth

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            output.update(flatten(item, name))
        else:
            output[name] = item
    return output


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def plot_comparison(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none"}):
        fig, panels = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        point_names = (("sr", "SR"), ("pelt", "PELT"), ("visual-v1", "Visual v1"),
                       ("visual-semantics-v2", "Visual v2 (semantics)"))
        for name, label in point_names:
            m = summary["methods"][name + "/point"]["metrics"]
            scores = [m["point_exact"]["f1"], m["sensitivity"]["1"]["f1"],
                      m["point_tolerance_3d"]["f1"], m["sensitivity"]["7"]["f1"]]
            panels[0].plot([0, 1, 3, 7], scores, "o-", label=label)
        panels[0].set(title="Point: localization tolerance", xlabel="Tolerance (days)",
                      ylabel="Micro F1", xticks=[0, 1, 3, 7], ylim=(0, 1))
        panels[0].legend(fontsize=8)
        names = ("matrix-profile", "theilsen", "visual-v1", "visual-semantics-v2")
        x = np.arange(len(names))
        for offset, metric, label, color in (
                (-0.19, "aff", "Affiliation F1", "#436b91"),
                (0.19, "iou", "Daily IoU (positive axes)", "#3b9989")):
            scores = []
            for name in names:
                m = summary["methods"][name + "/range"]["metrics"]
                scores.append(m["range_affiliation"]["f1"] if metric == "aff" else
                              m["range_daily_positive_axes"]["mean_iou"])
            bars = panels[1].bar(x + offset, scores, width=0.38, label=label, color=color)
            panels[1].bar_label(bars, fmt="%.3f", fontsize=8)
        panels[1].set(title="Range: complementary metric views", ylabel="Score",
                      xticks=x, xticklabels=["MP", "Theil-Sen", "Visual v1", "Visual v2"],
                      ylim=(0, 1.14))
        panels[1].legend(fontsize=8, loc="upper right")
        for panel in panels:
            panel.spines[["top", "right"]].set_visible(False)
            panel.grid(axis="y", alpha=0.18)
            panel.set_axisbelow(True)
        fig.suptitle("pilot-v1 development comparison | same predictions, multiple metrics",
                     fontsize=11)
        fig.savefig(out / "comparison.png", dpi=180)
        fig.savefig(out / "comparison.svg")
        plt.close(fig)


def main():
    config_path = ROOT / "configs/comparison-views-v1.json"
    config = json.loads(config_path.read_bytes())
    if config["protocol"] != PROTOCOL:
        raise ValueError("Unsupported comparison protocol")
    pilot = ROOT / "data/pilots" / config["pilot_id"]
    manifest = verify_pilot(pilot)
    entries = {row["case_id"]: row for row in manifest["cases"]}
    truths = {key: CaseTruth.model_validate_json(
        (pilot / "cases" / key / "truth.json").read_bytes()) for key in entries}
    groups = {"normal": [], "single": [], "multi": []}
    for key, entry in entries.items():
        count = entry["event_count"]
        group = "normal" if count == 0 else "single" if count == 1 else "multi"
        groups[group].append(key)
        if count:
            groups.setdefault("case_type:" + entry["case_type"], []).append(key)
    p6 = json.loads((ROOT / "configs/p6-frozen.json").read_bytes())
    summary = {"protocol": PROTOCOL, "scope": "development_pilot_only",
               "registry_sha256": sha(config_path),
               "pilot_sha256": {name: sha(pilot / f"{name}.json")
                                for name in ("manifest", "summary")},
               "source_sha256": {name: sha(ROOT / name) for name in (
                   "src/gnss_sim/evaluation.py", "src/gnss_sim/evaluation_views.py",
                   "scripts/export_comparison.py", "uv.lock")},
               "methods": {}, "paired_common_success": {}}
    per_case, grouped, overall, stored = [], [], [], {}
    for spec in config["methods"]:
        method, task = spec["method_id"], spec["task"]
        key = f"{method}/{task}"
        path = ROOT / spec["predictions"]
        rows, errors = load_results_jsonl(path, task)
        results = {row.case_id: row for row in rows}
        if (errors or len(rows) != len(truths) or set(results) != set(truths) or
                any(row.method != method for row in rows)):
            raise ValueError(f"Invalid method artifact {key}")
        if method == "visual-v1" and sha(path) != p6["predictions_sha256"][task]:
            raise ValueError("P6 prediction hash changed")
        if method == "visual-semantics-v2":
            run = json.loads((path.parent.parent / "run.json").read_bytes())
            if sha(path) != run["tasks"][task]["predictions_sha256"]:
                raise ValueError("Semantic prediction hash changed")
        metrics = evaluate_views(truths, results, task)
        official = json.loads(path.with_name("report.json").read_bytes())
        primary = metrics["point_exact"] if task == "point" else metrics["range_affiliation"]
        for name in ("precision", "recall", "f1"):
            official_key = name if task == "point" else "affiliation_" + name
            if abs(primary[name] - official[official_key]) > 1e-12:
                raise ValueError(f"Cannot reproduce P4 primary score {key}/{name}")
        if (metrics["negative_axes"]["far"] != official["far"] or
                metrics["execution_success_rate"] != official["execution_success_rate"]):
            raise ValueError("Cannot reproduce P4 FAR or success rate")
        metadata = {"method": method, "task": task}
        provenance = {"predictions": spec["predictions"], "predictions_sha256": sha(path),
                      "config": spec["config"], "config_sha256": sha(ROOT / spec["config"])}
        raw_path = path.with_name("raw.jsonl")
        usage = None
        if raw_path.exists():
            raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            if len(raw) != len(truths) or {row["case_id"] for row in raw} != set(truths):
                raise ValueError("Raw request count mismatch")
            usage = {"logical_calls": len(raw), "cost": None,
                     "response_models": sorted({row.get("response_model") for row in raw
                                                if row.get("response_model")}),
                     "usage_recorded_calls": sum(row.get("usage") is not None for row in raw),
                     **{name: sum((row.get("usage") or {}).get(name, 0) for row in raw)
                        for name in ("prompt_tokens", "completion_tokens", "total_tokens")}}
            provenance["raw_sha256"] = sha(raw_path)
        summary["methods"][key] = {"provenance": provenance, "usage": usage, "metrics": metrics}
        overall.append({**metadata, **flatten(metrics)})
        stored[key] = results
        for group, ids in groups.items():
            report = evaluate_views({i: truths[i] for i in ids}, {i: results[i] for i in ids}, task)
            grouped.append({**metadata, "group": group, **flatten(report)})
        for axis in AXES:
            report = evaluate_views(truths, results, task, axes=(axis,))
            grouped.append({**metadata, "group": "axis:" + axis, **flatten(report)})
        for case_id in truths:
            report = evaluate_views({case_id: truths[case_id]}, {case_id: results[case_id]}, task)
            per_case.append({**metadata, "case_id": case_id,
                             "case_type": entries[case_id]["case_type"], **flatten(report)})
    for task in ("point", "range"):
        left, right = stored[f"visual-v1/{task}"], stored[f"visual-semantics-v2/{task}"]
        ids = [key for key in truths if left[key].status == right[key].status == "success"]
        summary["paired_common_success"][task] = {
            "case_count": len(ids), "excluded_case_count": len(truths) - len(ids),
            "case_ids": ids,
            "original": evaluate_views({i: truths[i] for i in ids}, {i: left[i] for i in ids}, task),
            "semantics": evaluate_views({i: truths[i] for i in ids}, {i: right[i] for i in ids}, task)}
    out = ROOT / "runs" / PROTOCOL
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "registry.json"
    if registry.exists() and json.loads(registry.read_bytes()) != config:
        raise ValueError("Existing comparison registry differs; use a new protocol version")
    write_json(registry, config)
    write_json(out / "summary.json", summary)
    write_json(out / "grouped.json", grouped)
    write_json(out / "per-case.json", per_case)
    write_csv(out / "overall.csv", overall)
    write_csv(out / "grouped.csv", grouped)
    write_csv(out / "per-case.csv", per_case)
    lines = ["# 多版本开发集对照", "", "同一 pilot-v1；一次视觉运行；不是独立测试。",
             "", "## Point", "", "| 方法 | 精确 P | 精确 R | 精确 F1 | ±3日 P | ±3日 R | ±3日 F1 | FAR | 成功率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, record in summary["methods"].items():
        m = record["metrics"]
        if m["task"] == "point":
            vals = [m[view][metric] for view in ("point_exact", "point_tolerance_3d")
                    for metric in ("precision", "recall", "f1")]
            vals += [m["negative_axes"]["far"], m["execution_success_rate"]]
            lines.append("| " + key + " | " + " | ".join(map(fmt, vals)) + " |")
    lines += ["", "## Range", "",
              "| 方法 | Aff P | Aff R | Aff F1 | 正轴平均 IoU | 全轴逐日微 F1 | FAR | 成功率 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, record in summary["methods"].items():
        m = record["metrics"]
        if m["task"] == "range":
            vals = [m["range_affiliation"][metric] for metric in ("precision", "recall", "f1")]
            vals += [m["range_daily_positive_axes"]["mean_iou"],
                     m["range_daily_all_axes_micro"]["f1"], m["negative_axes"]["far"],
                     m["execution_success_rate"]]
            lines.append("| " + key + " | " + " | ".join(map(fmt, vals)) + " |")
    (out / "paper-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_comparison(summary, out)
    print("\n".join(lines))
    print(f"\nExports: {out}")


if __name__ == "__main__":
    main()
