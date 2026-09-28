"""Audit an existing P7b run and export its paper table/figure without model calls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_sim.candidate_review import (
    TASKS,
    batches,
    candidates_from,
    load_config,
    load_rows,
    parse_keep,
    reviewed_result,
    source_hashes,
)
from gnss_sim.evaluation_views import evaluate_views
from gnss_sim.input_only import sha, write_json
from gnss_sim.schemas import CaseTruth

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "runs/p7b"
    prepared, run = base / "prepared", base / "candidate-review-v1"
    summary = json.loads((base / "evaluation/summary.json").read_bytes())
    registered = json.loads((ROOT / "configs/p7b-registered.json").read_bytes())
    config_path = ROOT / "configs/p7b-candidate-review.json"
    config = load_config(config_path)
    assert registered["config_sha256"] == sha(config_path)
    assert registered["source_sha256"] == source_hashes() == summary["run"]["source_sha256"]
    orchestration = (ROOT / "src/gnss_sim/p7b.py").read_text(encoding="utf-8")
    assert registered["orchestration_sha256"] == hashlib.sha256(orchestration.encode()).hexdigest()
    assert registered["selection_sha256"] == sha(prepared / "selection.json")
    assert registered["input_manifest_sha256"] == sha(prepared / "inputs/manifest.json")
    assert registered["upstream_sha256"] == sha(prepared / "inputs/upstream.json")
    assert registered["preflight_report_sha256"] == sha(base / "preflight/report.json")
    selection = json.loads((prepared / "selection.json").read_bytes())
    truths = {}
    for entry in selection["cases"]:
        path = ROOT / "data/pilots/pilot-v1/cases" / entry["case_id"] / "truth.json"
        assert sha(path) == entry["truth_sha256"]
        truths[entry["case_id"]] = CaseTruth.model_validate_json(path.read_bytes())
    expected_records, reconstructed = set(), 0
    for task in TASKS:
        rows = {c: load_rows(run / c / task / "predictions.jsonl", task) for c in ("N", "V", "U", "C")}
        for condition, results in rows.items():
            assert set(results) == set(truths)
            assert evaluate_views(truths, results, task) == summary["metrics"][task][condition]
        for cid in truths:
            candidates = candidates_from(rows["N"][cid], rows["V"][cid], task)
            if candidates is None:
                assert rows["U"][cid].status == rows["C"][cid].status == "failed"
                continue
            assert candidates == json.loads((run / "candidates" / task / f"{cid}.json").read_bytes())
            records = []
            for index, group in enumerate(batches(candidates, config)):
                path = run / "raw" / task / cid / f"{index:02d}.json"
                expected_records.add(path)
                record = json.loads(path.read_bytes())
                assert record["case_id"] == cid and record["task"] == task
                assert record["attempts"] == 1
                assert record["image_sha256"] == sha(prepared / "inputs/images" / f"{cid}.png")
                if record["status"] == "success":
                    assert parse_keep(record["response"], group) == record["keep_ids"]
                records.append(record)
            assert reviewed_result(cid, task, candidates, records) == rows["C"][cid]
            reconstructed += 1
    assert expected_records == set((run / "raw").rglob("*.json"))
    assert len(expected_records) == summary["run"]["review_cost"]["requests"]
    out = base / "paper"
    out.mkdir(exist_ok=False)
    with (out / "metrics.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "condition", "primary_f1", "secondary_f1_or_iou", "far", "success"])
        for task in TASKS:
            for condition in ("N", "V", "U", "C"):
                row = summary["metrics"][task][condition]
                writer.writerow([task, condition,
                    row["point_exact" if task == "point" else "range_affiliation"]["f1"],
                    row["point_tolerance_3d"]["f1"] if task == "point" else
                    row["range_daily_positive_axes"]["mean_iou"],
                    row["negative_axes"]["far"], row["execution_success_rate"]])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, task, label in zip(axes, TASKS, ("Point: exact-day F1", "Range: Affiliation F1")):
        values = [summary["metrics"][task][c]["point_exact" if task == "point" else "range_affiliation"]["f1"]
                  for c in ("N", "V", "U", "C")]
        bars = ax.bar(["N", "V", "U", "C"], values,
                      color=["#28564b", "#6d8eae", "#b9aa87", "#bc7459"], width=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
        ax.set(title=label, ylim=(0, 1.08), xlabel="N: numerical   V: visual   U: union   C: review")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("P7b — 60 known development cases (not independent confirmation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "primary-f1.png", dpi=200)
    fig.savefig(out / "primary-f1.svg")
    plt.close(fig)
    audit = {"cases": len(truths), "conditions": 4, "tasks": 2,
             "review_responses_verified": len(expected_records),
             "reconstructed_non_dependency_failed_outputs": reconstructed,
             "reports_recomputed": 8, "frozen_source_config_input_verified": True}
    write_json(out / "audit.json", audit)
    artifact_roots = [base / "prepared", base / "preflight", run, base / "evaluation", out]
    artifacts = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                 for directory in artifact_roots for path in sorted(directory.rglob("*")) if path.is_file()}
    write_json(ROOT / "configs/p7b-frozen.json", {
        "protocol": "candidate-review-v1", "scope": "known-development-60",
        "registered_sha256": sha(ROOT / "configs/p7b-registered.json"),
        "config_sha256": sha(config_path), "source_sha256": source_hashes(),
        "artifacts_sha256": artifacts, "metrics": summary["metrics"],
        "review_effect": summary["review_effect"], "review_cost": summary["run"]["review_cost"],
        "preflight_cost": summary["run"]["preflight_cost"],
        "gate": {"point_primary_delta": summary["metrics"]["point"]["C"]["point_exact"]["f1"] -
                    summary["metrics"]["point"]["U"]["point_exact"]["f1"],
                 "range_primary_delta": summary["metrics"]["range"]["C"]["range_affiliation"]["f1"] -
                    summary["metrics"]["range"]["U"]["range_affiliation"]["f1"],
                 "proceed_to_confirmation": False}, "audit": audit})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
