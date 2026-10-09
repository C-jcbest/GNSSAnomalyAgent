"""Summarize actual example executions and audit every TAMA final interval without clipping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from reproduction_contracts import parse_tama_intervals

ROOT = Path(__file__).resolve().parents[1]


def audit_tama(directory: Path) -> dict:
    logs = yaml.safe_load((directory / "logs/GNSS_local_log.yaml").read_text(encoding="utf-8"))
    rows = []
    for case_id, windows in logs.items():
        item = windows[0][0]
        final = item["double_check"].get("corrected_abnormal_index", item["abnormal_index"])
        row = {"case_id": case_id, "initial": item["abnormal_index"], "final": final}
        try:
            row["intervals"] = parse_tama_intervals(final, 1095)
            row["contract_status"] = "passed"
        except ValueError as error:
            row["contract_status"] = "failed"
            row["error"] = str(error)
        rows.append(row)
    requests = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((directory / "requests").glob("*.response.json"))]
    return {"execution": json.loads((directory / "execution.json").read_text(encoding="utf-8")),
            "request_count": len(requests),
            "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in requests),
            "successful_requests": sum(row.get("status") == "success" for row in requests),
            "valid_final_records": sum(row["contract_status"] == "passed" for row in rows),
            "records": rows}


def draw_predictions(directory: Path, audit: dict, output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for axis, row in zip(axes, audit["records"]):
        values = np.load(directory / "data/GNSS" / row["case_id"] / "test/data.npy")[0, :, 0]
        axis.plot(values, color="#286285", linewidth=0.7)
        for interval in row.get("intervals", []):
            if interval["start"] == interval["end"]:
                axis.axvline(interval["start"], color="#bb7135", alpha=0.7)
            else:
                axis.axvspan(interval["start"], interval["end"], color="#e7af60", alpha=0.3)
        finite = values[np.isfinite(values)]
        padding = max(float(np.ptp(finite)) * 0.15, 5)
        axis.set_ylim(float(finite.min()) - padding, float(finite.max()) + padding)
        axis.set_ylabel("N / mm")
        axis.set_title(row["case_id"], loc="left", fontsize=10)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Day index (0–1094)")
    axes[-1].set_xlim(0, 1094)
    figure.suptitle("Adapted TAMA: generic anomaly predictions on development records\n"
                   "Orange = predicted anomaly, not landslide stage ground truth", fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/reproduction-2026-10-07")
    args = parser.parse_args()
    work = args.work.resolve()
    summary = {method: json.loads((work / "runs" / method / "result.json").read_text(encoding="utf-8"))
               for method in ("pelt", "xgboost", "tcn")}
    summary["tama_v1"] = audit_tama(work / "runs/tama-pilot")
    summary["tama_v2"] = audit_tama(work / "runs/tama-pilot-v2")
    summary["scope"] = "Implementation reproduction and development checks only; no held-out GNSS benchmark"
    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_predictions(work / "runs/tama-pilot-v2", summary["tama_v2"], work / "tama-development.png")
    print(json.dumps({"v1_valid": summary["tama_v1"]["valid_final_records"],
                      "v2_valid": summary["tama_v2"]["valid_final_records"],
                      "total_model_requests": summary["tama_v1"]["request_count"] + summary["tama_v2"]["request_count"],
                      "total_tokens": summary["tama_v1"]["total_tokens"] + summary["tama_v2"]["total_tokens"]}))


if __name__ == "__main__":
    main()
