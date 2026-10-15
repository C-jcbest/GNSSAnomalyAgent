"""Model comparison on unchanged EXP11 development observations and requests."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from experiment_exp11 import infer, prepare, summarize

from gnss_research.files import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    args = parser.parse_args()
    os.environ["GNSS_VLM_MODEL"] = "qwen3-vl-plus"
    run = Path("runs/exp11-development-baseline-v1")
    visual = Path("runs/exp12-development-qwen-vl-v1")
    output = Path("results/exp12-development-v1")
    env = Path(".env")
    if args.stage == "prepare":
        prepare(run, visual, env)
        registration = read_json(visual / "registration.json")
        protocol = Path("docs/experiment-exp12.md")
        registration["parent_protocol_sha256"] = registration["protocol_sha256"]
        registration["protocol_path"] = str(protocol)
        registration["protocol_sha256"] = sha256(protocol)
        registration["identity"] = {**registration["identity"],
            "experiment_id": "GNSS-VLM-EXP12", "model": "qwen3-vl-plus",
            "name": "固定视觉任务的模型能力对照", "protocol_path": str(protocol)}
        registration["experiment_files"][str(Path(__file__))] = sha256(Path(__file__))
        shutil.copyfile(__file__, visual / Path(__file__).name)
        (visual / "registration.json").rename(visual / "parent-registration.json")
        write_json(visual / "registration.json", registration)
    elif args.stage == "summarize":
        summarize(run, visual, output)
        summary = read_json(output / "summary.json")
        summary["experiment_id"] = "GNSS-VLM-EXP12"
        summary["decision"]["exp11_original_gate_only"] = True
        summary["decision"]["production_adopted"] = False
        summary["comparison_status"] = "development model comparison, not independent test"
        (output / "summary.json").rename(output / "parent-summary.json")
        write_json(output / "summary.json", summary)
        report = output / "results.md"
        content = report.read_text(encoding="utf8").replace(
            "# EXP11：视觉持续活动判别与固定数值边界融合",
            "# EXP12：固定视觉任务的模型能力对照（qwen3-vl-plus）")
        report.write_text(content + "\n自动通过字段仅保留EXP11原门槛；本模型比较不自动采纳。\n", encoding="utf8")
        shutil.copyfile(__file__, output / Path(__file__).name)
    else:
        infer(visual, env, args.stage)


if __name__ == "__main__":
    main()
