"""Copy exact frozen EXP10 inputs, changing only the requested visual model."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path

from experiment_exp10 import infer, summarize

from gnss_research.files import read_json, sha256, write_json


def prepare(visual: Path):
    parent = Path("runs/exp10-development-visual-v1")
    registration = copy.deepcopy(read_json(parent / "registration.json"))
    visual.mkdir(parents=True, exist_ok=False)
    for path in parent.rglob("*"):
        if not path.is_file() or path.name in ("registration.json", "response.json", "result.json"):
            continue
        if path.name.endswith(("-started.json", "-completed.json")):
            continue
        target = visual / path.relative_to(parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "request.json":
            payload = read_json(path)
            payload["model"] = "qwen3-vl-plus"
            write_json(target, payload)
        else:
            shutil.copyfile(path, target)
    pending = {(job["mode"], job["station"]) for job in registration["pending"]}
    for path in (parent / "predictions").glob("*/*/result.json"):
        if (path.parent.parent.name, path.parent.name) not in pending:
            shutil.copyfile(path, visual / path.relative_to(parent))
    for job in registration["pending"]:
        work = visual / "predictions" / job["mode"] / job["station"]
        job["request_sha256"] = sha256(work / "request.json")
        receipt = work / "input-receipt.json"
        receipt.rename(work / "parent-input-receipt.json")
        write_json(receipt, {"request_sha256": job["request_sha256"],
            "parent_request_sha256": sha256(parent / work.relative_to(visual) / "request.json"),
            "change": "model only; frozen images, prompts, candidates unchanged"})
    protocol = Path("docs/experiment-exp13.md")
    script = Path(__file__)
    registration.update(model="qwen3-vl-plus", protocol_path=str(protocol), protocol_sha256=sha256(protocol),
                        parent_registration_sha256=sha256(parent / "registration.json"))
    registration["identity"] = {**registration["identity"], "experiment_id": "GNSS-VLM-EXP13",
                                "model": "qwen3-vl-plus", "protocol_path": str(protocol)}
    registration["experiment_files"][str(script)] = sha256(script)
    shutil.copyfile(script, visual / script.name)
    write_json(visual / "registration.json", registration)
    print({"requests": len(pending), "pilot_ids": registration["pilot_ids"]}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    args = parser.parse_args()
    os.environ["GNSS_VLM_MODEL"] = "qwen3-vl-plus"
    run = Path("runs/exp10-development-baseline-v1")
    visual = Path("runs/exp13-development-qwen-vl-v1")
    output = Path("results/exp13-development-v1")
    if args.stage == "prepare":
        prepare(visual)
    elif args.stage == "summarize":
        summarize(run, visual, output)
        summary = read_json(output / "summary.json")
        summary.update(experiment_id="GNSS-VLM-EXP13", comparison_status="development model comparison")
        (output / "summary.json").rename(output / "parent-summary.json")
        write_json(output / "summary.json", summary)
        report = output / "results.md"
        report.write_text(report.read_text(encoding="utf8").replace(
            "# EXP10", "# EXP13（固定任务qwen3-vl-plus对照）"), encoding="utf8")
        shutil.copyfile(__file__, output / Path(__file__).name)
    else:
        infer(visual, Path(".env"), args.stage)


if __name__ == "__main__":
    main()
