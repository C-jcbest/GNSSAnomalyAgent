"""Command-line entry points for the current experiment workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gnss_sim.schemas import CASE_TYPES, GenerationRequest


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnss-sim", description="GNSS 合成异常实验台")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="生成合成数据")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--count", type=int, default=24)
    generate.add_argument("--case-type", choices=("all", *CASE_TYPES), default="all")
    generate.add_argument("--data-dir", type=Path)
    serve = commands.add_parser("serve", help="启动本地网页")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18765)
    serve.add_argument("--data-dir", type=Path)
    calibration = commands.add_parser("calibrate", help="在独立 Normal 数据集上校准数值参数")
    calibration.add_argument("--dataset", type=Path, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="登记检测输入、参数与源码；不调用模型")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--parameters", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name, description in (("numerical", "运行数值检测"), ("visual", "调用视觉模型"),
                              ("evaluate", "离线评分并导出 HTML 报告")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--run", type=Path, required=True)
        if name == "visual":
            command.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "serve":
            import uvicorn

            from gnss_sim.api import create_app

            uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
            return
        if args.command == "generate":
            from gnss_sim.storage import DatasetStore

            root = args.data_dir or Path(os.environ.get("GNSS_SIM_DATA_DIR", "data/generated"))
            store = DatasetStore(root)
            try:
                manifest = store.generate_sync(GenerationRequest(
                    seed=args.seed, count=args.count, case_type=args.case_type))
            finally:
                store.executor.shutdown(wait=True)
            if manifest.status != "complete":
                raise ValueError(manifest.error)
            report = {"dataset": str((root / manifest.dataset_id).resolve()),
                      "cases": manifest.generated_cases}
        elif args.command == "evaluate":
            from gnss_sim.report import run

            report = run(args.run)
        else:
            from gnss_sim import detection

            if args.command == "calibrate":
                report = detection.calibrate(args.dataset, args.out)
            elif args.command == "prepare":
                report = detection.prepare(args.dataset, args.parameters, args.out)
            elif args.command == "numerical":
                report = detection.run_numerical(args.run)
            else:
                report = detection.run_visual(args.run, args.env_file)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
