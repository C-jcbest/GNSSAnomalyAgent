"""Explicit data, registration, execution and scoring entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_research import data, experiment
from gnss_research.baselines import NUMERICAL_METHODS
from gnss_research.upstream import ROOT, UPSTREAM_METHODS, setup_sources
from gnss_research.vision import VISUAL_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnss-research")
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate", help="生成独立站点训练、验证与测试数据")
    generation.add_argument("--out", type=Path, required=True)
    generation.add_argument("--seed", type=int, required=True)
    generation.add_argument("--per-split", type=int, default=24)
    generation.add_argument("--days", type=int, default=730)
    registration = commands.add_parser("register", help="冻结方案、数据与源码；不调用模型")
    registration.add_argument("--dataset", type=Path, required=True)
    registration.add_argument("--out", type=Path, required=True)
    registration.add_argument("--methods", nargs="+", choices=NUMERICAL_METHODS + VISUAL_METHODS
                              + UPSTREAM_METHODS, default=list(NUMERICAL_METHODS))
    registration.add_argument("--split", choices=("validation", "test"), default="validation")
    registration.add_argument("--seed", type=int, default=20261012)
    registration.add_argument("--model")
    registration.add_argument("--max-requests", type=int, default=0)
    registration.add_argument("--workers", type=int, default=3)
    execution = commands.add_parser("run", help="执行已冻结方案，不自动重试失败调用")
    execution.add_argument("--run", type=Path, required=True)
    execution.add_argument("--env-file", type=Path)
    evaluation = commands.add_parser("evaluate", help="对已保存响应离线评分")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, help="重放评分需指定新的输出目录")
    evaluation.add_argument("--dataset", type=Path,
                            help="补充包迁移后显式指定数据目录；仍核验原登记哈希")
    commands.add_parser("setup-sources", help="核验或下载锁定的原始论文代码")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = data.generate(args.out, args.seed, args.per_split, args.days)
        elif args.command == "register":
            result = experiment.register(args.dataset, args.out, args.methods, args.split,
                                         args.seed, args.model, args.max_requests, args.workers)
        elif args.command == "run":
            result = experiment.execute(args.run, args.env_file)
        elif args.command == "evaluate":
            result = experiment.evaluate(args.run, args.out, args.dataset)
        else:
            receipt = setup_sources()
            result = {"sources": {name: entry["commit"] for name, entry in receipt.items()},
                      "lock": str(ROOT / "configs/upstream.json")}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
