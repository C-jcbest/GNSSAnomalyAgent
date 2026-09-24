"""Explicit collection commands and an offline experiment workflow."""

import argparse
import sys
from pathlib import Path

from .acquisition import Platform, load_environment, local_time, redact
from .contracts import Window
from .datasets import (
    build_dataset,
    build_pilot,
    demo_backgrounds,
    extract_window,
    load_dataset,
    prepare_backgrounds,
)
from .experiments import calibrate, capabilities, run_experiment, summarize
from .plotting import render
from .policies import METHODS
from .storage import child, read_json, write_json


def parser():
    root = argparse.ArgumentParser(description="GNSS 离线异常实验")
    root.add_argument("--env", type=Path, default=Path(".env"))
    sub = root.add_subparsers(dest="command", required=True)
    stations = sub.add_parser("stations", help="联网读取站点清单，私有信息仅保存本地")
    stations.add_argument("--out", type=Path, required=True)
    fetch = sub.add_parser("fetch", help="联网按天采集一个站点，保存不可覆盖的快照")
    fetch.add_argument("--stations", type=Path, required=True)
    fetch.add_argument("--station-index", type=int, required=True, help="清单索引，从 0 开始")
    for name in ("start", "end", "alias"):
        fetch.add_argument("--" + name, required=True)
    fetch.add_argument("--site", help="同一坡体的站点必须使用相同 site 别名")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--resume", action="store_true")
    collection = sub.add_parser("collect", help="按站点批量采集长历史，支持逐日断点续传")
    collection.add_argument("--stations", type=Path, required=True)
    collection.add_argument("--start", required=True)
    collection.add_argument("--end", required=True)
    collection.add_argument("--out", type=Path, required=True)
    collection.add_argument("--workers", type=int, default=4)
    collection.add_argument("--resume", action="store_true")
    extract = sub.add_parser("extract", help="从本地快照按时间提取未标注长/短窗口")
    extract.add_argument("--snapshot", type=Path, required=True)
    extract.add_argument("--start", required=True)
    extract.add_argument("--end", required=True)
    extract.add_argument("--out", type=Path, required=True)
    offline = sub.add_parser("prepare-offline", help="离线收束空白边界、按来源分组准备长短窗口")
    offline.add_argument("--source", type=Path, required=True)
    offline.add_argument("--stations", type=Path, required=True)
    offline.add_argument("--references", type=Path, required=True)
    offline.add_argument("--out", type=Path, required=True)
    offline.add_argument("--seed", type=int, default=20260922)
    daily = sub.add_parser("prepare-daily", help="离线提取北京时间每日15点，准备长期日窗口")
    daily.add_argument("--source", type=Path, required=True)
    daily.add_argument("--out", type=Path, required=True)
    daily.add_argument("--days", type=int, default=180)
    daily_set = sub.add_parser("build-daily-set", help="固定15点、保留来源划分的注入测试集")
    daily_set.add_argument("--source", type=Path, required=True)
    daily_set.add_argument("--out", type=Path, required=True)
    daily_set.add_argument("--seed", type=int, default=20260923)
    daily_set.add_argument(
        "--native-only", action="store_true", help="仅保留来源缺测，不派生额外缺测"
    )
    freeze = sub.add_parser("freeze-daily", help="验证运行完整后冻结配置，不用弱标签调阈值")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--validation-run", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="提取完整背景窗，默认等待人工审核")
    prepare.add_argument("--snapshots", type=Path, nargs="+", required=True)
    prepare.add_argument("--hours", type=int, default=336)
    prepare.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit-backgrounds", help="描述性统计背景范围与跳变，不自动批准正常标签")
    audit.add_argument("--backgrounds", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build", help="从已审核背景生成注入与缺测派生数据")
    build.add_argument("--backgrounds", type=Path, required=True)
    build.add_argument("--per-kind", type=int, default=100)
    build.add_argument("--seed", type=int, default=20260922)
    build.add_argument("--out", type=Path, required=True)
    pilot = sub.add_parser("build-pilot", help="单个已目视筛查开发背景的 24 案例预实验")
    pilot.add_argument("--background", type=Path, required=True)
    pilot.add_argument("--review", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.add_argument("--seed", type=int, default=20260922)
    demo = sub.add_parser("demo", help="生成纯合成工程示例，不能用于论文结论")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=20260922)
    calibration = sub.add_parser("calibrate", help="仅用验证集选择数值阈值")
    calibration.add_argument("--dataset", type=Path, required=True)
    calibration.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    calibration.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="读取本地数据，模型方法按需联网")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    run.add_argument(
        "--split", choices=["development", "validation", "test"], default="development"
    )
    run.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int, help="仅工程冒烟；禁止用它生成能力卡")
    summary = sub.add_parser("summarize")
    observation = sub.add_parser("observe", help="分析匿名未标注窗口，不生成P/R/F1")
    observation.add_argument("--window", type=Path, required=True)
    observation.add_argument("--config", type=Path, required=True)
    observation.add_argument("--out", type=Path, required=True)
    summary.add_argument("--run", type=Path, required=True)
    card = sub.add_parser("capabilities", help="依据完整开发/验证实验生成能力卡")
    card.add_argument("--run", type=Path, required=True)
    card.add_argument("--out", type=Path, required=True)
    plot = sub.add_parser("render")
    source = plot.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--window", type=Path, help="直接绘制背景窗口 JSON，便于人工审核")
    plot.add_argument("--case-id")
    plot.add_argument("--out", type=Path, required=True)
    dashboard = sub.add_parser("serve", help="打开本地历史运行与论文图表工作台")
    dashboard.add_argument("--workspace", type=Path, default=Path("."))
    dashboard.add_argument("--port", type=int, default=8765)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.env.exists():
        load_environment(args.env)
    try:
        if args.command == "observe":
            from .auxiliary import observe

            observe(args.window, args.out, read_json(args.config))
        elif args.command == "serve":
            from .dashboard import serve

            serve(args.workspace, args.port)
        elif args.command == "collect":
            from .collection import collect

            report = collect(
                args.stations,
                args.out,
                local_time(args.start),
                local_time(args.end),
                args.workers,
                args.resume,
            )
            print(f"采集结束：{len(report['results'])} 个站点；全部成功={report['all_successful']}")
            if not report["all_successful"]:
                return 1
        elif args.command == "prepare-daily":
            from .daily import prepare_daily, verify_daily

            prepare_daily(args.source, args.out, args.days)
            report = verify_daily(args.out)
            print(
                f"每日15点：{report['retained_stations']}站；{report['window_days']}天窗口"
                f"{report['long_windows']}个，辅助案例{report['auxiliary_cases']}个。未生成标签。"
            )
        elif args.command == "build-daily-set":
            from .daily_dataset import build_daily_dataset

            count = build_daily_dataset(args.source, args.out, args.seed, args.native_only)
            print(f"生成{count}例；标签仅标受控注入，原生背景未认证。")
        elif args.command == "freeze-daily":
            from .experiments import freeze_daily_validation

            freeze_daily_validation(args.dataset, args.validation_run, args.out)
            print(f"已冻结验证配置：{args.out}；未依据注入标签选阈值。")
        elif args.command == "prepare-offline":
            from .preparation import prepare_offline, verify_preparation

            prepare_offline(args.source, args.stations, args.references, args.out, args.seed)
            report = verify_preparation(args.out)
            print(
                f"收束完成：{report['retained_stations']} 站，移除 {report['removed_hours']} 个空白小时；"
                f"短窗 {report['short_windows']}，长窗 {report['long_windows']}。背景尚待审核。"
            )
        elif args.command == "extract":
            count = extract_window(
                args.snapshot, args.out, local_time(args.start), local_time(args.end)
            )
            print(f"已提取 {count} 个采样位置，保留缺测；未生成真值标签。")
        elif args.command == "stations":
            if args.out.exists():
                raise FileExistsError("站点清单已存在，请使用新路径")
            platform = Platform()
            try:
                rows = platform.stations()
                write_json(args.out, redact(rows))
                print(f"已保存 {len(rows)} 个站点；真实标识只保存在本地清单。")
            finally:
                platform.close()
        elif args.command == "fetch":
            rows = read_json(args.stations)
            if not 0 <= args.station_index < len(rows):
                raise ValueError("站点索引超出范围")
            platform = Platform()
            try:
                audit = platform.fetch(
                    rows[args.station_index],
                    local_time(args.start),
                    local_time(args.end),
                    args.out,
                    args.alias,
                    args.site,
                    resume=args.resume,
                )
                print(f"已保存快照：{audit}")
            finally:
                platform.close()
        elif args.command == "audit-backgrounds":
            from .background_review import audit_backgrounds

            report = audit_backgrounds(args.backgrounds, args.out)
            print(f"已描述性审核 {len(report['items'])} 个背景；未批准正常标签。")
        elif args.command == "prepare":
            count = prepare_backgrounds(args.snapshots, args.out, args.hours)
            print(f"提取 {count} 个完整窗口；请审核 backgrounds.json 中的 approved 字段。")
        elif args.command == "build-pilot":
            count = build_pilot(args.background, args.review, args.out, args.seed)
            print(f"已生成 {count} 个开发预实验输入，不可用于正式结论或能力卡。")
        elif args.command == "build":
            count = build_dataset(args.backgrounds, args.out, args.per_kind, args.seed)
            print(f"已生成 {count} 个输入。")
        elif args.command == "demo":
            if args.out.exists():
                raise FileExistsError("示例目录已存在，请使用新路径")
            demo_backgrounds(args.out / "backgrounds", args.seed)
            count = build_dataset(args.out / "backgrounds", args.out / "dataset", 15, args.seed)
            print(f"已生成 {count} 个纯合成工程示例：{args.out / 'dataset'}")
        elif args.command == "calibrate":
            calibrate(args.dataset, read_json(args.config), args.out)
            print(f"已保存验证集校准配置：{args.out}")
        elif args.command == "run":
            run_experiment(
                args.dataset,
                args.out,
                read_json(args.config),
                args.split,
                args.methods,
                args.resume,
                args.limit,
            )
            print(f"运行完成，请检查完成率与错误记录：{args.out / 'summary.md'}")
        elif args.command == "summarize":
            summarize(args.run)
            print(args.run / "summary.md")
        elif args.command == "capabilities":
            capabilities(args.run, args.out)
            print(f"已生成实测能力卡：{args.out}")
        elif args.command == "render":
            if args.window:
                window = Window.model_validate(read_json(args.window))
            else:
                manifest = load_dataset(args.dataset)
                record = next(
                    (r for r in manifest["records"] if r["case_id"] == args.case_id), None
                )
                if record is None:
                    raise ValueError("未知 case-id；使用 --dataset 时需指定 --case-id")
                window = Window.model_validate(read_json(child(args.dataset, record["file"])))
            if args.out.exists():
                raise FileExistsError("图像已存在，请使用新路径")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(render(window))
            print(args.out)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
