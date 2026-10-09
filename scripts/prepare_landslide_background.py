"""Generate new backgrounds once; serve only the development observation review bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide_background import (  # noqa: E402
    generate_background,
    local_review_windows,
    profile_schedule,
    render_raw_review,
    schedule_summary,
)
from gnss_sim.landslide_diagnostics import derive_motion, render_diagnostics  # noqa: E402
from gnss_sim.landslide_observation_review import ObservationReview  # noqa: E402

ACTIVITY_PROMPT = """先仅查看无标签的原始 N/E/U 位移图，定位持续位移活动。
确认的是可观测持续变化，而不是隐藏的物理机制。平台、噪声、单次阶跃不自动等于持续活动；
仅靠单序列不能确定缓慢变化来自坡体还是仪器。全局上下文与全部固定局部图共同查看，
使用原始日索引和毫米单位，不依据速度／加速度提前响应确认位移。
对每段活动给出最早可能起点、保守确认起点、保守确认终点、最晚可能终点，并写原图证据。
明确审查过的静稳范围另列；边界待定、缺测和未审日期保持未知。不在这一层输出阶段。
本提示为参考复核草案，未发送模型；不得当作已完成独立标注。"""

STAGE_PROMPT = """第一层原始位移活动审查完成并冻结后，才查看 31/61/91 日辅助图。
只在保守确认活动区间内部判断低速形变、加速、近稳速、减速或未知。
结合原始位移弯曲形态、分量速度、切向加速度及跨窗口一致性；
3D 速度零附近正偏、端点无支撑、长缺测、转折待定均不能变成确定阶段。
阶段范围必须属于确认活动；导出逐日标签时应用观测缺测与固定 61 日辅助支撑掩码。
居中估计包含未来数据，不能称在线提前预警。先记录活动审查 SHA-256，再提交阶段。
本提示为参考复核草案，未发送模型。"""


def historical_snapshot() -> dict:
    hashes = {}
    for name in ("axis", "boundary", "contract"):
        run = ROOT / f"artifacts/landslide-{name}-2026-10-08/run-v1"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        for source, expected in manifest["source_sha256"].items():
            path = ROOT / source
            if sha(path) != expected:
                raise ValueError(f"Historical source differs from freeze: {source}")
            hashes[source] = expected
        for path in run.rglob("*"):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return hashes


def write_observations_csv(path: Path, case):
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["case_id", "day_index", "date", "N_mm", "E_mm", "U_mm", "observed"])
        for day, (date, row) in enumerate(zip(case.dates, case.displacement_mm)):
            writer.writerow([case.case_id, day, date, *(row or ["", "", ""]), int(row is not None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/landslide-background-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/landslide-background-2026-10-08/batch-v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists():
        raise ValueError("Output exists; never overwrite or selectively redraw a frozen batch")
    args.output.mkdir(parents=True)
    public = args.output / "public"
    private = args.output / "private"
    public.mkdir()
    private.mkdir()
    start_time = time.monotonic()
    write_json(private / "run-ledger-start.json", {
        "status": "running", "workflow": "ARS experiment-agent/run",
        "command": sys.argv, "planned_cases": 24, "model_call_budget": 0,
        "expected_public_records": 12, "hard_timeout_seconds": 1200,
    })
    before = historical_snapshot()
    write_json(private / "historical-freeze-before.json", before)
    shutil.copyfile(args.config, private / "protocol.json")
    sources = ["src/gnss_sim/landslide.py", "src/gnss_sim/landslide_background.py",
               "src/gnss_sim/landslide_diagnostics.py", "src/gnss_sim/landslide_observation_review.py",
               "scripts/prepare_landslide_background.py", "scripts/landslide_background_page.html",
               "scripts/export_landslide_observation_review.py",
               "configs/landslide-background-v1.json", "docs/landslide-background-protocol.md",
               "docs/landslide-background-review-guide.md"]
    source_hashes = {}
    for source in sources:
        copy = private / "source" / source
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, copy)
        source_hashes[source] = sha(copy)
    manifest = {"schema_version": "landslide-development-review-v1", "days": config["days"],
                "reference_status": config["reference_status"], "new_model_calls": 0,
                "sealed_test_count": sum(config["profiles_per_split"].values()),
                "activity_prompt": ACTIVITY_PROMPT, "stage_prompt": STAGE_PROMPT, "cases": []}
    private_cases = []
    input_hashes = set()
    for split in config["split_seeds"]:
        for index, profile in enumerate(profile_schedule(config, split)):
            if time.monotonic() - start_time > 1200:
                raise TimeoutError("Background preparation exceeded its fixed 1200-second limit")
            case, truth, design = generate_background(config, split, index, profile)
            input_bytes = case.model_dump_json().encode("utf-8")
            case_hash = hashlib.sha256(input_bytes).hexdigest()
            if case_hash in input_hashes:
                raise ValueError("Repeated observation background across the batch")
            input_hashes.add(case_hash)
            case_dir = private / split / case.case_id
            case_dir.mkdir(parents=True)
            (case_dir / "input.json").write_bytes(input_bytes)
            write_json(case_dir / "truth.json", truth.model_dump(mode="json"))
            write_json(case_dir / "design.json", design)
            private_cases.append({"split": split, "case_id": case.case_id,
                                  "input_sha256": case_hash, "profile": profile})
            if split == "sealed_test":
                print(f"sealed_test {index + 1}: saved without review/rendering", flush=True)
                continue
            (public / "observations").mkdir(exist_ok=True)
            (public / "images").mkdir(exist_ok=True)
            (public / "diagnostics").mkdir(exist_ok=True)
            (public / "reviews").mkdir(exist_ok=True)
            input_name = f"observations/{case.case_id}.json"
            csv_name = f"observations/{case.case_id}.csv"
            (public / input_name).write_bytes(input_bytes)
            write_observations_csv(public / csv_name, case)
            observed_days = sum(row is not None for row in case.displacement_mm)
            record = {"case_id": case.case_id, "input": input_name, "csv": csv_name,
                      "input_sha256": case_hash, "start_date": str(case.dates[0]),
                      "end_date": str(case.dates[-1]), "observed_days": observed_days,
                      "missing_days": len(case.dates) - observed_days,
                      "review": f"reviews/{case.case_id}.pending.json", "views": [], "diagnostics": {}}
            windows = [{"target": [0, config["days"] - 1], "view": [0, config["days"] - 1]}]
            windows.extend(local_review_windows(config["days"], config["plot"]["local_target_days"],
                                                 config["plot"]["context_days"]))
            for ordinal, window in enumerate(windows):
                png, audit = render_raw_review(
                    case, window["view"][0], window["view"][1] + 1, config["plot"],
                    config["plot"]["global_tick_days" if ordinal == 0 else "local_tick_days"],
                )
                image_name = f"images/{case.case_id}-raw-{ordinal}.png"
                (public / image_name).write_bytes(png)
                record["views"].append({**window, "image": image_name, "image_sha256": sha(public / image_name),
                                        "audit": audit})
            for window_days in config["diagnostic_windows"]:
                diagnostics = derive_motion(case, window_days)
                json_name = f"diagnostics/{case.case_id}-{window_days}.json"
                image_name = f"images/{case.case_id}-aux-{window_days}.png"
                write_json(public / json_name, diagnostics.model_dump(mode="json"))
                (public / image_name).write_bytes(render_diagnostics(case, diagnostics))
                record["diagnostics"][str(window_days)] = {"data": json_name, "image": image_name,
                                                         "image_sha256": sha(public / image_name)}
            review = ObservationReview(case_id=case.case_id, input_sha256=case_hash,
                                       raw_image_sha256=record["views"][0]["image_sha256"])
            write_json(public / record["review"], review.model_dump(mode="json"))
            manifest["cases"].append(record)
            print(f"development {index + 1}: four raw views + three auxiliary views saved", flush=True)
    write_json(public / "manifest.json", manifest)
    write_json(public / "protocol.json", {"protocol_id": config["protocol_id"], "plot": config["plot"],
                                         "diagnostic_windows": config["diagnostic_windows"],
                                         "reference_status": config["reference_status"],
                                         "model_call_budget_this_step": 0})
    shutil.copyfile(ROOT / "scripts/landslide_background_page.html", public / "index.html")
    shutil.copyfile(ROOT / "docs/landslide-background-review-guide.md", public / "review-guide.md")
    after = historical_snapshot()
    if before != after:
        raise ValueError("Historical frozen artifacts changed during preparation")
    artifact_hashes = {path.relative_to(args.output).as_posix(): sha(path)
                       for path in args.output.rglob("*") if path.is_file()}
    write_json(private / "generation-manifest.json", {
        "status": "complete", "reference_status": config["reference_status"], "new_model_calls": 0,
        "source_sha256": source_hashes, "schedule_counts": schedule_summary(config),
        "cases": private_cases, "artifact_sha256": artifact_hashes,
        "historical_files_unchanged": len(before), "elapsed_seconds": time.monotonic() - start_time,
        "test_reviewed": False, "test_rendered": False, "resampled_cases": 0,
    })
    print("COMPLETE: 24 inputs; 12 development review packets; 12 test records sealed", flush=True)


if __name__ == "__main__":
    main()
