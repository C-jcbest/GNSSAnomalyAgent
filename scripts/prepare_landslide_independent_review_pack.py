"""Export observation-only materials and explicitly unreviewed reference forms."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from run_landslide_comparison import save_json, sha
from run_landslide_window_optimization import read_json


def prepare(activity_run, out):
    out.mkdir(parents=True, exist_ok=False)
    case_ids = read_json(activity_run / "protocol.json")["case_ids"]
    rows = []
    for cid in case_ids:
        source = activity_run / "inputs" / cid / "input.json"
        case = json.loads(source.read_text(encoding="utf-8"))
        folder = out / "observations" / cid
        folder.mkdir(parents=True)
        shutil.copyfile(source, folder / "input.json")
        shutil.copyfile(activity_run / "images" / f"{cid}-raw.png", folder / "global-raw.png")
        for local in sorted((activity_run / "images").glob(f"{cid}-window-*.png")):
            shutil.copyfile(local, folder / local.name)
        for index, value in enumerate(case["displacement_mm"]):
            rows.append({"case_id": cid, "day_index": index, "date": case["dates"][index],
                         "observed": int(value is not None), "review_status": "unreviewed",
                         "activity_label": "", "feature": "", "episode_id": "",
                         "evidence": "", "reviewer_id": ""})
    with (out / "unreviewed-daily-reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "reviewer-provenance.json", {
        "status": "awaiting_independent_domain_reviewer", "reviewer_id": None,
        "qualification": None, "prior_exposure_to_predictions": None,
        "prior_exposure_to_generation_or_process_labels": None,
        "observations_reviewed": None, "review_date": None,
        "adjudicator_id": None, "conflicts_resolved": None})
    instructions = """# 独立观测参考复核材料

本包只含12条原始N/E/U观测、相同规则的全局／固定局部原图和未复核表。
没有待测预测、活动色带、生成场景、潜在速度或过程标签。
所有表行是unreviewed，空活动／阶段字段不是unknown或正常标签，不能用于评分。

请先记录复核者专业背景、身份与已有材料暴露情况。需具备领域背景，且未见待测
预测与生成过程；AI输出或本包自动生成本身不构成独立专家复核。

先看原始N/E/U的连续累积变化，确定活动内部与平台；噪声、一个阶跃、衍生峰或
保留累计位移的平台不足以确认活动。起止不确定时留未知，不假造精确物理边界。
活动标签1=可确认持续位移，0=在此评价定义下未确认，-1=已复核但不确定。
缺测两层留空并注明missing，不补零或压缩时间。

活动冻结后再判断A=增速、S=可辨非零近恒速、D=减速。低速不是S，方向正负不是A/D。
只在确认活动内部赋阶段；支撑不足或转折待定保留阶段未知。记录原始位移证据、
可能边界和待定原因。图中最小纵轴跨度会隐藏弱变化，应查看固定局部原图。

两名互盲复核者分别复制本包作答，再仲裁分歧，是优选安排。若只能单人，必须
披露其资格、暴露和仲裁缺口。不要在读过模型结果后修改参考以配合预测。
返回表的review_status必须逐日区分reviewed、unknown、missing及unreviewed。
最终以审核后的参考另立评分运行，不修改既有过程恢复分数和冻结预测。
"""
    (out / "README.md").write_text(instructions, encoding="utf-8")
    save_json(out / "manifest.json", {"status": "unreviewed_not_scorable",
                                      "case_ids": case_ids, "files": {
                                          str(path.relative_to(out)): sha(path)
                                          for path in out.rglob("*") if path.is_file()}})
    shutil.make_archive(str(out), "zip", out)
    print("Observation-only pack saved; reference remains unreviewed", len(rows), "calendar days")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.activity_run, args.out)


if __name__ == "__main__":
    main()
