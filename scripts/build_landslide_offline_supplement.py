"""Package frozen data and exact scoring functions for repository-free score replay."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ("data/inputs", "expected", "gnss_sim", "provenance", "figures"):
        (out / name).mkdir(parents=True)
    analysis = BASE / "tama-budget-comparison-v1/analysis-v3"
    run = BASE / "budget-selection-confirmation-v1"
    for folder in (analysis, BASE / "manuscript-v23"):
        for name, expected in read_json(folder / "manifest.json")["files"].items():
            if sha(folder / name) != expected:
                raise ValueError(f"Published freeze changed: {folder}, {name}")
    copies = {
        analysis / "predictions.json": "data/predictions.json",
        run / "reference-subset.csv": "data/reference.csv",
        Path(__file__): "provenance/builder.py",
        ROOT / "scripts/replay_landslide_offline_supplement.py": "replay.py",
        BASE / "manuscript-v23/manuscript-v23.md": "manuscript-v23.md",
        ROOT / "docs/landslide-budget-selection-results.md": "provenance/placement-results.md",
        ROOT / "docs/landslide-tama-budget-comparison-results.md": "provenance/tama-results.md",
        analysis / "comparison.png": "figures/comparison.png",
        run / "protocol.json": "provenance/placement-protocol.json",
        BASE / "tama-budget-comparison-v1/protocol.json": "provenance/tama-protocol.json",
    }
    for name in ("results.json", "internal-stages.json", "summary.csv", "per-case.csv",
                 "family-results.csv", "corrections.csv"):
        copies[analysis / name] = "expected/" + name
    for name in ("__init__.py", "landslide_evaluation.py", "landslide_review_utility.py"):
        copies[ROOT / "src/gnss_sim" / name] = "gnss_sim/" + name
    families = {}
    for cid in read_json(run / "protocol.json")["case_ids"]:
        copies[run / "inputs" / cid / "input.json"] = f"data/inputs/{cid}.json"
        truth = BASE / "budget-selection-confirmation-data-v2/mechanism-reference" / cid / "truth.json"
        families[cid] = read_json(truth)["family"]
        copies[truth] = f"provenance/{cid}-truth.json"
    provenance = []
    for source, destination in copies.items():
        shutil.copyfile(source, out / destination)
        provenance.append({"source": str(source.relative_to(ROOT)).replace("\\", "/"),
                           "packaged": destination, "sha256": sha(source)})
    prefix = ('"""Exact scoring function extraction; see provenance/scoring-functions.json."""\n'
              'import numpy as np\nfrom scipy.optimize import linear_sum_assignment\n'
              'from sklearn.metrics import precision_recall_fscore_support\n'
              'from gnss_sim.landslide_evaluation import (STAGES, ActivityPrediction, '
              'aggregate_scores, evaluate_case, event_masks, score_counts)\n\n')
    functions = []
    for filename, name in (("analyze_landslide_adaptive_review.py", "boundary_errors"),
                           ("analyze_landslide_observation_probe.py", "evaluate_population"),
                           ("summarize_landslide_motion_competition.py", "verify_metrics")):
        source = ROOT / "scripts" / filename
        text = source.read_text(encoding="utf-8")
        node = next(item for item in ast.parse(text).body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        function = ast.get_source_segment(text, node)
        prefix += function + "\n\n"
        shutil.copyfile(source, out / "provenance" / filename)
        functions.append({"name": name, "source": "provenance/" + filename,
                          "source_sha256": sha(source), "first_line": node.lineno,
                          "last_line": node.end_lineno,
                          "function_sha256": hashlib.sha256(function.encode()).hexdigest()})
    (out / "scoring.py").write_text(prefix, encoding="utf-8")
    write_json(out / "provenance/scoring-functions.json", functions)
    write_json(out / "provenance/copied-files.json", provenance)
    write_json(out / "data/families.json", families)
    with (analysis / "summary.csv").open(encoding="utf-8", newline="") as handle:
        usage = {row["method"]: {"deployment_requests": int(row["deployment_requests"]),
                                 "reported_tokens": int(row["reported_tokens"])}
                 for row in csv.DictReader(handle)}
    write_json(out / "data/usage.json", usage)
    write_json(out / "expected/daily-digest.json", {"sha256": sha(analysis / "daily.csv")})
    (out / "requirements.txt").write_text(
        "numpy==2.2.6\nscipy==1.15.3\nscikit-learn==1.7.1\n"
        "joblib==1.6.0\nthreadpoolctl==3.7.0\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# GNSS有限视觉复核：离线评分复现包\n\n"
        "本包从冻结的41组预测及12条受控机制参考重算完整活动、弱召回、事件IoU .3/.5、"
        "匹配边界、端到端和活动内A/S/D、家族与纠错表。所有CSV与原归档逐字节比较。"
        "原评分函数按AST精确提取，来源与函数哈希见provenance；原源码不改。\n\n"
        "Python 3.11，先安装requirements.txt，再在解压后的本目录执行：\n\n"
        "```shell\npython replay.py --out ../replayed-results\n```\n\n"
        "输出目录必须不存在。无需原仓库、绝对历史路径、API凭据或联网推理。"
        "本包是已归档预测的评分复现，不重新训练数值模型、生成数据或调用VLM；"
        "调用和token是归档部署元数据，不能写为重新测得。模型请求、完整训练及其它历史"
        "批次的复现入口仍在原仓库补充报告。本包未覆盖所有历史试验。\n\n"
        "参考为潜在机制标签，不是正式领域复核参考；不支持现场准确率。"
        "两遍不视为独立记录。主方法未整体优于强XGBoost，TAMA适配有误报与覆盖优势，"
        "范围严格增益未完全复制。所有41组保留，不挑最佳组。\n\n"
        "manuscript-v23.md为内容快照，其原仓库相对补充链接不会在本包中全部解析；"
        "本包图在figures/comparison.png，当前完整稿与所有原图在原仓库。\n",
        encoding="utf-8")
    write_json(out / "manifest.json", {"files": {
        str(path.relative_to(out)).replace("\\", "/"): sha(path)
        for path in sorted(out.rglob("*")) if path.is_file()}})
    print(json.dumps({"files": len(read_json(out / "manifest.json")["files"]),
                      "methods": len(usage), "records": len(families), "new_model_requests": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
