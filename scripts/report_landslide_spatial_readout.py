"""Export complete pilot comparisons, paired record uncertainty and errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores

MAIN = "global_contrast_numeric"
CONTROLS = [
    "duplicate_global_numeric",
    "global_center_numeric",
    "numeric",
    "xgb_raw84",
    "xgb_chart384",
]


def report(data, encoded, run, verified, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(encoded / "protocol.json")
    ids = protocol["split"]["validation"]
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    ledger = read_json(data / "training-generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in ledger if row["case_id"] in ids}
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in ids if families[cid] == family]
            )
            family_rows.append(
                {
                    "method": name,
                    "family": family,
                    **score["daily"],
                    "event_f1_iou03": score["events"]["f1"],
                }
            )
    write_csv(out / "family-summary.csv", family_rows)
    draws = np.random.default_rng(2026101252).integers(0, len(ids), (2000, len(ids)))
    intervals = []
    for control in CONTROLS:
        for metric in ("daily", "events"):
            values = {}
            valid = np.ones(2000, dtype=bool)
            for name in (MAIN, control):
                counts = np.array(
                    [
                        [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
                        for cid in ids
                    ]
                )
                total = counts[draws].sum(axis=1)
                denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
                valid &= denominator > 0
                values[name] = np.divide(
                    2 * total[:, 0], denominator, out=np.full(2000, np.nan), where=denominator > 0
                )
            difference = (values[MAIN] - values[control])[valid]
            intervals.append(
                {
                    "control": control,
                    "metric": metric + "_f1",
                    "defined_draws": int(valid.sum()),
                    "low": float(np.quantile(difference, 0.025)),
                    "high": float(np.quantile(difference, 0.975)),
                }
            )
    write_csv(out / "paired-bootstrap.csv", intervals)
    names = [name for name in predictions if name != MAIN]
    names.insert(3, MAIN)
    table = {row["method"]: row for row in summary}
    figure, panels = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    for panel, key, title in zip(
        panels,
        ("f1", "event_f1_iou03"),
        ("Daily deformation F1", "Event F1 (IoU 0.3)"),
        strict=True,
    ):
        values = [table[name][key] for name in names]
        panel.barh(
            np.arange(len(names)),
            values,
            color=["#b95b45" if name == MAIN else "#52869a" for name in names],
        )
        panel.set_xlim(0, 1.06)
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
        panel.set_axisbelow(True)
        for index, value in enumerate(values):
            panel.text(value + 0.01, index, f"{value:.4f}", va="center", fontsize=9)
    panels[0].set_yticks(np.arange(len(names)), names, fontsize=9)
    panels[0].invert_yaxis()
    figure.suptitle("Center/context readout pilot: 16 training, 8 exposed validation records")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    figure.savefig(out / "daily-event-comparison.png", dpi=160)
    figure.savefig(out / "daily-event-comparison.pdf")
    plt.close(figure)
    net_by_case = {}
    for cid in ids:
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results["duplicate_global_numeric"]["per_case"][cid]["daily"]
        net_by_case[cid] = control["fp"] + control["fn"] - main["fp"] - main["fn"]
    error_ids = list(
        dict.fromkeys(
            [min(ids, key=lambda cid: net_by_case[cid]), max(ids, key=lambda cid: net_by_case[cid])]
        )
    )
    labels = {cid: [] for cid in error_ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    for cid in error_ids:
        case, _ = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=0.6)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
            panels[axis].grid(alpha=0.2)
        states = [
            ("Process reference", np.array([int(row["activity_label"]) for row in labels[cid]]))
        ]
        states.extend(
            (name, np.array(predictions[name][cid]["activity"]))
            for name in ["duplicate_global_numeric", "global_center_numeric", MAIN, "xgb_raw84"]
        )
        for index, (name, state) in enumerate(states):
            display = state.astype(float)
            display[state == -1] = np.nan
            panels[-1].plot(display + index * 1.5, linewidth=0.9)
        panels[-1].set_yticks(
            np.arange(len(states)) * 1.5, [name for name, _ in states], fontsize=8
        )
        panels[-1].set_xlabel("Original day; unknown decisions remain gaps")
        figure.suptitle(
            cid + f" | post-score error analysis, net vs duplicate {net_by_case[cid]:+d}"
        )
        figure.tight_layout()
        figure.savefig(out / (cid + "-error.png"), dpi=130)
        plt.close(figure)
    cost = read_json(encoded / "encoding-cost.json")
    lines = [
        "# 中心—上下文视觉数值读取：机制小试",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent；用户授权自主研究",
        "- Origin Date: 2026-10-12",
        "- Version Label: spatial-readout-pilot-v1",
        "- Verification Status: "
        + read_json(verified / "verification.json")["verification_status"]
        + "；完整概率/活动/阶段与统一评分重放；不是独立评价",
        "",
        "## 实际结论",
        "",
        f"**{decision['decision']}**，{sum(decision['checks'].values())}/7预设条件。主对同维度重复全局/中心无差分净纠错分别{decision['net_correct']['duplicate']:+d}/{decision['net_correct']['center']:+d}日；停止或扩展仅依该冻结门槛，不事后挑模型称新贡献。",
        "",
        "## 方法、公平性与局限",
        "",
        "候选CC-VN（Center-Context Visual-Numeric Readout，中心—上下文视觉数值读取）沿用同一个冻结MobileCLIP2-S0/dfndr2b。仅改变预训练空间图的读取，原三轴图、数据、数值输入和分类器训练固定；不是新Agent框架或新的基础模型。",
        "",
        "$$",
        r"g=\operatorname{norm}(W\operatorname{avg}(F)+b),\quad c=\operatorname{norm}(W\operatorname{avg}_{C}(F)+b),\quad f=\operatorname{norm}(W\operatorname{avg}_{L\cup R}(F)+b),",
        "$$",
        "",
        "$$",
        r"d=\frac{c-f}{2},\qquad p_t=\sigma\!\left(h_\theta([g;d;n_t])\right).",
        "$$",
        "",
        "空间图为1024×8×8，中心取行0:7/列3:5，两侧取相同行/列1,2,5,6。平均池化后使用相同原预训练1024→512投影。Attention已混入全局信息、空间stride32、三轴图区还存在边框/中心标记，因此它只是粗区域读出，不能声称精确逐日对齐或纯局部因果证据。VLM4TS的patch读取及VETime的对齐已有近邻，单独的patch池化/融合不构成新颖性。",
        "",
        "八族按原ID取每族2训练/1验证，共16/8；没有根据分数抽样。神经臂同stride3已知训练日、宽16/32、原seed20261020、12epoch；每臂38机会。同维度重复全局与中心无差分均1060输入，和主方法完全等参数；原全局548、去数值1024、纯数值36维明确披露。强树同划分重训并用全部已知支撑日，2深×19阈值38机会；规则与PELT各9机会。数值阶段头仅在16训练记录重训，各臂共用，A/S/D不选主活动模型。",
        "",
        "所有方法共同采用14阳性观测的持续性判定，仅跨≤7原缺测连接，不跨观测阴性或有观测但无支撑日，缺测仍未知；没有用它单独救主方法。纯视觉臂是预训练视觉编码器小头，并非语言生成VLM；本批未追加直接VLM/固定Agent请求，不混入旧人口分数。",
        "",
        "## 完整统一对比",
        "",
        "| 方法 | P | R | 日F1 | 事件F1(.3) | 事件F1(.5) | FP | FN | 起止MAE（日） | 阶段宏F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary:

        def fmt(key):
            return "未定义" if row[key] is None else f"{row[key]:.6f}"

        lines.append(
            "| "
            + row["method"]
            + " | "
            + " | ".join(
                fmt(key)
                for key in ["precision", "recall", "f1", "event_f1_iou03", "event_f1_iou05"]
            )
            + f" | {row['fp']} | {row['fn']} | "
            + fmt("onset_core_mae_days")
            + "/"
            + fmt("offset_core_mae_days")
            + " | "
            + fmt("stage_end_to_end_macro_f1")
            + " |"
        )
    lines.extend(
        [
            "",
            "完整门槛与分组：",
            "",
            "```json",
            json.dumps(decision, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 代价、不确定性与错误图",
            "",
            f"重新计算{cost['centers']}个中心的空间图，CPU实测{cost['seconds']:.3f}秒；每个中心的全局读出与官方head及原缓存核验通过，最大绝对误差{cost['maximum_global_embedding_absolute_error']:.3g}。原2个案例后的v1进度文件写入失败保留，v2只修正保存机制及基线名称笔误，不改方法或门槛。没有新增API或保留评价访问。",
            "",
            "2000次整记录配对bootstrap仅描述八条已参与选择记录的波动；不把逐日当独立样本，不据此宣称统计显著或现场泛化。分母为零的抽样不定义F1，已披露有效次数。八族各1条验证记录的族表只是个案描述，不能推断族总体。匹配边界MAE可能基于不同事件集合。",
            "",
            "![全部方法日与事件对比](daily-event-comparison.png)",
            "",
            "原图按主对同维度重复全局的净纠错最小/最大各取1条，评分后查看，仅用于原因分析，不改参考或参数：",
            "",
        ]
    )
    for cid in error_ids:
        lines.append(f"![{cid}原始三轴及活动结果]({cid}-error.png)")
        lines.append("")
    lines.extend(
        [
            "## 复现",
            "",
            "运行encode_landslide_spatial_readout.py、train_landslide_spatial_readout.py、verify_landslide_spatial_readout.py和本脚本；源文件、权重SHA、原数据/编码绑定、38配置/臂、选中权重与全部原日概率/活动/阶段留存。common decoder另重放原16概率臂×24记录，与诊断结果一致。",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        out / "analysis-plan.json",
        {
            "bootstrap_seed": 2026101252,
            "draws": 2000,
            "unit": "paired complete record",
            "case_selection": "Minimum and maximum net correct days vs same-dimensional duplicate; after scoring",
            "review_ids": error_ids,
            "net_by_case": net_by_case,
            "labels_modified": False,
            "test_truth_opened": False,
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(
        args.data.resolve(),
        args.encoded.resolve(),
        args.run.resolve(),
        args.verified.resolve(),
        args.out.resolve(),
    )
