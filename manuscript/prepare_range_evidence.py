"""Build manuscript tables and figures from saved EXP37/45 evidence, without inference."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

REPO = Path(__file__).resolve().parents[1]
REVISION = REPO / "manuscript/revisions/range-20261018"
FIGURES = REVISION / "figures"
SOURCE_DATA = Path("D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp30/data")
SOURCES = {}
FONT = FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
plt.rcParams.update(
    {
        "font.family": "SimSun",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
COLORS = ["#246e8c", "#91aab4", "#616161", "#b7b7b7"]


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def daily(scores, kind):
    return scores["original"]["types"][kind]["daily_micro"]


def aff(scores, kind):
    return scores["range"][kind]["affiliation"]


def save(figure, key):
    figure.tight_layout()
    figure.savefig(FIGURES / f"{key}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    main_scores = read(REPO / "results/exp37-independent-test-v1/summary.json")["methods"]
    confirmation = read(REPO / "results/exp37-fresh-confirmation-v1/summary.json")["methods"]
    shape = read(REPO / "results/exp45-confirmation-v1/summary.json")["methods"]
    labels = {
        "owv_flash": "图像数值联合",
        "owv_flash_text": "文字数值联合",
        "vector_nearest": "去视觉数值组件",
        "segment_numeric": "分段数值方法",
        "rf_split_temporal": "分粒度随机森林",
        "random_forest": "随机森林默认",
        "rf_balanced": "平衡随机森林",
        "extra_trees": "极端随机树",
        "owv_flash_pure": "图像独立输出",
        "owv_flash_text_pure": "文字独立输出",
        "uninformed_full_monitoring_pure": "无信息全监测期均报",
    }
    tables = {}
    tables["main"] = [["方法", "单点日F1", "阶跃日F1", "暂态Aff F1", "形变Aff F1"]]
    for method in [
        "owv_flash",
        "rf_split_temporal",
        "random_forest",
        "rf_balanced",
        "extra_trees",
        "segment_numeric",
    ]:
        s = main_scores[method]
        tables["main"].append(
            [
                labels[method],
                *[f"{daily(s, k)['f1']:.4f}" for k in ("spike", "step")],
                *[f"{aff(s, k)['station_macro_f1']:.4f}" for k in ("transient", "deformation")],
            ]
        )
    tables["range"] = [["暂态读出", "Aff P", "Aff R", "Aff F1", "日F1", "阴性FP日"]]
    tables["counts"] = [["暂态读出", "日P", "日R", "TP日", "FP日", "FN日"]]
    for method in [
        "owv_flash",
        "owv_flash_text",
        "vector_nearest",
        "owv_flash_pure",
        "owv_flash_text_pure",
        "uninformed_full_monitoring_pure",
    ]:
        s = main_scores[method]
        a = aff(s, "transient")
        d = daily(s, "transient")
        tables["range"].append(
            [
                labels[method],
                *[f"{a[k]:.4f}" for k in ("precision", "recall", "station_macro_f1")],
                f"{d['f1']:.4f}",
                str(s["range"]["transient"]["negative_station_fp_days"]),
            ]
        )
        tables["counts"].append(
            [
                labels[method],
                f"{d['precision']:.4f}",
                f"{d['recall']:.4f}",
                *[str(d[k]) for k in ("tp", "fp", "fn")],
            ]
        )
    tables["point"] = [["异常类型", "日P", "日R", "日F1", "TP FP FN日"]]
    for kind, name in [
        ("spike", "单点异常"),
        ("step", "永久阶跃"),
        ("deformation", "持续形变辅助"),
    ]:
        d = daily(main_scores["owv_flash"], kind)
        tables["point"].append(
            [
                name,
                *[f"{d[k]:.4f}" for k in ("precision", "recall", "f1")],
                " / ".join(str(d[k]) for k in ("tp", "fp", "fn")),
            ]
        )
    tables["dev"] = [["54站开发确认", "暂态Aff F1", "暂态日F1", "TP FP FN日"]]
    for method in [
        "owv_flash",
        "owv_flash_text",
        "vector_nearest",
        "owv_flash_pure",
        "owv_flash_text_pure",
    ]:
        s = confirmation[method]
        d = daily(s, "transient")
        tables["dev"].append(
            [
                labels[method],
                f"{aff(s, 'transient')['station_macro_f1']:.4f}",
                f"{d['f1']:.4f}",
                " / ".join(str(d[k]) for k in ("tp", "fp", "fn")),
            ]
        )
    tables["shape"] = [["108站形变读出", "Aff F1", "日F1", "TP FP FN日", "阴性FP日"]]
    for method, name in [
        ("nab_image", "图像数值择优"),
        ("segment_numeric", "去视觉数值组件"),
        ("nab_text", "文字数值择优"),
        ("visual_bounds_image", "条件图像直接边界"),
        ("visual_bounds_text", "条件文字直接边界"),
    ]:
        s = shape[method]
        d = daily(s, "deformation")
        tables["shape"].append(
            [
                name,
                f"{aff(s, 'deformation')['station_macro_f1']:.4f}",
                f"{d['f1']:.4f}",
                " / ".join(str(d[k]) for k in ("tp", "fp", "fn")),
                str(s["range"]["deformation"]["negative_station_fp_days"]),
            ]
        )
    tables["seeds"] = [
        ["测试根种子", "图像融合Aff F1", "文字融合Aff F1", "去视觉Aff F1", "新增TP日"],
        ["2026101601", ".9893", ".9276", ".8993", "66"],
        ["2026101602", ".9947", ".9235", ".8990", "174"],
        ["2026101603", ".9720", ".9248", ".9154", "99"],
    ]
    tables["scenarios"] = [
        ["暂态阳性场景", "读出", "Aff F1", "TP FP FN日"],
        ["单独暂态60站", "图像数值联合", ".9998", "2373 / 22 / 7"],
        ["单独暂态60站", "文字数值联合", ".9409", "2247 / 748 / 133"],
        ["单独暂态60站", "去视觉数值组件", ".9167", "2143 / 0 / 237"],
        ["混合60站", "图像数值联合", ".9708", "1856 / 615 / 151"],
        ["混合60站", "文字数值联合", ".9097", "1815 / 975 / 192"],
        ["混合60站", "去视觉数值组件", ".8925", "1747 / 85 / 260"],
    ]
    # Fixed published report strata are checked against the saved per-station records below.
    path = REPO / "results/exp37-independent-test-v1/per-station.json.gz"
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads(gzip.decompress(path.read_bytes()))
    for method in ["owv_flash", "owv_flash_text", "vector_nearest"]:
        for seed in [2026101601, 2026101602, 2026101603]:
            rows = [
                r
                for r in saved[method]
                if r["cohort_seed"] == seed and not r["range"]["transient"]["negative_station"]
            ]
            value = np.mean([r["range"]["transient"]["affiliation"]["f1"] or 0 for r in rows])
            assert (
                abs(
                    value
                    - float(
                        next(r for r in tables["seeds"][1:] if int(r[0]) == seed)[
                            {"owv_flash": 1, "owv_flash_text": 2, "vector_nearest": 3}[method]
                        ]
                    )
                )
                < 0.000051
            )
    for row in tables["scenarios"][1:]:
        method = {v: k for k, v in labels.items()}[row[1]]
        scenario = "transient" if row[0].startswith("单独") else "mixed"
        rows = [r for r in saved[method] if r["scenario"] == scenario]
        scores = [r["range"]["transient"]["affiliation"]["f1"] or 0 for r in rows]
        assert abs(np.mean(scores) - float(row[2])) < 0.000051
        assert (
            " / ".join(
                str(sum(r["original"]["types"]["transient"]["daily"][k] for r in rows))
                for k in ("tp", "fp", "fn")
            )
            == row[3]
        )
    (REVISION / "working/tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    methods = ["owv_flash", "owv_flash_text", "vector_nearest", "rf_balanced"]
    for axis, metric, title in [
        (axes[0], lambda s: aff(s, "transient")["station_macro_f1"], "暂态范围指标"),
        (axes[1], lambda s: daily(s, "transient")["f1"], "暂态辅助日指标"),
    ]:
        axis.barh(np.arange(4), [metric(main_scores[m]) for m in methods], color=COLORS)
        axis.set_yticks(np.arange(4), [labels[m] for m in methods])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.07)
        axis.set_xlabel("Affiliation F1" if axis == axes[0] else "日F1")
        axis.set_title(title)
        for i, m in enumerate(methods):
            axis.text(
                metric(main_scores[m]) + 0.015,
                i,
                f"{metric(main_scores[m]):.3f}",
                va="center",
                fontsize=9,
            )
    save(fig, "metrics")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, field, title in [(axes[0], "tp", "正确检出日"), (axes[1], "fp", "误报日")]:
        values = [
            daily(main_scores[m], "transient")[field]
            for m in ["owv_flash_pure", "owv_flash_text_pure"]
        ]
        ax.bar(["图像独立输出", "文字独立输出"], values, color=COLORS[:2])
        ax.set_title(title)
        ax.set_ylabel("日数")
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.025, str(v), ha="center")
        ax.set_ylim(0, max(values) * 1.18)
    save(fig, "modality")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for j, (name, vals) in enumerate(
        [("图像融合", [0.9998, 0.9708]), ("文字融合", [0.9409, 0.9097])]
    ):
        axes[0].bar(np.arange(2) + (j - 0.5) * 0.3, vals, width=0.3, label=name, color=COLORS[j])
    axes[0].set_xticks([0, 1], ["单独暂态", "混合事件"])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("暂态Affiliation F1")
    axes[0].legend(fontsize=8)
    axes[1].bar(["正常", "单点", "阶跃", "形变"], [2502, 2764, 4011, 2125], color=COLORS[0])
    axes[1].set_ylabel("无暂态场景的暂态误报日")
    save(fig, "strata")
    for station, key in [("e0135", "positive"), ("e0260", "negative")]:
        obs = read(SOURCE_DATA / "observations" / f"{station}.json")
        truth = read(SOURCE_DATA / "truth" / f"{station}.json")
        values = np.array([v if v is not None else [np.nan] * 3 for v in obs["displacement_mm"]])
        truth_events = [e for e in truth["events"] if e["kind"] == "transient"]
        selected = {
            m: next(r for r in saved[m] if r["id"] == station)["events"]["events"]
            for m in ["vector_nearest", "owv_flash"]
        }
        fig, axes = plt.subplots(
            4, 1, figsize=(8.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.85]}
        )
        for i, ax in enumerate(axes[:3]):
            ax.plot(np.arange(730), values[:, i], color="#555555", linewidth=0.75)
            ax.set_ylabel("NEU"[i] + "位移 mm")
            ax.grid(alpha=0.15)
        rows = [
            truth_events,
            *[
                [e for e in selected[m] if e["kind"] == "transient"]
                for m in ["vector_nearest", "owv_flash"]
            ],
        ]
        for i, events in enumerate(rows):
            for event in events:
                observed = np.isfinite(values).all(axis=1)
                days = np.arange(event["start"], event["end"] + 1)
                days = days[observed[days]]
                axes[3].scatter(
                    days,
                    np.full(len(days), i),
                    marker="|",
                    s=95,
                    color=["#2b773f", "#616161", "#246e8c"][i],
                )
        axes[3].set_yticks([0, 1, 2], ["暂态真值", "去视觉", "图像融合"])
        axes[3].set_ylim(-0.6, 2.6)
        axes[3].set_xlim(365, 729)
        axes[3].set_xlabel("原始日索引 缺测处不着色")
        axes[0].set_title(station + " 原始观测及暂态标签")
        save(fig, key)
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    names = ["图像数值择优", "去视觉数值组件", "条件图像直接边界", "条件文字直接边界"]
    for j, (field, vals) in enumerate(
        [
            ("Affiliation F1", [0.980245, 0.980245, 0.722322, 0.683219]),
            ("日F1", [0.932374, 0.932374, 0.767105, 0.598306]),
        ]
    ):
        ax.bar(np.arange(4) + (j - 0.5) * 0.33, vals, width=0.33, label=field, color=COLORS[j])
    ax.set_xticks(np.arange(4), names)
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=2, fontsize=9)
    ax.set_ylabel("108站形变指标")
    save(fig, "deformation")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.65, 0.26, 0.22, "原始NEU与正常历史\n观测掩码和本站噪声尺度"),
        (0.37, 0.65, 0.26, 0.22, "数值位移结构\n单点 阶跃 暂态 形变"),
        (0.72, 0.65, 0.26, 0.22, "数值拥有点类与形变\n不扩展阶跃平台为范围"),
        (0.02, 0.15, 0.26, 0.22, "三个重叠时间窗\n三轴联合图像 无目标候选"),
        (0.37, 0.15, 0.26, 0.22, "视觉提出暂态范围\n观测支持与反向返回"),
        (0.72, 0.15, 0.26, 0.22, "固定类型融合\n暂态补充 其他类型保留"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="white", edgecolor="#666666", linewidth=1)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for start, end in [
        ((0.28, 0.76), (0.37, 0.76)),
        ((0.63, 0.76), (0.72, 0.76)),
        ((0.15, 0.65), (0.15, 0.37)),
        ((0.28, 0.26), (0.37, 0.26)),
        ((0.63, 0.26), (0.72, 0.26)),
        ((0.85, 0.65), (0.85, 0.37)),
    ]:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2}
        )
    save(fig, "framework")
    (REVISION / "working/evidence-sources.json").write_text(
        json.dumps(SOURCES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(tables)} tables and 7 figures; no model requests")


if __name__ == "__main__":
    main()
