"""Draw paper figures from the frozen EXP06 observations and predictions."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ASSETS = ROOT / "figures/exp06"
COLORS = ["#7e8995", "#b78a52", "#498398", "#234f77"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def draw_framework():
    canvas = Image.new("RGB", (1800, 1020), "white")
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    bold = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 34)
    ink = "#24394c"

    def box(rect, title, lines):
        draw.rounded_rectangle(rect, radius=10, fill="#f4f6f8", outline="#718496", width=3)
        x = (rect[0] + rect[2]) / 2
        draw.text((x, rect[1] + 20), title, font=bold, fill=ink, anchor="mt")
        for i, line in enumerate(lines):
            draw.text((x, rect[1] + 76 + 44 * i), line, font=regular, fill=ink, anchor="mt")

    def arrow(start, end):
        draw.line([start, end], fill=ink, width=4)
        x, y = end
        if y > start[1]:
            points = [(x, y), (x - 10, y - 18), (x + 10, y - 18)]
        else:
            points = [(x, y), (x - 18, y - 10), (x - 18, y + 10)]
        draw.polygon(points, fill=ink)

    box((25, 20, 555, 230), "本站正常参考与尖峰分类", ["线性与年周期  MAD归一化", "48维平衡随机森林", "剔除尖峰后拟合位移"])
    box((635, 20, 1165, 230), "三分量位移结构分解", ["PELT分段与局部边缘模型", "有界活动与停止平台", "独立跳变及数值返回配对"])
    box((1245, 20, 1775, 230), "独立四类数值结果", ["尖峰  阶跃  暂态  活动", "输出存在与时间边界", "segment_numeric"])
    arrow((555, 125), (635, 125))
    arrow((1165, 125), (1245, 125))
    arrow((900, 230), (900, 325))
    box((635, 325, 1165, 535), "存在未选中的返回对", ["提供全局原始NEU", "局部原始图与辅助残差图", "专门核验永久变化和返回"])
    draw.text((880, 274), "有歧义才请求", font=regular, fill=ink, anchor="mm")
    arrow((900, 535), (900, 620))
    box((635, 620, 1165, 830), "受限语义修复", ["明确确认  两端各±7日", "两端仍有数值阶跃支持", "以数值边界暂态替换二阶跃"])
    draw.line([(1510, 230), (1510, 725)], fill=ink, width=4)
    draw.text((1520, 455), "保留数值", font=regular, fill=ink)
    draw.text((1520, 500), "独立判决", font=regular, fill=ink)
    arrow((1165, 725), (1510, 725))
    arrow((1510, 725), (1510, 885))
    box((1245, 885, 1775, 1000), "最终 SDR-VLM 结果", ["pair_focus_repair"])
    draw.text((55, 655), "无歧义  不确定  请求失败", font=bold, fill=ink)
    draw.text((55, 715), "均保留独立数值结果", font=regular, fill=ink)
    draw.text((55, 905), "图示、导出与评价共享原始时间轴和缺测掩码", font=regular, fill=ink)
    canvas.save(ASSETS / "framework.png")


def draw_comparisons():
    methods = read_json(REPO / "results/exp06-main-v1/summary.json")["methods"]
    names = ["平衡随机森林", "数值时间解码", "既有完整定位融合", "结构分解 无视觉", "SDR-VLM"]
    keys = ["rf_balanced", "rf_split_temporal", "localized_fusion", "segment_numeric", "pair_focus_repair"]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for offset, field, color, label in [(-.18, "types_daily_macro_f1", "#234f77", "四类等权日F1"), (.18, "types_event_macro_f1", "#88a7b8", "四类等权事件F1")]:
        values = []
        for key in keys:
            value = methods[key][field]
            values.append(value["iou_0.5"] if isinstance(value, dict) else value)
        ax.barh(y + offset, values, .32, color=color, label=label)
        for row, value in zip(y + offset, values):
            ax.text(value + .013, row, f"{value:.4f}", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1.02), xlabel="F1", title="EXP06 同组48站")
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison_overall.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    x = np.arange(4)
    for i, (key, label) in enumerate(zip(keys[1:], names[1:])):
        values = [methods[key]["types"][kind]["daily_micro"]["f1"] for kind in ["spike", "step", "transient", "deformation"]]
        ax.bar(x + (i - 1.5) * .19, values, .18, label=label, color=COLORS[i])
    ax.set(xticks=x, xticklabels=["尖峰", "阶跃", "暂态", "持续形变"], ylim=(0, 1.16), ylabel="日F1")
    ax.legend(loc="upper center", ncols=2, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison_types.png", dpi=220)
    plt.close(fig)


def prediction_labels(run, method, station):
    path = run / "predictions" / method / station / "labels.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {kind: np.array([row["observed"] == "1" and row[kind] == "1" for row in rows]) for kind in ["spike", "step", "transient", "deformation"]}


def draw_case(station, limits, filename, activity=False):
    seed = "2026101405"
    data = REPO / f"data/exp06-seed{seed}"
    run = REPO / f"runs/exp06-seed{seed}-v1"
    observation = read_json(data / "observations" / f"{station}.json")
    truth = read_json(data / "truth" / f"{station}.json")
    displacement = np.asarray([
        row if row is not None else [np.nan, np.nan, np.nan]
        for row in observation["displacement_mm"]
    ], dtype=float)
    observed = np.isfinite(displacement).all(axis=1)
    days = np.arange(len(displacement))
    truth_labels = {kind: np.zeros(len(days), dtype=bool) for kind in ["spike", "step", "transient", "deformation"]}
    for event in truth["events"]:
        truth_labels[event["kind"]][event["start"]:event["end"] + 1] = True
    for mask in truth_labels.values():
        mask &= observed
    numeric = prediction_labels(run, "segment_numeric", station)
    repaired = prediction_labels(run, "pair_focus_repair", station)
    old = prediction_labels(run, "localized_fusion", station)
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 5.9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.65 if not activity else 1.2]})
    target = "deformation" if activity else "transient"
    for dimension, ax in enumerate(axes[:3]):
        ax.plot(days, displacement[:, dimension], color="#355b78", lw=.9)
        ax.fill_between(days, 0, 1, where=truth_labels[target], transform=ax.get_xaxis_transform(), step="mid", color="#d7e8e0", alpha=.6)
        if not activity:
            for index in np.flatnonzero(truth_labels["step"]):
                ax.axvline(index, color="#b46c38", ls="--", lw=.8)
        ax.set_ylabel(["N / mm", "E / mm", "U / mm"][dimension])
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    if activity:
        rows = [("真值活动", truth_labels[target], "#608c78", False), ("旧定位活动", old[target], "#b78a52", False), ("SDR活动", repaired[target], "#234f77", False)]
    else:
        rows = [("真值阶跃", truth_labels["step"], "#b46c38", True), ("数值阶跃", numeric["step"], "#b46c38", True), ("SDR阶跃", repaired["step"], "#b46c38", True), ("真值暂态", truth_labels[target], "#608c78", False), ("数值暂态", numeric[target], "#498398", False), ("SDR暂态", repaired[target], "#234f77", False)]
    ax = axes[3]
    for i, (_, mask, color, point) in enumerate(rows):
        indices = np.flatnonzero(mask)
        if point:
            ax.scatter(indices, np.full(len(indices), i), marker="v", s=38, color=color, zorder=3)
        else:
            ax.bar(indices, np.full(len(indices), .5), bottom=i - .25, width=1, color=color, linewidth=0)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[row[0] for row in rows], ylim=(len(rows) - .45, -.65), xlim=limits, xlabel="原始日索引（0起始）")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    axes[0].set_title(f"{seed} / {station}  " + ("有界位移活动与停止平台" if activity else "暂态恢复与同日起始阶跃损失"), fontsize=11)
    fig.tight_layout(h_pad=.5)
    fig.savefig(ASSETS / f"{filename}.png", dpi=220)
    plt.close(fig)


def prepare_figures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
    draw_framework()
    draw_comparisons()
    draw_case("s0065", (545, 675), "case_transient")
    draw_case("s0052", (465, 729), "case_deformation", activity=True)


if __name__ == "__main__":
    prepare_figures()
