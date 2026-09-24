"""Publication exports use the same read-only aggregates as the dashboard."""

import csv
import io
import json
import textwrap
import threading
import zipfile
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .contracts import CHANNELS
from .detectors import segments

LOCK = threading.RLock()  # Matplotlib's global state is not safe across HTTP worker threads.
FORMATS = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
CHARTS = ("overview", "heatmap", "missing", "efficiency", "ablation")
PALETTE = [
    "#22645c",
    "#bc7343",
    "#64788f",
    "#867094",
    "#b8903b",
    "#729383",
    "#bb7370",
    "#434c57",
    "#8c9caa",
    "#ac927a",
]
METHODS = [
    "hampel",
    "cusum",
    "reference_trend",
    "iforest",
    "visual",
    "daily_union",
    "fixed",
    "split_fusion",
    "union_diagnostic",
    "rule",
    "generic",
    "lma",
    "lma_no_vision",
    "lma_no_review",
]
LABELS = {
    "hampel": "Hampel",
    "cusum": "CUSUM",
    "reference_trend": "Reference trend",
    "iforest": "Isolation Forest",
    "visual": "Visual",
    "daily_union": "Paired H+Visual union",
    "fixed": "Fixed N+V",
    "split_fusion": "Independent N+V",
    "union_diagnostic": "H+Visual union (derived)",
    "rule": "Rule",
    "generic": "Generic Agent",
    "lma": "LMA",
    "lma_no_vision": "LMA w/o vision",
    "lma_no_review": "LMA w/o review",
}
KINDS = ["spike", "step", "drift", "acceleration", "variance", "normal"]
LABEL_COLORS = {
    "spike": "#bb5b47",
    "step": "#ad772e",
    "drift": "#328466",
    "acceleration": "#846693",
    "variance": "#327f98",
}
STATES = ["native", "complete", "random10", "random25", "block25"]
METRICS = {
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "completion_rate": "Completion rate",
    "normal_false_alarm_rate_completed": "Uninjected-window report rate (not field FAR)",
    "first_attempt_completion_rate": "First-attempt model completion rate",
    "numerical_coverage": "Numerical observed-cell coverage",
}


def footer(detail):
    meta = detail["meta"]
    mode = (
        "ENGINEERING DEMO"
        if meta["dataset_kind"] == "synthetic_demo_only"
        else "PROVISIONAL PILOT - injection-only truth"
        if meta["dataset_kind"] == "platform_pilot_provisional"
        else "DAILY INJECTION - native background unverified"
        if meta["dataset_kind"] == "daily_injection_unverified_v1"
        else "Controlled injection"
    )
    extras = " | PARTIAL RECORDS" if not detail["summary"]["all_records_present"] else ""
    if meta.get("limit") is not None:
        extras += " | SMOKE RUN"
    return f"{mode} | {meta['split']} | run {meta['run_id'][:12]}{extras}"


def selected_rows(detail, dimension="all", value=None, methods=None):
    return [
        r
        for r in detail["summary"]["summary"]
        if r["dimension"] == dimension
        and (value is None or r["value"] == value)
        and (not methods or r["method"] in methods)
    ]


def save(figure, fmt, note):
    if fmt not in FORMATS:
        raise ValueError("图像格式必须为 png/svg/pdf")
    figure.text(0.015, 0.014, textwrap.fill(note, 145), fontsize=8, color="#69736f")
    output = io.BytesIO()
    figure.savefig(output, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def figure_bytes(
    detail,
    chart,
    fmt="svg",
    metric="f1",
    dimension="kind",
    methods=None,
    case=None,
    bounds=None,
    trend=False,
):
    if chart not in (*CHARTS, "case", "labels", "snapshot") or metric not in METRICS:
        raise ValueError("未知图表或指标")
    if dimension not in ("kind", "state", "semantic"):
        raise ValueError("不支持的分组维度")
    with (
        LOCK,
        plt.rc_context(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.labelcolor": "#344640",
                "text.color": "#253a34",
            }
        ),
    ):
        fig = None
        try:
            if chart in ("case", "labels", "snapshot"):
                window, truth, prediction = case
                if chart == "labels" and (truth is None or prediction is not None):
                    raise ValueError("标注图只接受评测标签，不叠加检测预测")
                is_daily = window.sampling_hours == 24
                if is_daily:
                    trend = False  # Already exact daily observations; never aggregate again.
                values = window.array()
                a, b = bounds or (0, len(values) - 1)
                if not 0 <= a < b < len(values):
                    raise ValueError("曲线范围无效，请选择至少两个采样位置")
                fig, axes = plt.subplots(
                    3, 1, figsize=(10.4, 5.4 if chart == "labels" else 6.4), sharex=True
                )
                label_color = LABEL_COLORS.get(
                    truth.events[0].kind if truth and truth.events else "", "#327f98"
                )
                for c, ax in enumerate(axes):
                    if chart == "labels":
                        for event in truth.events:
                            if (
                                CHANNELS[c] in event.channels
                                and event.start <= b
                                and event.end >= a
                            ):
                                ax.axvspan(
                                    max(a, event.start) - 0.5,
                                    min(b, event.end) + 0.5,
                                    color=label_color,
                                    alpha=0.14,
                                )
                                if a <= event.start <= b:
                                    ax.axvline(event.start, color=label_color, lw=1, alpha=0.7)
                    if trend and chart == "snapshot":
                        series = pd.Series(values[:, c], index=pd.DatetimeIndex(window.timestamps))
                        daily = series.resample("1D").median()
                        daily[series.resample("1D").count() < 12] = np.nan
                        positions = (
                            np.asarray((daily.index - window.timestamps[0]).total_seconds() / 3600)
                            + 12
                        )
                        chosen = (positions >= a) & (positions <= b)
                        ax.plot(
                            positions[chosen],
                            daily.to_numpy()[chosen],
                            color="#22645c",
                            lw=1.2,
                            marker=".",
                            markersize=2,
                        )
                    else:
                        ax.plot(np.arange(a, b + 1), values[a : b + 1, c], color="#365d72", lw=0.85)
                    if chart == "labels":
                        for event in truth.events:
                            if (
                                CHANNELS[c] in event.channels
                                and event.start <= b
                                and event.end >= a
                            ):
                                left, right = max(a, event.start), min(b, event.end)
                                ax.plot(
                                    np.arange(left, right + 1),
                                    values[left : right + 1, c],
                                    color=label_color,
                                    lw=1.7,
                                    zorder=3,
                                )
                                if left <= event.start <= right and np.isfinite(
                                    values[event.start, c]
                                ):
                                    ax.scatter(
                                        event.start,
                                        values[event.start, c],
                                        color=label_color,
                                        s=22,
                                        zorder=4,
                                    )
                    for left, right in segments(~np.isfinite(values[:, c])):
                        if left <= b and right > a:
                            ax.axvspan(
                                max(a, left) - 0.5,
                                min(b, right - 1) + 0.5,
                                color="#9baba9",
                                alpha=0.2,
                            )
                    for pred, color, low, high in (
                        (truth if chart == "case" else None, "#3f9b78", 0, 0.13),
                        (prediction, "#d48843", 0.87, 1),
                    ):
                        if pred:
                            for e in pred.events:
                                if CHANNELS[c] in e.channels and e.start <= b and e.end >= a:
                                    ax.axvspan(
                                        max(a, e.start) - 0.5,
                                        min(b, e.end) + 0.5,
                                        ymin=low,
                                        ymax=high,
                                        color=color,
                                        alpha=0.65,
                                    )
                    ax.set_ylabel(CHANNELS[c] + " (mm)")
                    ax.grid(alpha=0.14)
                    ax.set_xlim(a - 0.5, b + 0.5)
                ticks = np.unique(np.linspace(a, b, min(7, b - a + 1)).astype(int))
                axes[-1].set_xticks(
                    ticks,
                    [
                        f"{i}\n{window.timestamps[i].strftime('%Y-%m-%d' if b - a > 744 else '%m-%d %H:%M')}"
                        for i in ticks
                    ],
                )
                if chart != "labels":
                    axes[-1].set_xlabel(
                        "Day index / Beijing date (15:00)"
                        if is_daily
                        else "Global hour index / UTC"
                    )
                if is_daily:
                    axes[-1].set_xticklabels(
                        [
                            f"{i}\n{window.timestamps[i].astimezone(ZoneInfo('Asia/Shanghai')):%Y-%m-%d}"
                            for i in ticks
                        ]
                    )
                legend = [Patch(color="#9baba9", alpha=0.4, label="Unobserved")]
                if chart == "labels" and truth.events:
                    legend.insert(0, Patch(color=label_color, alpha=0.7, label="Injected target"))
                elif truth is not None and chart == "case":
                    legend.insert(0, Patch(color="#3f9b78", label="Truth (bottom band)"))
                if prediction is not None:
                    legend.insert(0, Patch(color="#d48843", label="Prediction (top band)"))
                axes[0].legend(
                    handles=legend,
                    loc="upper left",
                    bbox_to_anchor=(0, 1.35),
                    ncol=3,
                    frameon=False,
                    fontsize=9,
                )
                fig.subplots_adjust(
                    bottom=0.23 if chart == "labels" else 0.16, top=0.85, hspace=0.12
                )
                note = (
                    footer(detail)
                    if detail
                    else "Unlabelled platform observations | coordinate differences in mm"
                )
                note += f" | case {window.case_id[:16]} | {'days' if is_daily else 'hours'} {a}-{b}"
                if is_daily:
                    note += " | Exact Beijing 15:00; no aggregation or imputation"
                if chart == "labels":
                    note += " | Controlled injection labels only; native changes unlabelled"
                if trend:
                    note += " | UTC daily median (>=12 observations/channel/day); no imputation"
                if detail and detail.get("case_info"):
                    info = detail["case_info"]
                    fig.suptitle(
                        f"{LABELS.get(info['method'], info['method'])} / repeat {info['repeat'] + 1}",
                        x=0.91,
                        y=0.97,
                        ha="right",
                        fontsize=10,
                    )
            else:
                rows = selected_rows(detail, methods=methods)
                present = [m for m in METHODS if any(r["method"] == m for r in rows)]
                if chart == "ablation":
                    present = [m for m in present if m.startswith("lma")]
                fig, ax = plt.subplots(figsize=(10.4, 5.1))
                note = (
                    footer(detail)
                    + " | repeat means; SD bars where shown (not confidence intervals)"
                )
                if not present:
                    ax.text(
                        0.5,
                        0.5,
                        "No results for this selection",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                elif chart in ("overview", "ablation"):
                    fields = ["precision", "recall", "f1"] if chart == "overview" else [metric]
                    lookup = {r["method"]: r for r in rows}
                    width = 0.75 / len(fields)
                    x = np.arange(len(present))
                    for j, field in enumerate(fields):
                        scores = [lookup[m][field + "_mean"] for m in present]
                        errors = [lookup[m][field + "_std"] or 0 for m in present]
                        positions = x + (j - (len(fields) - 1) / 2) * width
                        ax.bar(
                            positions,
                            [v if v is not None else 0 for v in scores],
                            width,
                            yerr=errors,
                            capsize=2,
                            color=PALETTE[j],
                            label=METRICS[field],
                        )
                        for pos, v in zip(positions, scores):
                            if v is None:
                                ax.text(pos, 0.02, "N/A", rotation=90, ha="center", fontsize=8)
                    ax.set_xticks(x, [LABELS[m] for m in present], rotation=20, ha="right")
                    ax.set_ylim(0, 1.08)
                    ax.set_ylabel("Score")
                    ax.legend(frameon=False, ncol=3)
                    ax.set_title(
                        "Detection performance" if chart == "overview" else "LMA ablation",
                        loc="left",
                        pad=16,
                    )
                    ax.yaxis.grid(alpha=0.15)
                    ax.set_axisbelow(True)
                elif chart == "heatmap":
                    selected = selected_rows(detail, dimension, methods=present)
                    order = (
                        KINDS
                        if dimension == "kind"
                        else STATES
                        if dimension == "state"
                        else ["quality", "deformation", "undetermined", "normal"]
                    )
                    categories = [v for v in order if any(r["value"] == v for r in selected)]
                    matrix = np.full((len(present), len(categories)), np.nan)
                    for r in selected:
                        v = r[metric + "_mean"]
                        if v is not None:
                            matrix[present.index(r["method"]), categories.index(r["value"])] = v
                    cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#eff1ec")
                    im = ax.imshow(
                        np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto"
                    )
                    for i in range(len(present)):
                        for j in range(len(categories)):
                            v = matrix[i, j]
                            ax.text(
                                j,
                                i,
                                "N/A" if np.isnan(v) else f"{v:.3f}",
                                ha="center",
                                va="center",
                                color="white" if v > 0.6 else "#263b34",
                                fontsize=9,
                            )
                    ax.set_yticks(np.arange(len(present)), [LABELS[m] for m in present])
                    ax.set_xticks(np.arange(len(categories)), categories)
                    ax.set_title(
                        METRICS[metric] + " by " + dimension + " (N/A = undefined or unavailable)",
                        loc="left",
                        pad=16,
                    )
                    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
                elif chart == "missing":
                    selected = selected_rows(detail, "state", methods=present)
                    states = [s for s in STATES if any(r["value"] == s for r in selected)]
                    for i, m in enumerate(present):
                        lookup = {r["value"]: r for r in selected if r["method"] == m}
                        y = [lookup.get(s, {}).get(metric + "_mean") for s in states]
                        ax.plot(
                            np.arange(len(states)),
                            [np.nan if v is None else v for v in y],
                            "o-",
                            lw=1.4,
                            color=PALETTE[i],
                            label=LABELS[m],
                        )
                    ax.set_xticks(np.arange(len(states)), states)
                    ax.set_ylim(-0.02, 1.05)
                    ax.set_ylabel(METRICS[metric])
                    ax.grid(alpha=0.15)
                    ax.set_title("Missing-data conditions (categorical)", loc="left", pad=16)
                    ax.legend(frameon=False, fontsize=9, ncol=3)
                else:
                    for i, r in enumerate(rows):
                        if r["f1_mean"] is not None:
                            ax.scatter(
                                r["mean_seconds_mean"],
                                r["f1_mean"],
                                s=55 + 20 * r["mean_model_requests_mean"],
                                color=PALETTE[i % len(PALETTE)],
                            )
                            ax.annotate(
                                LABELS[r["method"]],
                                (r["mean_seconds_mean"], r["f1_mean"]),
                                xytext=(6, 6),
                                textcoords="offset points",
                                fontsize=9,
                            )
                    ax.set_xlabel("Mean wall time per case (seconds)")
                    ax.set_ylabel("F1")
                    ax.set_ylim(-0.03, 1.08)
                    ax.margins(x=0.25)
                    ax.grid(alpha=0.15)
                    ax.set_title(
                        "Performance and runtime (marker size: model requests)", loc="left", pad=16
                    )
                fig.tight_layout(rect=(0, 0.09, 1, 1))
            return save(fig, fmt, note)
        finally:
            if fig is not None:
                plt.close(fig)


def summary_csv(detail):
    rows = detail["summary"]["summary"]
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def paper_bundle(detail):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metrics.csv", summary_csv(detail))
        archive.writestr(
            "provenance.json",
            json.dumps(
                {
                    "meta": detail["meta"],
                    "notices": detail["notices"],
                    "summary": detail["summary"],
                    "error_bars": "population SD across model repeats, NOT confidence intervals",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        for chart in CHARTS:
            if chart == "ablation" and not any(
                r["method"].startswith("lma") for r in detail["summary"]["summary"]
            ):
                continue
            for fmt in FORMATS:
                archive.writestr(f"{chart}.{fmt}", figure_bytes(detail, chart, fmt))
    return output.getvalue()
