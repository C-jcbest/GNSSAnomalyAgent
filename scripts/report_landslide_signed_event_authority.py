"""Report exact paired changes and candidate-event failures without relabeling."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs


def report(run, replay, out):
    verify_manifest(run, "freeze.json")
    verify_manifest(replay, "freeze.json")
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["probe_manifest_sha256"] != sha(run / "freeze.json"):
        raise ValueError("Matching exact replay required")
    protocol = read_json(run / "protocol.json")
    upstream = read_json(Path(protocol["source"]) / "protocol.json")
    predictions = read_json(run / "predictions.json")
    details = read_json(run / "event-authority.json")
    rows = {row["method"]: row for row in read_json(run / "summary.json")}
    out.mkdir(parents=True, exist_ok=False)
    totals, paired, traces = [], [], {}
    for rep in range(2):
        main = f"sera_signed_event_r{rep}"
        controls = (f"visual_daily_partial_r{rep}", f"sera_whole_event_r{rep}",
                    f"sera_signed_no_event_r{rep}", f"sera_signed_center_r{rep}",
                    f"sera_signed_no_local_r{rep}", f"visual_neighborhood_partial_r{rep}")
        for cid in upstream["ids"]:
            with np.load(Path(upstream["encoded"]) / "validation" / (cid + ".npz")) as bank:
                actual = bank["activity"]
            after = np.array(predictions[main][cid]["activity"])
            for method in controls:
                before = np.array(predictions[method][cid]["activity"])
                changes = {
                    "removed_fp": (actual == 0) & (before == 1) & (after != 1),
                    "added_fp": (actual == 0) & (before != 1) & (after == 1),
                    "gained_tp": (actual == 1) & (before != 1) & (after == 1),
                    "lost_tp": (actual == 1) & (before == 1) & (after != 1),
                }
                row = dict(method=main, comparator=method, case_id=cid,
                           **{name: int(mask.sum()) for name, mask in changes.items()})
                row["net_correct"] = row["removed_fp"] + row["gained_tp"] - row["added_fp"] - row["lost_tp"]
                paired.append(row)
                traces[f"{main}:{method}:{cid}"] = {name: runs(mask) for name, mask in changes.items()}
        for method in controls:
            subset = [row for row in paired if row["method"] == main and row["comparator"] == method]
            total = dict(method=main, comparator=method,
                         **{name: sum(row[name] for row in subset)
                            for name in ("removed_fp", "added_fp", "gained_tp", "lost_tp", "net_correct")})
            expected = rows[method]["fp"] + rows[method]["fn"] - rows[main]["fp"] - rows[main]["fn"]
            if total["net_correct"] != expected:
                raise ValueError("Paired corrections disagree with score")
            totals.append(total)
    equality = {}
    additions = {}
    for rep in range(2):
        main = predictions[f"sera_signed_event_r{rep}"]
        equality[str(rep)] = {
            name: main == predictions[f"sera_{name}_r{rep}"] for name in ("signed_center", "whole_event")
        }
        additions[str(rep)] = {
            name: sum(record[name] for key, record in details.items() if key.endswith(f"-r{rep}"))
            for name in ("proposed_addition_days", "seeded_event_days", "supported_addition_days")
        }
    counts = Counter(event["reason"] for record in details.values() for event in record["event_checks"])
    diagnostics = dict(reasons=dict(counts), additions=additions, exact_decoded_control_equality=equality)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "exact-error-traces.json", dict(interval_convention="Original zero-based half-open", traces=traces))
    write_csv(out / "paired-errors.csv", paired)
    write_csv(out / "correction-summary.csv", totals)
    candidates = sorted(
        [row for row in paired if row["method"] == "sera_signed_event_r0" and row["comparator"] == "visual_daily_partial_r0"],
        key=lambda row: (row["net_correct"], row["case_id"]),
    )
    review_ids = list(dict.fromkeys([candidates[0]["case_id"], candidates[-1]["case_id"]]))
    write_json(out / "analysis-plan.json", dict(
        review_ids=review_ids, selection="Repeat0 minimum/maximum net correction versus daily authority;posthoc",
        reference_changes=0,
    ))
    for cid in review_ids:
        case, _ = load_case_features(Path(upstream["data"]), cid)
        raw = np.array([value if value is not None else [np.nan] * 3 for value in case.displacement_mm])
        fig, panels = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        for axis, name in enumerate(("N", "E", "U")):
            panels[axis].plot(raw[:, axis], lw=.8)
            panels[axis].set_ylabel(name + " (mm)")
            panels[axis].grid(alpha=.2)
        with np.load(Path(upstream["encoded"]) / "validation" / (cid + ".npz")) as bank:
            actual = bank["activity"]
        for level, (label, values) in enumerate((
            ("Process reference", actual),
            ("RCRA daily", predictions["visual_daily_partial_r0"][cid]["activity"]),
            ("SERA event", predictions["sera_signed_event_r0"][cid]["activity"]),
        )):
            for start, stop in runs(np.array(values) == 1):
                panels[3].plot([start, stop - 1], [level, level], lw=6)
        panels[3].set_yticks([0, 1, 2], ["Process reference", "RCRA daily", "SERA event"])
        panels[3].set_xlabel("Original day index; diagnostic only, labels not sent to VLM")
        panels[3].set_xlim(0, len(raw) - 1)
        fig.suptitle(cid + " | Posthoc developer diagnostic")
        fig.tight_layout()
        fig.savefig(out / (cid + "-diagnostic.png"), dpi=140)
        plt.close(fig)
    methods = ["query_motion_mass"]
    for rep in range(2):
        methods += [f"{name}_r{rep}" for name in (
            "direct_visual", "visual_whole_partial", "visual_neighborhood_partial", "visual_daily_partial",
            "sera_signed_event", "sera_whole_event", "sera_signed_no_event", "sera_signed_center",
            "sera_signed_strict", "sera_signed_no_local",
        )]
    text = (
        "# SERA方向分离事件授权：完整开发负结果\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent。\n"
        "- Scope: 同48TRAIN／24已暴露VALIDATION、两遍缓存回答；0新API、0新记录，保留24内容未加载。\n"
        "- Verification Status: VERIFIED，仅指确定性重放，不表示效果通过。\n"
        "- Reference: 受控过程参考与开发者辅助诊断；无独立专家标签。\n\n"
        "SERA（Signed Event Revision Authority）组合原日冲突撤回、局部视觉阳性事件的有序累计运动证据授权新增。"
        "事件证据、partial和解码均复用已有实现；新检验仅是方向分离与局部候选事件归属。\n\n"
        + score_table(list(rows.values()), methods)
        + "\n\n固定候选状态为 **STOP_SIGNED_EVENT_AUTHORITY**。两遍主F1下降，相对RCRA净损82／84日。"
        "主输出与同新增日数中心、全窗撤回控制逐字段一致，不能以非零新增授权冒称新增位置选择有效。"
        "不调整本版事件长度、排名门槛或提示救分，不扩本版新数据。\n\n"
        "事件整体累计运动证据只能支持期间发生过运动，不能确定其中每一天的持续活动边界。"
        "精确增删日期及按预定极值选出的原图另存；解释以实际打开图像为准。"
        "两遍回答及逐日相关数据不视为独立样本，不报告显著性或现场泛化。\n\n"
        "下一步应检验新增观察内容的定位价值，暂停继续排列权限范围。优先检验原位移与"
        "原轴有符号变化的日期对应，与同值打乱日期和普通局部图匹配比较；这是待验证假设。\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(dict(diagnostics=diagnostics, paired_totals=totals, review_ids=review_ids), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.replay.resolve(), arguments.out.resolve())
