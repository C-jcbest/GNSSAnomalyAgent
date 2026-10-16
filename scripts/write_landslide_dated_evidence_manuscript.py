"""Add the completed chronology feasibility experiment to the complete v8 draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "dated-evidence-probe-v1"
    verified = base / "dated-evidence-replay-v1"
    for folder in (run, verified):
        verify_manifest(folder, "freeze.json")
    if read_json(verified / "verification.json")["status"] != "VERIFIED":
        raise ValueError("Require verified probe results")
    summary = read_json(run / "summary.json")
    decision = read_json(run / "decision.json")
    methods = (
        "dated_numeric", "permuted_dated_numeric", "pooled_dated_numeric",
        "flat_dated_numeric", "dated_visual_numeric", "query_motion_mass",
        "global_motion_mass", "motion_numeric_mlp", "raw_visual_motion",
        "embedding_motion_xgb", "raw_motion_information_xgb", "pure_visual",
        "fixed_blend", "robust_rule", "pelt_rule",
    )
    output = base / "dated-evidence-report-v1"
    output.mkdir(exist_ok=False)
    table = score_table(summary, methods)
    indexed = {row["method"]: row for row in summary}
    figure, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for axis, metric in zip(axes, ("f1", "event_f1_iou03")):
        axis.barh(
            methods, [indexed[name][metric] for name in methods],
            color=["#ab5144" if name in ("dated_numeric", "dated_visual_numeric") else "#548097" for name in methods],
        )
        axis.set_xlim(0, 1)
        axis.set_title(metric)
    axes[0].invert_yaxis()
    figure.suptitle("Date-ordered evidence probe:48TRAIN/24exposedVALIDATION")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=140)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)
    report = (
        "# 日期排列证据利用：完整开发负结果\n\n"
        "这是强数值时序读取及其普通视觉融合的依据检验，不是已成立的新Agent方法。"
        "原13块、12原相邻变化和39运动数值不变，缺测不跨接；打乱控制保留两条目标相邻变化。"
        "同10187TRAIN合法中心、12epoch、五臂190配置、原29方法保留，共34方法。\n\n"
        + table
        + "\n\n目标读取.773789低于打乱.788170、全窗池化.789805、展平.790869"
        "及全局分布树.837166；视觉拼接.782069低于原拼接.805651。"
        "精确值以绑定summary.json为准。数值与视觉预设判据均0/4，"
        + decision["numerical_status"] + "／" + decision["visual_status"]
        + "。不能推断日期顺序普遍无用，也不能反推此前图坐标一定是失败原因；"
        "仅否定这一固定小网络与输入实现，不搜索新宽度／epoch／seed救分。\n\n"
        "72缓存记录、TRAIN边变化缩放、5选中模型87600概率、34方法816输出及全部统一评分精确重放。"
        "没有重训10个分类器、没有新图／编码／API或新独立记录；保留24内容未加载。"
        "中心／打乱／池化的参数量逐宽完全一致，展平和视觉组的额外参数另存。"
        "实际测试两条缺测／排列业务契约，图表由冻结评分生成，不构造新的观测参考。\n"
    )
    # Derive the numerical wording from saved scores rather than hand rounding.
    report = report.replace(
        "目标读取.773789低于打乱.788170、全窗池化.789805、展平.790869",
        f"目标读取{indexed['dated_numeric']['f1']:.6f}低于打乱{indexed['permuted_dated_numeric']['f1']:.6f}、全窗池化{indexed['pooled_dated_numeric']['f1']:.6f}、展平{indexed['flat_dated_numeric']['f1']:.6f}",
    ).replace("视觉拼接.782069", f"视觉拼接{indexed['dated_visual_numeric']['f1']:.6f}")
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(output / "evidence-bindings.json", {
        str(p.resolve()): sha(p) for p in (run / "freeze.json", verified / "freeze.json")
    })
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "freeze.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    text = (root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v8.md").read_text(encoding="utf-8")
    text = text.replace("2026-10-13，v8。", "2026-10-13，v9。", 1)
    text = text.replace(
        "本版补数值证据条件化视觉残差（NCVR）的完整48／24开发实验、强对照、消融与负结果。",
        "本版保留NCVR完整结果，补日期排列证据利用探针的五臂190配置、强对照与普通视觉融合负结果。",
        1,
    )
    text = text.replace("deformation-manuscript-20261013-v8", "deformation-manuscript-20261013-v9", 1)
    text = text.replace(
        "训练期望、选中模型概率及29方法统一输出均已精确重放。",
        "随后检验原日期运动块读取：目标读取日F1为0.7738，低于展平0.7909与全局分布树0.8372；加入视觉为0.7821，也未改善。两版训练期望／缩放、选中模型概率及统一输出均已精确重放。",
        1,
    )
    text = text.replace(
        "The version is stopped. The expectation fits, selected probabilities, and all 29 methods are exactly replayed.",
        "The version is stopped. A subsequent chronology probe achieves daily F1 of 0.7738 for target-adjacent readout, below flattened readout at 0.7909 and the global distribution tree at 0.8372; adding cached visual embeddings yields 0.7821. The representation fits/scaling, selected probabilities, and all unified outputs are exactly replayed.",
        1,
    )
    method = r"""### 3.13 原日期运动证据利用探针

为先核实定位证据的利用能力，采用普通两层一维卷积读取原13块的12条变化，卷积核3、膨胀率1／3，不生成新位移。原相邻两块都可用时才定义三轴变化；仅用合法TRAIN中心拟合逐位置缩放及不可用指示，每边6通道。$\mathbf e_{t,k}$ 包含原日期边变化及缺测flags，令普通卷积表示为 $\mathbf H_t$，目标读取为：

$$
\mathbf H_t=\operatorname{TCN}_\theta(\mathbf e_{t,0},\ldots,\mathbf e_{t,11}),
\qquad
\mathbf q_t=[\mathbf H_{t,5};\mathbf H_{t,6}],
\qquad
p_t=\sigma(h_\phi([\mathbf q_t;\mathbf u_t])).
$$

两条目标边保持在原位置；打乱控制逐查询重新排列其余十条已计算变化与对应flags，不在排列后重新差分，避免制造假导数。全窗池化使用相同卷积和参数量；展平读取拿到同完整边信息；普通视觉融合再拼接原512编码。该探针不称TCN、拼接或日期位置为本项目创新，没有视觉token精确对齐或新语言推理。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    result = "### 5.15 日期排列证据利用：未支持定位机制扩展\n\n" + report.split("\n\n", 1)[1] + "\n\n"
    text = text.replace("## 6 讨论", result + "## 6 讨论", 1)
    text = text.replace(
        "后续先检验局部日期证据能否改善识别及事件边界，再考虑一个机制，避免继续围绕同一全局向量增添归一化和融合模块。",
        "随后完成的日期排列探针也没有支持当前目标读取机制：同容量池化／打乱及展平更高，加入视觉仍低于强数值。它并不证明定位证据本身无价值；只有这一固定模型和输入被否定。应优先回到仍有受控收益的方法检验适用条件，或在真实可纠正错误上获得新观察证据，避免继续增加未经支持的表示模块。",
        1,
    )
    text = text.replace(
        "## 数据、代码与AI使用说明",
        "日期排列探针进一步通过0/4数值与0/4视觉条件，停止固定小网络；原数值分布的信息收益保留，但稳定Agent/VLM贡献仍未建立。\n\n## 数据、代码与AI使用说明",
        1,
    )
    text = text.replace(
        "Codex参与文献定位",
        "日期证据探针由probe/verify_landslide_dated_evidence.py及landslide_dated_evidence.py复现，10训练模型、190配置、87600选中概率与816统一输出保存于dated-evidence-probe/replay/report-v1。没有新图编码／API或保留样本曝光。\n\nCodex参与文献定位",
        1,
    )
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v9.md"
    if target.exists():
        raise FileExistsError("Preserve the frozen draft")
    target.write_text(text, encoding="utf-8")
    print(str(target), flush=True)


if __name__ == "__main__":
    write()
