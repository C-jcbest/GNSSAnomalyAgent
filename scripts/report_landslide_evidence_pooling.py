"""Save exact pooling metrics, physical-guide examples and paired errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_evidence_pooling import MAIN
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores


def report(data, encoded, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(run / "freeze.json"):
        raise ValueError("A matching selected-model replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    results = read_json(run / "results.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    banks = load_banks(encoded, protocol["split"])
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in protocol["split"]["validation"]
    }
    paired = []
    for cid, family in families.items():
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results["uniform_pool_numeric"]["per_case"][cid]["daily"]
        paired.append(dict(
            case_id=cid, family=family, main_fp=main["fp"], main_fn=main["fn"],
            uniform_fp=control["fp"], uniform_fn=control["fn"],
            net_correct=control["fp"] + control["fn"] - main["fp"] - main["fn"],
        ))
    write_csv(out / "paired-case-errors.csv", paired)
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores([
                results[name]["per_case"][cid] for cid in families if families[cid] == family
            ])
            family_rows.append(dict(method=name, family=family, **score["daily"],
                                    event_f1=score["events"]["f1"]))
    write_csv(out / "family-summary.csv", family_rows)
    diagnostics = []
    for group, ids in protocol["split"].items():
        for cid in ids:
            bank = banks[cid]
            selected = bank["selected"]
            visual = bank["embeddings"][selected].astype(float)
            for kind in ("continuity", "permuted", "jump"):
                guided = bank[kind + "_embedding"][selected].astype(float)
                weights = bank[kind + "_weights"][selected]
                cosine = np.sum(visual * guided, axis=1)
                diagnostics.append(dict(
                    group=group, case_id=cid, kind=kind, queries=int(selected.sum()),
                    median_cosine_to_native=float(np.median(cosine)),
                    median_max_token_weight=float(np.median(weights.max(axis=(1, 2)))),
                    median_effective_tokens=float(np.median(1 / np.sum(weights**2, axis=(1, 2)))),
                ))
    write_csv(out / "pooling-diagnostics.csv", diagnostics)
    methods = (
        MAIN, "uniform_pool_numeric", "permuted_pool_numeric", "jump_pool_numeric",
        "continuity_pool_no_branch", "uniform_pool_visual", "guide_numeric_mlp",
        "guide_information_xgb", "all_guide_information_xgb", "pilot_global_mass_xgb",
        "pilot_query_mass_xgb", "global_numeric", "numeric", "xgb_raw84",
        "xgb_chart384", "robust_rule", "pelt_rule",
    )
    indexed = {row["method"]: row for row in summary}
    figure, panels = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    for panel, metric in zip(panels, ("f1", "event_f1_iou03")):
        panel.barh(methods, [indexed[name][metric] for name in methods],
                   color=["#ad5146" if name == MAIN else "#547f99" for name in methods])
        panel.set_title(metric)
        panel.set_xlim(0, 1)
    panels[0].invert_yaxis()
    figure.suptitle("Continuity-guided visual pooling:16TRAIN/8exposedVALIDATION")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=140)
    figure.savefig(out / "comparison.pdf")
    plt.close(figure)
    for group in ("train", "validation"):
        example = read_json(encoded / (group + "-example-weights.json"))
        original = plt.imread(encoded / (group + "-example.png"))
        figure, panels = plt.subplots(1, 4, figsize=(13, 3.5))
        panels[0].imshow(original)
        panels[0].axis("off")
        panels[0].set_title("Unchanged input chart")
        maximum = max(np.max(example[kind]) for kind in ("continuity", "permuted", "jump"))
        for panel, kind in zip(panels[1:], ("continuity", "permuted", "jump")):
            panel.imshow(example[kind], cmap="magma", vmin=0, vmax=maximum)
            panel.set_title(kind + " readout weights")
            panel.set_xticks([])
            panel.set_yticks([])
        figure.suptitle(f"{example['case_id']} day{example['day']};coarse weights,not causal attribution")
        figure.tight_layout()
        figure.savefig(out / (group + "-guide-example.png"), dpi=140)
        figure.savefig(out / (group + "-guide-example.pdf"))
        plt.close(figure)
    review_ids = list(dict.fromkeys([
        min(paired, key=lambda row: row["net_correct"])["case_id"],
        max(paired, key=lambda row: row["net_correct"])["case_id"],
    ]))
    selected_methods = (
        MAIN, "uniform_pool_numeric", "permuted_pool_numeric",
        "jump_pool_numeric", "pilot_global_mass_xgb", "guide_information_xgb",
    )
    for cid in review_ids:
        case, _ = load_case_features(data, cid)
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=.8)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        for row, name in enumerate(selected_methods):
            states = np.array(predictions[name][cid]["activity"], dtype=float)
            states[states < 0] = np.nan
            panels[3].plot(np.arange(len(states)), row + states * .7, linewidth=1)
        panels[3].set_yticks(np.arange(len(selected_methods)))
        panels[3].set_yticklabels(selected_methods, fontsize=8)
        panels[3].set_xlabel("Original day;post-score review;unknown/missing never filled")
        figure.suptitle(cid)
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        figure.savefig(out / (cid + "-raw-review.pdf"))
        plt.close(figure)
    write_json(out / "analysis-plan.json", dict(
        review_ids=review_ids, selection="First worst/best known-day correction vs uniform;post-score",
        independent_reference=False, reserved_content_loaded=False,
    ))
    table = score_table(summary, tuple(row["method"] for row in sorted(summary, key=lambda row: row["f1"], reverse=True)))
    cost = read_json(encoded / "encoding-cost.json")
    checks = "\n".join(f"- {name}: {'成立' if passed else '不成立'}。" for name, passed in decision["checks"].items())
    text = f"""# SCVP：连续位移证据引导视觉池化的完整开发小试

同原16TRAIN／8暴露VALIDATION、三轴原图／预训练骨干／头／阶段协议。改变的是8x8最终特征图的读取权重，由相邻同向原14日块变化的调和量确定，保留均匀底权重1。最终token已混合全局上下文，不能将它解释为孤立日期的因果注意力。

七神经／两强树各38机会，342新配置，加原21方法共30统一方法。四主要融合臂均1198维，原global512、辅助pool512及共同87数值／flags174；每个数值对照都得到连续与跳变guide48值。去显式数值分支仍使用数值引导池化，不能称纯视觉；uniform_pool_visual才去掉全部数值反馈。两树87／505维，后者还读取完整原图描述符和原块profile；原分布强树及规则、PELT、普通视觉控制保留。训练规模与48／24不可直接横比。

{table}

六预设条件{sum(decision["checks"].values())}/6，**{decision["status"]}**。对均匀／打乱／跳变净纠错{decision["net_correct"]["uniform_pool_numeric"]}/{decision["net_correct"]["permuted_pool_numeric"]}/{decision["net_correct"]["jump_pool_numeric"]}日；最强数值{decision["best_nonvisual"]}。不将对照更名为创新，不调整floor／位置／epoch／seed救分。

{checks}

24原记录、连续／跳变guide及三权重、TRAIN缩放逐值重构；{receipt["sampled_auxiliary_encodings"]}辅助编码抽样最大误差{receipt["maximum_auxiliary_encoding_error"]:.8g}；9选中模型52560概率、30方法240输出及统一活动／事件／边界／阶段评分精确重放。没有重训18分类模型或重编码全部输入。原图{cost["original_image_forwards"]}次前向、三辅助pool共{cost["auxiliary_pool_vectors"]}向量，准备总耗时{cost["seconds"]:.3f}秒；它包括消融准备，不称端到端部署成本。主推理只需同一原图的一个骨干前向、两池化投影和活动头，无新语言请求。

pooling-diagnostics.csv只描述表征与权重变化，不是识别增益或因果证明。事后原图{', '.join(review_ids)}的实际Codex复核另存session；8验证记录每族仅1条，不作独立显著性结论。没有新记录／独立参考，保留24内容未加载。

MedTVL已研究文本条件化双路径与实例级融合，VESTA已研究数值工具支持的视觉模型修订；本项目不称条件化／池化／多模态通用首创，也不引入复杂MoE或动态多工具。官方原文、固定作者源码、实际阅读边界另存，作者模型未运行。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(out / "evidence-bindings.json", {
        str(p.resolve()): sha(p) for p in (encoded / "encoding-freeze.json", run / "freeze.json", verified / "freeze.json")
    })
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(dict(status=decision["status"], checks=f"{sum(decision['checks'].values())}/6",
               review_ids=review_ids), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.data.resolve(), args.encoded.resolve(), args.run.resolve(),
           args.verified.resolve(), args.out.resolve())
