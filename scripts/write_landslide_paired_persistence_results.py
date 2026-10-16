"""Preserve v52 before adding the actual persistence-pair calibration outcome."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json
from write_landslide_claim_review_results import DOCUMENTS

from gnss_sim.artifacts import sha


def historical_checks(archive):
    manifest = read_json(archive / "v52-delivery-manifest-original.json")
    checks = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if name in {"docs/" + value + ".md" for value in DOCUMENTS}:
            path = archive / (Path(name).stem + "-v52-original.md")
        if sha(path) != expected:
            raise ValueError("Historical delivered bytes changed: " + name)
        checks.append({"original_path": name, "preserved_path": path.relative_to(ROOT).as_posix(), "sha256": expected})
    return checks


def write():
    run = BASE / "paired-persistence-calibration-v1"
    analysis = BASE / "paired-persistence-analysis-v1"
    development = BASE / "paired-persistence-development-v1"
    development_analysis = BASE / "paired-persistence-development-analysis-v1"
    verify_bindings(run, "calibration-freeze.json")
    verify_bindings(development, "result-freeze.json")
    for filename, expected in read_json(development_analysis / "manifest.json")["files"].items():
        if sha(development_analysis / filename) != expected:
            raise ValueError("Development report changed: " + filename)
    decision = read_json(run / "calibration-decision.json")
    development_decision = read_json(development / "decision.json")
    checks = read_json(analysis / "verification.json")
    if checks["weight_predictions"] != "exact" or checks["common_scores"] != "exact":
        raise ValueError("Deliver only independently replayed frozen predictions")
    archive = BASE / "manuscript-v53"
    archive.mkdir(exist_ok=False)
    for name in DOCUMENTS:
        (archive / (name + "-v52-original.md")).write_bytes((ROOT / "docs" / (name + ".md")).read_bytes())
    (archive / "v52-delivery-manifest-original.json").write_bytes((BASE / "manuscript-v52/delivery-manifest.json").read_bytes())
    export(archive / "v52-preservation-before.json", historical_checks(archive))
    summary = read_json(run / "validation-summary.json")
    lookup = {row["method"]: row for row in summary}
    main = lookup["visual_numeric_pair"]
    augmentation = lookup["visual_numeric_augmentation"]
    plain = lookup["visual_numeric_plain"]
    numeric = lookup["numeric_pair"]
    corrections = read_json(run / "corrections.json")
    net = {name: sum(row["net"] for row in corrections if row["control"] == name) for name in lookup}
    encoding = read_json(run / "encoding-cost.json")
    training = read_json(run / "training-cost.json")
    outcome = (f"主F1 {main['f1']:.6f}，相对普通视觉数值头{main['f1'] - plain['f1']:+.6f}，"
               f"相对同数据无配对损失{main['f1'] - augmentation['f1']:+.6f}，"
               f"相对纯数值配对{main['f1'] - numeric['f1']:+.6f}；"
               f"净确定纠错对plain/augmentation为{net['visual_numeric_plain']}/{net['visual_numeric_augmentation']}，"
               f"弱R {main['weak_recall']:.6f}。{decision['passed']}/{decision['total']}条件，{decision['decision']}。")
    development_summary = read_json(development_analysis / "summary.json")
    development_lookup = {row["method"]: row for row in development_summary}
    development_main = development_lookup["visual_numeric_pair"]
    development_outcome = (f"完整24条暴露开发主F1 {development_main['f1']:.6f}，"
        f"相对普通头{development_main['f1'] - development_lookup['visual_numeric_plain']['f1']:+.6f}、"
        f"相对增广{development_main['f1'] - development_lookup['visual_numeric_augmentation']['f1']:+.6f}、"
        f"相对纯数值配对{development_main['f1'] - development_lookup['numeric_pair']['f1']:+.6f}；"
        f"弱R {development_main['weak_recall']:.6f}。{development_decision['passed']}/{development_decision['total']}条件，"
        f"{development_decision['decision']}。")
    bootstrap = read_json(development_analysis / "descriptive-record-bootstrap.json")
    pair_interval = bootstrap["visual_numeric_augmentation"]["exploratory_95_percentile_interval"]
    development_limit = (f"主事件F1 {development_main['event_f1_iou03']:.6f}，"
        f"低于增广的{development_lookup['visual_numeric_augmentation']['event_f1_iou03']:.6f}；"
        f"相对增广的记录描述性95%区间[{pair_interval[0]:.6f},{pair_interval[1]:.6f}]跨零。"
        "日F1收益不能解释为事件识别全面改善或统计显著，独立新数据确认仍必要。")
    development_cost = read_json(development / "inference-cost.json")
    development_cost_text = (
        f"完整开发共享绘图/编码{sum(row['encoded_images'] for row in development_cost['rows'])}图，"
        f"耗时{sum(row['render_encoder_seconds'] for row in development_cost['rows']):.3f}秒；"
        f"数值输入提取{sum(row['feature_seconds'] for row in development_cost['rows']):.3f}秒，"
        f"主小头合计{sum(row['heads']['visual_numeric_pair'] for row in development_cost['rows']):.6f}秒；"
        f"含载入与十头推断的本地实测总时间{development_cost['wall_seconds']:.3f}秒。"
        "编码在各视觉臂间共享，不能将小头时间当完整部署成本；这些CPU时间不直接等价于远端VLM延迟。"
    )
    development_review = (
        "开发评分后Codex实际看四幅极值图并核对原数值：case_0183单阶跃后平台，主新增25个FP而增广/专家均0；"
        "case_0169弱活动检出124/212对增广98/212，FP均0；case_0178强运动后弱续动，主和增广均0/196，"
        "expert24检出65/196，长缺测保持未知；case_0188主弱检189/208对增广178/208，FP均4。"
        "观察解释同时保留收益、平台误报和弱续动漏检，不回写参考、不把带参考的极值图称盲标。"
        "详见paired-persistence-development-error-review-v1/actual-review.md与review-evidence.json。"
    )
    stop = development_decision["decision"].startswith("STOP_")
    next_action = ("停止此固定候选，不换seed、阈值、宽度、margin、画法或最佳消融救分；"
                   "继续寻找具有不同识别依据的方法。" if stop else
                   "校准和完整开发门槛已通过，参数继续冻结；下一步是一次新完整记录确认，不能提前称独立贡献成立。")
    selected_names = ["visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain",
                      "visual_numeric_shuffled_pairs", "visual_pair_no_numeric", "numeric_pair", "numeric_augmentation",
                      "xgb_raw18", "xgb_expert24", "xgb_chart384"]
    positions = np.arange(len(selected_names))
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(positions - .18, [lookup[name]["f1"] for name in selected_names], width=.36, label="Daily F1")
    axes[0].bar(positions + .18, [lookup[name]["event_f1_iou03"] for name in selected_names], width=.36, label="Event F1 IoU.3")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title("Original 12-record calibration, same augmented negatives | not independent confirmation")
    axes[1].bar(positions, [lookup[name]["weak_recall"] for name in selected_names])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Weak-motion recall")
    axes[1].set_xticks(positions, selected_names, rotation=30, ha="right", fontsize=8)
    figure.tight_layout()
    figure.savefig(archive / "paired-persistence-comparison.png", dpi=140)
    plt.close(figure)
    formula_section = r"""训练配对使用受控训练原观察 $x_t=m_t+e_t$，端点 $u=c-91$、$v=c+91$，仅把已知训练运动分量替换成同净位移单次偏移：

$$
\widetilde{x}_t=x_t-m_t+m_u+\mathbf{1}_{t\geq k}(m_v-m_u),\qquad k\in\{c-42,c,c+42\}.
$$

端点原观测直接复制，因而 $\widetilde{x}_u=x_u$、$\widetilde{x}_v=x_v$；原误差和缺测保留。$m_t$ 是训练专用合成latent，验证与部署不访问。对照的中心目标为静稳，因为仪器偏移本身不构成持续位移活动。对照不是同一固定输入换“正常规则”；不直接移植Anlu[41]的reference条件标签。

用冻结MobileCLIP2-S0/dfndr2b视觉分支[42]编码256像素三轴图，保留61个三日原槽和缺槽掩码、中心标记；使用精确预训练预处理、512维L2归一化编码，不使用语言生成。

$$
h_c=\operatorname{normalize}\bigl(E_{\mathrm{frozen}}(I_c)\bigr),\qquad
z_c=f_\theta([h_c,n_c]).
$$

数值输入为已有18项raw84及18缺值指示，统一用原抽样训练统计归一化。头仅16或32隐单元。BCE采用原正负训练量平衡；配对项在每个监督minibatch上取全部137对的均值：

$$
\mathcal{L}_{\mathrm{pair}}=\frac{1}{|\mathcal P|}\sum_{(c^+,c^-)\in\mathcal P}
\log\bigl(1+\exp(1-z_{c^+}+z_{c^-})\bigr),\qquad
\mathcal L=\mathcal L_{\mathrm{BCE}}+\mathcal L_{\mathrm{pair}}.
$$

去配对损失保留完全相同增广；打乱配对只改变负样本对应关系；去数值、纯数值配对和纯数值增广检验模态作用。普通头仅原样本。新增训练数据与配对作用分别检验，不能将增广收益都归于视觉。CAPS[43]固定源码已有matched/shuffled对照及BCE加softplus排序项，配对、排序损失和打乱消融不称新思想。可检验差异限于GNSS同误差同净位移持续性负例与冻结视觉支撑，须由识别收益证明。
"""
    columns = ("method", "precision", "recall", "f1", "event_f1_iou03", "event_f1_iou05",
               "weak_recall", "onset_core_mae_days", "offset_core_mae_days", "coverage")
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summary:
        table.append("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key]) for key in columns) + " |")
    stage = read_json(run / "results.json")["visual_numeric_pair"]["internal_stages"]
    text = ("# 同净位移持续运动／单次偏移成对监督：预训练视觉校准\n\n"
            "## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n"
            "- Mode: run / validate inline；2026-10-11\n- Version: paired-persistence-calibration-v1 / manuscript-v53\n"
            "- Verification: 冻结权重、概率、原日预测与统一评分重放；非独立现场结论\n\n" + outcome + "\n\n" +
            development_outcome + "\n\n" + development_limit + "\n\n" +
            next_action + "\n\n" + formula_section + "\n\n"
            "原24训练/12验证划分保持，7656原抽样中心、137新增负中心来自16个训练parent，其中48弱中心；"
            "不计新独立记录。latent只在构造训练对时读取，分类输入只包含观测图和数值特征。"
            "12epoch、seed20261020、Adam.001、wd.0001、CPU2线程，各臂2头宽×19阈值=38校准机会。"
            "XGB18、expert24、384图信息对照得到同样137负中心及全部已知支撑原训练日，各2深度×19阈值；"
            "原规则、PELT、12项XGB使用同划分已校准模型，只作既有方法对照并报告覆盖，不冒充同增广强基线。"
            "普通VLM/固定Agent的完整开发对照不在这12验证校准中，已按通过校准后的冻结计划完成24条对照；此前完整比较仍保留。\n\n" +
            "\n".join(table) + "\n\n"
            f"共享数值阶段头，内部A/S/D支持{stage['support_ASD']}，主内部宏F1 {stage['macro_f1']:.6f}；"
            "所有臂内部及端到端阶段、事件.3/.5、逐记录和边界完整保存在results.json，"
            "没有新增原生视觉阶段头，不把活动门控带来的阶段变化归于视觉阶段推理。\n\n"
            "## 完整24条开发对照\n\n" + development_outcome + "\n\n" + development_limit + "\n\n"
            "十个新方法与原37方法共47方法，在同一24条完整过程参考上评分；没有据结果改宽度、阈值、数据或画法。"
            "新预测全部冻结后评分。原37方法的日、事件、边界、阶段分数逐值复现，VLM部署调用/token从原回执支持的汇总保留。"
            "重放0新API不冒充原VLM无部署代价。完整47方法与原日逐记录、净纠错、记录bootstrap和四幅极值诊断图见"
            "[完整开发报告](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-development-analysis-v1/report.md)。"
            "该24条已被此前研究使用，不能改称本候选独立确认。\n\n"
            + development_review + "\n\n" + development_cost_text + "\n\n" +
            "Codex在结果开放前实际看前四训练对八图并核对13原日期块：E渐进下降与单阶跃/平台区别可见；"
            "第一对阶跃在缺测内，仅两肩差值不足以判仪器来源。四对误差差异≤8.882e−16毫米，"
            "但重新估计噪声尺度略不同，未声称全部输入尺度相同。当前复核者知道历史标签与生成设计，"
            "不冒充独立专家、不改参考、不追加缺支撑A/S/D。原图与数字见paired-persistence-visual-review-v1。"
            "校准后另实际查看四幅完整诊断图：case_0029新增FP41而净纠错−36；case_0036弱检出92/199对增广36/199，"
            "但新增FP，expert24弱检出109/199仍更强。好坏案例都保留，不以极值图替代全量评价。\n\n"
            f"准备214.608秒；共享绘图/预处理/编码总耗时{encoding['wall_seconds']:.3f}秒，"
            f"14个神经头训练终端三位小数合计约{sum(row['train_seconds_rounded'] for row in training['rows'] if row['train_seconds_rounded'] is not None):.3f}秒；"
            "XGB训练、各头推断及总校准时间未在导出失败前保存，保留未知。"
            "视觉部署必须计入编码而不能只报小头时间；模型下载/首次载入另在runtime记录。0新API请求，"
            "原受控192/半合成60及4017启动/4015回执、至少17245825报告token不变。"
            "准备时Pydantic提示模型copy的list与tuple类型差异，进程正常完成；不以类型提示当算法失败，"
            "该源码和原产物冻结保留。训练后的评分汇总与已有准备summary.json同名，独占导出拒绝覆盖而中断；"
            "finalize脚本只从已有预测恢复validation-summary及原gate，不重新训练、选参数或改分，原失败源码/记录保留。\n\n"
            "Anlu实际阅读官方PDF第1、3、4、5页；MobileCLIP2仅官方README及实际配置/权重核查，"
            "未读全文或重训作者模型。CAPS官方47页PDF实际读1、3—6页及22页部分，固定作者源码读README、"
            "配对dataset1—233及detector504—647行，未运行作者代码；一般排序配对已存在，不称首创。"
            "完整检索与阅读范围见residual-identifiability-literature-v1/reading-notes.md。\n\n"
            "## 复现\n\n"
            "```powershell\n$env:PYTHONPATH='artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-runtime-v1/site-packages;src;scripts'\n"
            "$env:PYTHONIOENCODING='utf-8'\n& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe "
            "scripts/report_landslide_paired_persistence.py --run artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-calibration-v1 "
            "--out artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-user-replay\n```\n\n"
            "输出目录需尚不存在。重放不训练、不重新请求；协议、源码、原输入、权重、380校准组合、"
            "全部选择和gate保留。九项相关业务测试与校准新增源码Ruff通过；14文件校准报告重放字节一致。"
            "开发runner的Ruff检查返回一个import格式I001，冻结执行期间未修改源码；开发报告脚本Ruff通过。"
            "goal active，尚需稳定机制贡献及未参与选择的可信验证。\n\n"
            "完整开发两次重放15文件字节一致，175200方法原日概率/预测检查，原37基线任务指标逐值一致。\n\n"
            "![成对监督与强基线](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v53/paired-persistence-comparison.png)\n")
    report_path = ROOT / "docs/landslide-paired-persistence-results.md"
    report_path.write_text(text, encoding="utf-8")
    manuscript_path = ROOT / "docs" / (DOCUMENTS[0] + ".md")
    manuscript = manuscript_path.read_text(encoding="utf-8").replace("完整中文初稿v52", "完整中文初稿v53", 1)
    manuscript = manuscript.replace("更新前v51主稿及三状态文档原字节存档。新增原块事实复核原型及34真实请求；符号陈述变化未改变活动标签，候选停止，12契约失败及未定义Precision保留。",
        "更新前v52主稿及三状态文档原字节存档。新增冻结预训练视觉编码及同净位移持续性配对监督；同数据增广、打乱配对、模态消融和强XGB均用统一任务校准，决定据真实实验报告。", 1)
    addition = ("### 6.4.17 训练持续性配对与冻结预训练视觉编码\n\n" + formula_section + "\n\n" +
                "原24训练/12验证，137对来自16训练parent、48弱中心，不是新独立记录。七个学习臂与"
                "三个同增广强XGB各38校准机会，普通VLM/固定Agent完整开发比较按通过校准后的冻结方案完成。\n\n" +
                outcome + "\n\n" + development_outcome + "\n\n" + development_limit + "\n\n" + next_action + "\n\n"
                + development_review + "\n\n" + development_cost_text + "\n\n" +
                "八幅训练QA图由Codex实际看图/原块复核，无参考回写或专家独立性；噪声代理改变与缺测内阶跃的不可辨识性披露。"
                "统一评分、冻结权重与确定性重放和全部结果见[成对持续性结果](landslide-paired-persistence-results.md)。\n\n")
    marker = "### 6.5 阶段与计算成本"
    if manuscript.count(marker) != 1:
        raise ValueError("Manuscript insertion point changed")
    manuscript = manuscript.replace(marker, addition + marker, 1).replace("截至v52累计", "截至v53累计", 1)
    manuscript += ("\n[41] Lan, T., Gao, Y., Lu, Y., An, X., Wang, M., Pan, Y., He, W., & Zhang, C. (2026). "
        "Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision. "
        "[arXiv:2610.06180v1](https://arxiv.org/abs/2610.06180v1). 官方PDF第1、3、4、5页实际读取，未运行作者模型。\n\n"
        "[42] Faghri, F., Vasu, P. K. A., Koc, C., Shankar, V., Toshev, A. T., Tuzel, O., & Pouransari, H. (2025). "
        "MobileCLIP2: Improving Multi-Modal Reinforced Training. [arXiv:2508.20691](https://arxiv.org/abs/2508.20691). "
        "[官方源码](https://github.com/apple/ml-mobileclip)。核对README、OpenCLIP实际配置并载入预训练视觉权重，未读论文全文/重训。\n\n"
        "[43] Gao, Y., Lan, T., Lu, Y., An, X., Wang, M., He, W., Li, Y., & Zhang, C. (2026). "
        "Supervision Recovery for Time Series Anomaly Detection via Context-Anchored Pairing. "
        "[arXiv:2609.33610v1](https://arxiv.org/abs/2609.33610v1). [作者源码](https://github.com/gyf-0623/caps)。"
        "实际读PDF1、3—6页及22页部分、README、dataset1—233与detector504—647行，"
        "固定head e6b4c4ad6403d5aac569241ad19f30990e71b57a，未运行作者代码。\n")
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = "\n## 2026-10-11：持续性配对与预训练视觉校准，初稿v53\n\n" + outcome + "\n\n" + development_outcome + "\n\n" + next_action + "\n\n"
    status += ("137训练对/48弱中心、16原parent，0新独立记录/0新API。七学习臂和三同增广XGB各38机会；"
               "八训练图与原数字Codex实读，非独立专家。冻结权重/概率/原日和评分重放，公式继续LaTeX。"
               "[完整结果](landslide-paired-persistence-results.md)。v52四文档原字节及旧manifest存档，goal active。\n\n")
    for name in DOCUMENTS[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n" + status + rest, encoding="utf-8")
    export(archive / "v52-preservation-after.json", historical_checks(archive))
    export(archive / "progress.json", {"status": "goal_active", "classification": "progress",
        "new_requests": 0, "new_independent_records": 0, "new_training_descendants": 137,
        "decision": development_decision["decision"], "independent_confirmation": False})
    print("Wrote v53 with exact v52 preservation", flush=True)


def seal():
    archive = BASE / "manuscript-v53"
    if (archive / "delivery-manifest.json").exists():
        raise ValueError("Seal only a new delivery")
    historical_checks(archive)
    for name, folder in ((DOCUMENTS[0], "formula-verification"),
                         ("landslide-paired-persistence-results", "result-formula-verification")):
        audit = read_json(archive / folder / "formula-audit.json")
        if audit["status"] != "passed" or audit["source_sha256"] != sha(ROOT / "docs" / (name + ".md")):
            raise ValueError("Math audit differs from current delivered source")
    folders = ("manuscript-v53", "paired-persistence-calibration-v1", "paired-persistence-analysis-v1",
               "paired-persistence-analysis-replay-v1", "paired-persistence-visual-review-v1",
               "paired-persistence-calibration-error-review-v1", "paired-persistence-development-v1",
               "paired-persistence-development-error-review-v1",
               "paired-persistence-development-analysis-v1", "paired-persistence-development-replay-v1",
               "residual-identifiability-literature-v1")
    paths = [path for name in folders for path in (BASE / name).rglob("*") if path.is_file()]
    runtime = BASE / "paired-persistence-runtime-v1"
    paths.extend(path for path in runtime.iterdir() if path.is_file())
    paths.extend(ROOT / "docs" / (name + ".md") for name in (*DOCUMENTS, "landslide-paired-persistence-results"))
    paths.extend(ROOT / name for name in (
        "src/gnss_sim/landslide_paired_persistence.py", "scripts/prepare_landslide_paired_persistence.py",
        "scripts/encode_landslide_paired_persistence.py", "scripts/train_landslide_paired_persistence.py",
        "scripts/report_landslide_paired_persistence.py", "scripts/review_landslide_persistence_pairs.py",
        "scripts/write_landslide_paired_persistence_results.py", "scripts/finalize_landslide_paired_persistence.py",
        "scripts/run_landslide_paired_persistence_development.py", "scripts/report_landslide_paired_persistence_development.py",
        "tests/test_landslide_paired_persistence.py",
        "tests/test_landslide_paired_persistence_learning.py"))
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}
    export(archive / "delivery-manifest.json", {"files": files})
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Delivered bytes changed: " + name)
    print("Sealed", len(files), "files; goal active", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    arguments = parser.parse_args()
    if arguments.seal:
        seal()
    else:
        write()
