"""Preserve v51 and deliver the stopped factual-claim mechanism experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha

DOCUMENTS = ("landslide-adaptive-review-manuscript-draft", "project-status",
             "landslide-adaptive-review-method", "landslide-submission-roadmap")


def historical_checks(archive):
    manifest = read_json(archive / "v51-delivery-manifest-original.json")
    checks = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if name in {"docs/" + value + ".md" for value in DOCUMENTS}:
            path = archive / (Path(name).stem + "-v51-original.md")
        if sha(path) != expected:
            raise ValueError("Historical delivered bytes changed: " + name)
        checks.append({"original_path": name, "preserved_path": path.relative_to(ROOT).as_posix(),
                       "sha256": expected})
    return checks


def write():
    archive = BASE / "manuscript-v52"
    archive.mkdir(exist_ok=False)
    for name in DOCUMENTS:
        (archive / (name + "-v51-original.md")).write_bytes((ROOT / "docs" / (name + ".md")).read_bytes())
    (archive / "v51-delivery-manifest-original.json").write_bytes(
        (BASE / "manuscript-v51/delivery-manifest.json").read_bytes())
    export(archive / "v51-preservation-before.json", historical_checks(archive))
    analysis = BASE / "claim-review-analysis-v2"
    decision = read_json(analysis / "decision.json")
    if decision["decision"] != "STOP_FACTUAL_CLAIM_REVIEW_PILOT":
        raise ValueError("This delivery describes the actual stopped candidate")
    review = BASE / "claim-review-visual-review-v1"
    review.mkdir(exist_ok=False)
    export(review / "review.json", {
        "reviewer": "Current Codex visual reading plus original numerical block inspection",
        "historical_label_exposure": True, "independent_expert": False, "reference_changed": False,
        "images_actually_viewed": ["claim-review-pilot-v1/q" + str(i) + "-time.png" for i in range(1, 5)],
        "observations": [
            {"query": "q1", "case": "case_0010", "window": [315, 357],
             "interpretation": "N/E/U strong earlier cumulative movement approaches plateau; center N reverses after slot6, E first rises then falls. Exact daily transition unresolved from14-day medians.",
             "center_blocks": [[29.844874163560316, -15.598107366668174, -17.036337260952312],
                               [30.76315473913291, -15.05772725977816, -17.63395006754137],
                               [30.54804353243577, -16.458365384853487, -18.600381468483494]]},
            {"query": "q2", "case": "case_0012", "window": [455, 497],
             "interpretation": "Small NE cumulative rise continues into following blocks; U turns down later. Query onset is weak and daily boundary is not visually exact. First image reply fails contract; second misses38 known active days."},
            {"query": "q3", "case": "case_0003", "window": [245, 287],
             "interpretation": "E has sustained cumulative decrease across query and surroundings; N/U fluctuate. Both initial and fact-reviewed replies label all42 dates inactive despite correction of numerical sign claims."},
            {"query": "q4", "case": "case_0001", "window": [77, 119],
             "interpretation": "NEU fluctuations and isolated excursions without clear continuing cumulative query change; raw sign changes are noise descriptions, not motion confirmations. Initial answers both fail strict output contract."},
        ],
        "stage_support_ASD": [0, 80, 14], "A_evaluable": False,
        "stage_head": "Original shared numerical head, activity and original support gated",
        "stage_macro_limit": "Internal fixed three-class empty-class zero; end-to-end undefined-class exclusion retained",
        "claim_limit": "Five triggered activity arrays byte-value identical before/after; false claims46/60 to20/57 concrete pairs,3 pairs unknown. Claim arithmetic improvement is not recognition improvement.",
        "preparation_QA": "Four actual original time images checked before inference; no image modification",
    })
    literature = BASE / "claim-review-literature-v1"
    (literature / "reading-notes.md").write_text(
        "# 原始块事实复核的最近邻阅读\n\n官方arXiv RefChecker查询4条题名及完整摘要实际读取。"
        "本轮相关来源是Hu等RefChecker arXiv2405.14486v1；另两条同名引用工具不混为同一个项目。"
        "作者amazon-science/RefChecker固定head1df1b25cee792ba2b171302e31ca4f768bd67703。"
        "实际读README第1—80行以及refchecker/checker/llm_checker.py第1—140行，后者展示"
        "按参考/问题构造claim-triplet联合检查提示并调用LLM。未读论文全文、未运行作者代码，"
        "不称通用事实分解或参考核查首创。GNSS候选仅将相邻原日期块的三轴符号声明"
        "直接与确定观测算术核对，误读时至多一次修正；数字符号正确仍不证明运动。"
        "论文与作者源码链接：https://arxiv.org/abs/2405.14486v1 ；"
        "https://github.com/amazon-science/RefChecker 。\n", encoding="utf-8")
    summaries = read_json(analysis / "results.json")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    positions = np.arange(2)
    for offset, name, label in ((-.23, "image_initial", "Image initial"),
                               (0, "conditional_fact_review", "Checked-fact review"),
                               (.23, "matched_repeat", "Matched repeat")):
        axes[0].bar(positions + offset, [summaries[f"{name}_r{i}"]["aggregate"]["daily"]["f1"] for i in range(2)],
                    width=.23, label=label)
    axes[0].set_xticks(positions, ["Repeat0", "Repeat1"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Activity F1 | four selected training queries")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions - .15, [18, 28], width=.3, label="Initial | same triggered queries")
    axes[1].bar(positions + .15, [14, 6], width=.3, label="Fact review")
    axes[1].set_xticks(positions, ["Repeat0: 24/24 concrete pairs", "Repeat1: 36/33 concrete pairs"], fontsize=8)
    axes[1].set_ylabel("False concrete pair claims | not activity errors")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(archive / "claim-vs-recognition.png", dpi=160)
    plt.close(figure)
    report = r"""# 原始块事实核对与一次视觉更正：没有活动纠错，停止候选

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run / validate inline；2026-10-11
- Verification: 34原payload/回答/标签/事实及168原日支撑重放；仅训练窗口
- Version: claim-review-pilot-v1 / analysis-v2 / manuscript-v52

已见错误显示，视觉回答的“连续同向变化”描述可能与实际原块值矛盾。本轮实现一个有界事实复核原型：图像初判同时返回42原日活动标签及十二相邻块的NEU符号声明。对十三14日原始中位块 $z_j$ 核对

$$
c_{j,a}=\operatorname{sgn}(z_{j+1,a}-z_{j,a}),\qquad j=0,\ldots,11,\quad a\in\{N,E,U\}.
$$

每块至少8完整原日；缺块不压缩或跨连。绝对差至多 $10^{-12}$ 视算术零，非运动/置信门槛。null表示未读清，已有块的null不当作错误；若给缺块非空声明，或具体符号与原值不符，触发一次同图视觉更正，只附错误原块对及实际差值。数值不直接赋活动，不证明两块间每天运动，原未知/缺测和阶段支撑保持。全程无循环或重试。

RefChecker已有细粒度声明与参考核对，不以事实分解、核查或再次询问本身称首创。本轮实际读其官方摘要、README1—80行和固定head检查器1—140行，未读全文/运行模型；范围在literature-v1记录。此候选检验的是GNSS原日块算术反馈能否改变活动识别。

只从原24训练按强base_expert残余错误选一最高FP窗、一最高弱FN窗及两个正确正/负控制，不同记录、42日查询/182日原观察；已暴露标签用于选择，不进入提示。四窗0010[315,357)、0012[455,497)、0003[245,287)、0001[77,119)。共167已知观测日，94活动/73静稳，80弱活动；仅训练诊断，不能说独立效果或公平超过数值。

两遍共享图像初判，五臂：图像初判、首轮原块数值表、按真实错误条件追加更正、同条件同次数普通再观察、总是追加所有块差值。实际触发2/3窗，主每遍6/7部署调用，对普通再观察6/7、固定8/8、图像4/4、数值首轮4/4。后续臂在每窗依更正→重复→固定顺序执行，未交错随机化；模型版本不能由客户端固定。它们均保留声明任务，不能据此比较历史无声明提示的因果效果。

| 方法 | F1 r0/r1 | P r0/r1 | R r0/r1 | 弱R | FP r0/r1 | 覆盖 r0/r1 | 调用 r0/r1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 图像初判 | .254545/.245614 | .875/.700 | .148936/.148936 | 0/0 | 2/6 | .502994/.748503 | 4/4 |
| 条件事实更正 | .254545/.245614 | .875/.700 | .148936/.148936 | 0/0 | 2/6 | .502994/.748503 | 6/7 |
| 同次数普通再观察 | .254545/.245614 | .875/.700 | .148936/.148936 | 0/0 | 2/6 | .502994/.748503 | 6/7 |
| 固定全块事实 | .254545/.245614 | .875/.700 | .148936/.148936 | 0/0 | 2/6 | .748503/.748503 | 8/8 |
| 首轮数值表 | 0/0 | 未定义/未定义 | 0/0 | 0/0 | 0/0 | 0/.251497 | 4/4 |
| base18训练诊断 | .817204 | .826087 | .808511 | .775 | 16 | 1 | 0 |
| base_expert训练诊断 | .766667 | .802326 | .734043 | .6875 | 17 | 1 | 0 |

更正、普通再观察与初判的全部活动数组逐值相同：两遍净确定纠错0/0，没有新增修正、弱R均0。五个实际触发窗的错误具体块对声明46/60降为20/57，另3个变未知；第一遍q1反而7→12错误，故非稳定事实提升。q3两遍声明11/10→2/2错误，仍把E持续下降的42活动日全判静稳；事实陈述变好不等于运动识别变好。12/17登记条件通过，STOP_FACTUAL_CLAIM_REVIEW_PILOT；不换schema、窗、阈值、seed、提示救分，不接入主方法。

34启动/34回执均返回JSON，但只有22回复满足完整契约。12失败中10使用NEU对象替代符号三元列表，另2为13或11块对；失败整个活动输出未知、不解析修复后重评。7/8数值首轮失败，不能把其0分解释为数值观察没有价值。主初判3失败、固定2失败；纯图重复与条件更正均5/5合法。失败、原回答和代价全部保留，0重试。总73930报告token，API117.013秒/循环117.217秒；主两遍10854/13370token，普通再观察9319/10936，固定16498/16535。CPU/绘图未单独计时，不填零CPU。

事件F1 .3/.5主和初判两遍皆.5/.5；边界起/止MAE0/2日及0/6日，是42日裁窗且只一个匹配事件，不是完整物理起止。阶段支持[A,S,D]=[0,80,14]，A不可评价；内部固定三类宏F1 .333333，两遍端到端.466667/.411765，空类口径保留、不能直接比较推断收益。共享旧阶段头，仅活动门控，无视觉阶段贡献。原规则、PELT、XGB、直接VLM、固定Agent完整比较保持，本小试不替代完整任务。

作者实际看四原始图并用中心原块复核：0010强运动后接平台，原N中心增量变号；0012弱NE累计、起边界仍不精确；0003明显E持续下降而模型仍全静稳；0001只有波动和孤立尖峰。作者有历史标签暴露、未回写参考，不新增专家金标准或A/S/D标签。图可见性与算术声明纠正均不保证模型使用证据。

评分v1因零阳性输出的未定义Precision与sklearn约定零直接比较失败，在评分前停止、原源码保留；新评分副本只按原计数分母验证None，所有指标仍复用统一evaluate_case/aggregate_scores与事件边界协议，不改旧冻结评分器。一个新增全弃判回归验证未定义P、零R/F1、14漏检/零覆盖。追加测试误改冻结测试文件时哈希检查阻止报告执行；已原字节恢复，新增评分测试放独立文件，未改推理或重请求。

## 复现与研究状态

```powershell
$env:PYTHONPATH='src;scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/report_landslide_claim_review.py --run artifacts/landslide-adaptive-review-2026-10-09/claim-review-pilot-v1 --out artifacts/landslide-adaptive-review-2026-10-09/claim-review-user-replay
```

使用未存在目录；只重放，不重新infer终态试验。17相关业务测试与新增源码Ruff通过；无API确定性报告重放13文件字节相同。累计4017启动/4015回执、至少17245825报告token，受控192/半合成60不变。goal active：尚未取得可支撑核心的新稳定贡献；本候选完成实现、对照、验证与负结果，不扩展。

![事实错误与活动识别](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v52/claim-vs-recognition.png)
"""
    (ROOT / "docs/landslide-claim-review-results.md").write_text(report, encoding="utf-8")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8").replace("完整中文初稿v51", "完整中文初稿v52", 1)
    manuscript = manuscript.replace("更新前v50主稿及三状态文档原字节存档。新增监督冲突训练前检及空间轨迹三臂两遍24请求；轨迹无净增益而停止，潜在标签可辨性与阶段空类口径单独披露。",
        "更新前v51主稿及三状态文档原字节存档。新增原块事实复核原型及34真实请求；符号陈述变化未改变活动标签，候选停止，12契约失败及未定义Precision保留。", 1)
    addition = """
### 6.4.16 原始块事实核对未带来活动纠错

受已见回答与原值矛盾启发，新增原型把十三原始中位块相邻NEU符号作为可核对声明，具体假声明触发至多一次同图事实更正；未知不当假声明，正确算术不直接赋活动。一般声明核查已有RefChecker[40]，不称首创。四个训练错误/正确控制窗、两遍34真实请求中22符合契约，12失败全留未知；7/8数值首轮失败，不能以其0分否定数值价值。更正两遍F1 .254545/.245614、弱R0，与初判和同次数普通再观察活动数组全部相同，净纠错0/0。五触发窗假声明46/60→20/57、另3未知，但无活动修正，12/17条件，STOP_FACTUAL_CLAIM_REVIEW_PILOT。它说明在本小试中核对算术没有使模型改变运动判断，不据此断言所有事实反馈无用。

作者实际四图/原块复核及34payload/事实/预测、168原日/阶段支撑和评分重放完成。新73930token/API117.013秒；A无支持，S/D80/14，阶段仍来自共同数值头。零阳性输出Precision保留未定义，评分独立核验只修正None边界，不改历史评分器或推理。完整对照、失败及复现见[事实复核结果](landslide-claim-review-results.md)，不扩展此无收益候选。

"""
    marker = "### 6.5 阶段与计算成本"
    if manuscript.count(marker) != 1:
        raise ValueError("Manuscript insertion point changed")
    manuscript = manuscript.replace(marker, addition + marker, 1).replace(
        "截至v51累计3983启动/3981回执、至少17171895报告token",
        "截至v52累计4017启动/4015回执、至少17245825报告token", 1)
    manuscript += ("\n[40] Hu, X., Ru, D., Qiu, L., Guo, Q., Zhang, T., Xu, Y., Luo, Y., Liu, P., Zhang, Y., & Zhang, Z. (2024). "
        "RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models. "
        "[arXiv:2405.14486v1](https://arxiv.org/abs/2405.14486v1). "
        "[作者源码](https://github.com/amazon-science/RefChecker)。官方摘要、README1—80行及检查器1—140行已读，"
        "固定head1df1b25cee792ba2b171302e31ca4f768bd67703；未读全文/运行模型。\n")
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = ("\n## 2026-10-11：原块事实复核没有活动纠错，初稿v52\n\n"
        "[完整结果](landslide-claim-review-results.md)。四训练窗两遍，明确符号假声明触发一次更正，"
        "34请求/34JSON回执但22契约成功、12失败留未知；0重试。主F1 .254545/.245614，"
        "弱R0，活动数组同初判/普通再观察、净纠错0/0。假声明46/60→20/57另3未知，"
        "不等于运动识别收益；12/17条件STOP_FACTUAL_CLAIM_REVIEW_PILOT，不换schema/提示救分。\n\n"
        "作者实际四图/原块；未改参考。34payload/事实/预测、168原日/阶段和统一评分核验，"
        "新增全弃判边界测试保留未定义Precision；旧评分器未改。新73930token/API117.013秒，"
        "累计4017/4015、至少17245825token，受控192/半合成60不变。RefChecker官方摘要与"
        "README/检查器限定范围实读，非作者模型复现。v51四文档原字节、328旧文件映射保留；"
        "公式继续LaTeX。goal active：新稳定实质贡献与独立可信确认尚未完成。\n\n")
    for name in DOCUMENTS[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n" + status + rest, encoding="utf-8")
    export(archive / "v51-preservation-after.json", historical_checks(archive))
    export(archive / "progress.json", {"status": "goal_active", "classification": "progress",
        "new_requests": 34, "new_receipts": 34, "new_reported_tokens": 73930, "new_records": 0,
        "decision": decision["decision"], "independent_confirmation": False})
    print("Wrote v52; preserved328 historical files")


def seal():
    archive = BASE / "manuscript-v52"
    if (archive / "delivery-manifest.json").exists():
        raise ValueError("Seal only a new delivery")
    run = BASE / "claim-review-pilot-v1"
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen inference source changed")
    historical_checks(archive)
    for name, folder in ((DOCUMENTS[0], "formula-verification"),
                         ("landslide-claim-review-results", "result-formula-verification")):
        audit = read_json(archive / folder / "formula-audit.json")
        if audit["status"] != "passed" or audit["source_sha256"] != sha(ROOT / "docs" / (name + ".md")):
            raise ValueError("Math audit changed")
    folders = ("manuscript-v52", "claim-review-pilot-v1", "claim-review-analysis-v1",
               "claim-review-analysis-v2", "claim-review-analysis-replay-v1",
               "claim-review-visual-review-v1", "claim-review-literature-v1")
    paths = [path for name in folders for path in (BASE / name).rglob("*") if path.is_file()]
    paths += [ROOT / "docs" / (name + ".md") for name in (*DOCUMENTS, "landslide-claim-review-results")]
    paths += [ROOT / name for name in (
        "src/gnss_sim/landslide_claim_review.py", "scripts/probe_landslide_claim_review.py",
        "scripts/report_landslide_claim_review.py", "scripts/score_landslide_claim_review.py",
        "scripts/write_landslide_claim_review_results.py", "tests/test_landslide_claim_review.py",
        "tests/test_landslide_claim_review_scoring.py")]
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}
    export(archive / "delivery-manifest.json", {"files": files})
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Delivered bytes differ")
    print("Sealed", len(files), "files; goal remains active")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        write()
