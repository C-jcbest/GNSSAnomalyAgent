"""Close the stopped axis-witness probe, its actual inspection and manuscript v41."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
PROBE = BASE / "axis-persistence-probe-v3"
REPORT = BASE / "axis-persistence-report-v1"
LITERATURE = BASE / "axis-persistence-literature-v1"
OUT = BASE / "manuscript-v41"
DOCUMENTS = [
    ROOT / "docs" / name
    for name in (
        "landslide-adaptive-review-manuscript-draft.md",
        "project-status.md",
        "landslide-adaptive-review-method.md",
        "landslide-submission-roadmap.md",
    )
]


def previous_delivery(archives=None):
    archives = archives or {}
    files = read_json(BASE / "manuscript-v40/delivery-manifest.json")["files"]
    for name, expected in files.items():
        if sha(ROOT / archives.get(name, name)) != expected:
            raise ValueError("v40 evidence changed: " + name)
    return len(files)


def close():
    for directory in (PROBE, REPORT, BASE / "axis-persistence-replay-v1"):
        verify_manifest(directory)
    previous_delivery()
    original = read_json(REPORT / "manifest.json")["files"]
    for name, expected in original.items():
        if sha(BASE / "axis-persistence-replay-v1" / name) != expected:
            raise ValueError("Offline replay differs: " + name)
    OUT.mkdir(exist_ok=False)
    archives = {}
    for path in DOCUMENTS:
        archived = OUT / (path.stem + "-v40-original.md")
        archived.write_bytes(path.read_bytes())
        archives[path.relative_to(ROOT).as_posix()] = archived.relative_to(ROOT).as_posix()
    write_json(OUT / "v40-document-archives.json", archives)
    selections = read_json(PROBE / "selections.json")
    findings = {
        "case_0187": "原始N在约270—400日缓慢上升；N分量分数明显高于E/U并多次越0.5。第一遍原零触发变为[270,450)一窗，61个漏运动冲突；第二遍同窗由47个否定性冲突触发。该观察没有执行，不能把图中选窗当新增TP。",
        "case_0171": "N两轮明显变化与中间约62mm平台清楚；分量分数在两轮强运动高，在中间平台低。第一遍选[180,360)，6个漏运动及9个否定性冲突；第二遍选[270,450)，86个否定性冲突，主要针对平台。全局初判与原始证据的一致强运动不一定取得观察优先级，图中新窗不等于完整活动覆盖。",
        "case_0183": "原始NEU在约220日共同单阶跃，其前后均为有噪声平台。新分量分数在阶跃邻域出现多次超过0.5的脉冲，每遍累计26个漏运动代理冲突，触发[180,360)；原选窗为空。理想无噪声单阶跃测试未产生14日活动，但真实噪声叠加后仍能形成14个累计冲突；测试不能外推普遍排阶跃。触发是额外观察，不是已经发生的视觉活动误报。",
    }
    write_json(
        REPORT / "developer-inspection.json",
        {
            "date": "2026-10-11",
            "reviewer": "Current Codex developer, actually viewed all three generated figures with original NEU and numerical evidence",
            "prior_exposure": "Historical and current methods/predictions/process labels; diagnostic review only",
            "independent_reference": False,
            "labels_or_predictions_changed": False,
            "new_agent_inference_executed": False,
            "records": [
                {
                    "case_id": cid,
                    "image_sha256": sha(REPORT / (cid + "-opportunities.png")),
                    "finding": finding,
                    "selected": {
                        str(rep): selections[f"{cid}-r{rep}"]["selected"] for rep in (0, 1)
                    },
                }
                for cid, finding in findings.items()
            ],
        },
    )
    inspection = "# 实际视觉与原始数值辅助复核\n\n当前Codex为开发者，已见标签与评分；本次是错误解释，不生成独立专家标签。三张图均已实际查看，没有执行新的VLM复核。\n"
    for cid, finding in findings.items():
        inspection += f"\n## {cid}\n\n{finding}\n\n![{cid}实际查看图]({cid}-opportunities.png)\n"
    (REPORT / "inspection.md").write_text(inspection, encoding="utf-8")
    notes = """# 本轮最近邻定向检索范围

官方arXiv查询与八条登记摘要见search-ledger.json及原XML。本轮实际读取全部八摘要，未新增全文阅读、作者源码执行或模型复现。LEARN-TS、LLM-Generated Feature Pools、ORCA、CGTime、CHARM等均有已有统计/通道/语言表示先例；不能把逐轴统计、秩符号或Agent框架本身主张首创。CHARM摘要的通道标识和JEPA训练与本轮原始位移观察触发不同，仅读摘要不推断其GNSS效果或公开实现。

此前ORCA完整第4节及作者相关代码、CGTime相关第2节的实际阅读范围仍按feedback-placement-literature-v1/reading-scope.json记录，没有因本轮命中就扩大为全文阅读。当前新候选的价值仅由本地统一任务验证；已停止，未声称新方法贡献得到确认。
"""
    (LITERATURE / "reading-notes.md").write_text(notes, encoding="utf-8")
    write_json(
        LITERATURE / "reading-scope.json",
        {
            "abstracts_actually_read": [
                row["id"] for row in read_json(LITERATURE / "search-ledger.json")["records"]
            ],
            "fulltext_newly_read": [],
            "author_models_executed": False,
            "systematic_review": False,
        },
    )
    write_json(
        LITERATURE / "manifest.json",
        {"files": {path.name: sha(path) for path in LITERATURE.iterdir() if path.is_file()}},
    )
    formula = r"""
## 候选证据的明确公式

七个14日原始块中位数记为 $m_0,\ldots,m_6$，每轴 $j\in\{N,E,U\}$ 的两半秩符号为

$$
u_j^-=\frac{1}{6}\sum_{0\le p<q\le3}\operatorname{sgn}(m_{q,j}-m_{p,j}),\qquad
u_j^+=\frac{1}{6}\sum_{3\le p<q\le6}\operatorname{sgn}(m_{q,j}-m_{p,j}).
$$

两半标准化净变化为 $d_j^-=(m_{3,j}-m_{0,j})/\sigma_j$、$d_j^+=(m_{6,j}-m_{3,j})/\sigma_j$，同向条件及分数为

$$
z_j=\mathbf{1}[u_j^-u_j^+>0,\ d_j^-d_j^+>0]\min(|u_j^-|,|u_j^+|)\min(|d_j^-|,|d_j^+|),\qquad z=\max_j z_j.
$$

任一块不足8个完整NEU日时，整个中心证据不可用，保持null。0.5/0.75门槛来自各自固定验证划分；最大分量并未校准为多重检验，也不是概率。证据只提出观察，不自动确认活动。

## 实际复核与来源阅读

[三例实际视觉/数值复核](inspection.md)解释弱单轴观察机会、平台冲突以及有噪声单阶跃触发；不回填任何标签或预测。[本轮最近邻检索与阅读范围](../axis-persistence-literature-v1/reading-notes.md)仅读八摘要，已有全文/源码范围没有扩大。本轮候选在推理前停止；研究目标仍active。
"""
    complete = (REPORT / "report.md").read_text(encoding="utf-8") + formula
    (REPORT / "report-complete.md").write_text(complete, encoding="utf-8")
    prefix = "../artifacts/landslide-adaptive-review-2026-10-09/"
    canonical = complete
    for cid in findings:
        canonical = canonical.replace(
            f"({cid}-opportunities.png)",
            f"({prefix}axis-persistence-report-v1/{cid}-opportunities.png)",
        )
    canonical = canonical.replace(
        "(inspection.md)", f"({prefix}axis-persistence-report-v1/inspection.md)"
    ).replace(
        "(../axis-persistence-literature-v1/reading-notes.md)",
        f"({prefix}axis-persistence-literature-v1/reading-notes.md)",
    )
    (ROOT / "docs/landslide-axis-persistence-results.md").write_text(canonical, encoding="utf-8")
    main = DOCUMENTS[0].read_text(encoding="utf-8").replace("完整中文初稿v40", "完整中文初稿v41", 1)
    addition = """### 6.4.5 逐分量连续变化触发的开发否定结果

针对弱运动零触发，另检验七个不相交14日原始块的逐轴双侧秩一致性与净变化幅度，以保留弱单轴证据并移除固定速率下限；原始块支撑、14冲突日、两窗预算和49日权限保持。门槛按既有过程24训练/12验证选为0.5，作者数值对照按其6/6划分选0.75；同信息XGBoost仍给六个合理候选。没有用当前开发24/12调整门槛。

两遍弱日观察覆盖733/1125→1049/1172，复核次数23/27→23/25，但阴性记录每遍从0增至1次观察，未通过准备前固定的准入条件。因此停止此触发候选，未执行17个缺失问题；观察覆盖不冒称识别Recall。纯数值规则/新XGBoost过程F1 .7534/.7484，作者.7224/.7192；过程覆盖均.7011、作者.6726，严格七块支撑降低可判日期，分数差不能全部归于表示优劣。原强XGB84过程F1 .8848、原始多尺度规则.8276等全部保留。

实际看三图并结合原始数值：0187弱N累积获新窗；0171新窗主要针对高位平台冲突；0183原始NEU单阶跃叠加噪声产生26个累计代理冲突，取得额外观察。后者是观察预算退化，未建立视觉误报或领域活动。新候选STOP_AXIS_WITNESS_BEFORE_CALLS，原主确认和反馈停止结论不变。完整47/44方法、活动/事件/边界/阶段、公式、失败来源与读图见[逐分量触发结果](landslide-axis-persistence-results.md)。

"""
    main = main.replace("### 6.5 阶段与计算成本", addition + "### 6.5 阶段与计算成本", 1)
    DOCUMENTS[0].write_text(main, encoding="utf-8")
    (OUT / "manuscript-v41.md").write_bytes(DOCUMENTS[0].read_bytes())
    status = """## 2026-10-11：逐分量连续变化触发停止于请求前，初稿v41

[完整结果](landslide-axis-persistence-results.md)。新候选将七原始块的逐轴双侧秩符号与净变化作为选窗证据，保持180日窗、14冲突日、两窗预算、49日权限和纯图提示。原过程24训练/12验证选0.5，作者新数值控制6/6选0.75；各九规则/六ML机会，没有开发集调分。

完整24过程两遍弱日观察覆盖733/1125→1049/1172、复核23/27→23/25，但每遍静稳/阶跃观察从0增1，准入失败，STOP_AXIS_WITNESS_BEFORE_CALLS；未执行17缺失问题，不伪造Agent结果。数值规则/同信息XGB过程F1 .7534/.7484，作者.7224/.7192；支撑覆盖过程.7011/作者.6726。原45/42全部指标及成本保持，新增2数值后47/44；原NOT_CONFIRMED、STOP_FEEDBACK_PROMOTION不变。

实际看0187/0171/0183三图，用原始NEU及逐日代理辅助检查：弱N获观察、平台冲突占窗、有噪声单阶跃累计26代理日导致额外观察。观察机会不等于识别Recall，额外阴性观察不等于视觉FP。八篇新检索摘要读取，未新增作者模型执行。v1读入旧测试编号在拟合前失败；v2冻两套校准后NaN JSON导出失败；v3只修输出null并逐字节复用已冻校准，未重拟合；失败目录保留。

新API0、记录0；累计3725启动/3723回执、至少16167212报告token，受控192/半合成60不变。v40主稿/三状态文档精确存档，旧manifest保持，经档案映射核验；v41公式继续LaTeX。研究goal active，未完成独立现场参考验证，不复活停止候选或改门槛救分。下方为历史记录。

"""
    for path in DOCUMENTS[1:]:
        title, body = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(title + "\n\n" + status + body.lstrip("\n"), encoding="utf-8")
    write_json(
        OUT / "closure.json",
        {
            "decision": read_json(PROBE / "decision.json"),
            "new_remote_requests": 0,
            "old_deliverables_changed": "Only four mutable documents; exact prior bytes archived",
            "replay_byte_identical_files": len(original),
            "research_goal": "active",
            "formal_independent_field_reference": False,
        },
    )
    print("Closed stopped axis-witness evidence and manuscript v41")


def finalize():
    count = previous_delivery(read_json(OUT / "v40-document-archives.json"))
    for folder in (PROBE, REPORT, BASE / "axis-persistence-replay-v1", LITERATURE):
        verify_manifest(folder)
    days = 0
    additions = read_json(REPORT / "numerical-predictions.json")
    pairs = {}
    for target in additions.values():
        for cases in target.values():
            for cid, prediction in cases.items():
                if cid not in pairs:
                    pairs[cid] = load_case_features(BASE / "core-lockbox-data-v1", cid)
                feature = pairs[cid][1]
                activity, stage = np.array(prediction["activity"]), np.array(prediction["stage"])
                if np.any(activity[~feature.observed] != -1) or np.any(
                    stage[~feature.observed] != -1
                ):
                    raise ValueError("Missing observation has definite output")
                if np.any((stage > 0) & ((activity != 1) | ~feature.supported)):
                    raise ValueError("Stage exceeds supported activity")
                days += len(activity)
    for doc, audit in (
        (DOCUMENTS[0], OUT / "math-check-v1/formula-audit.json"),
        (
            ROOT / "docs/landslide-axis-persistence-results.md",
            OUT / "report-math-check-v1/formula-audit.json",
        ),
    ):
        result = read_json(audit)
        if result["status"] != "passed" or result["source_sha256"] != sha(doc):
            raise ValueError("Current math not verified")
        text = doc.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target and not (doc.parent / target.split("#")[0]).exists():
                raise ValueError("Broken local link: " + target)
    write_json(
        OUT / "validation.json",
        {
            "status": "passed",
            "old_v40_files_verified_via_exact_archives": count,
            "new_prediction_days_checked": days,
            "missing_and_stage_gate": "passed",
            "old_all_results_and_costs": "Preserved; common scoring independently checks daily activity, events and stage counts",
            "offline_replay_byte_identical_files": read_json(OUT / "closure.json")[
                "replay_byte_identical_files"
            ],
            "math_and_local_links": "passed",
            "business_tests": "Three passed, including ideal-step/slow-ramp, irrelevant-axis/sign and missing block",
            "readability": "All new functions and reused call sites reviewed; no stopped module integrated into default",
        },
    )
    write_json(
        REPORT / "delivery-manifest.json",
        {
            "files": {
                path.relative_to(REPORT).as_posix(): sha(path)
                for path in REPORT.rglob("*")
                if path.is_file() and path.name != "delivery-manifest.json"
            }
        },
    )
    folders = [
        BASE / name
        for name in (
            "axis-persistence-probe-v1",
            "axis-persistence-probe-v2",
            "axis-persistence-probe-v3",
            "axis-persistence-report-v1",
            "axis-persistence-replay-v1",
            "axis-persistence-literature-v1",
            "manuscript-v41",
        )
    ]
    paths = {path for folder in folders for path in folder.rglob("*") if path.is_file()}
    paths.update(DOCUMENTS)
    paths.add(ROOT / "docs/landslide-axis-persistence-results.md")
    paths.update(
        ROOT / "scripts" / name
        for name in (
            "probe_landslide_axis_persistence.py",
            "report_landslide_axis_persistence.py",
            "close_landslide_axis_persistence.py",
        )
    )
    paths.update(
        (
            ROOT / "src/gnss_sim/landslide_axis_persistence.py",
            ROOT / "tests/test_landslide_axis_persistence.py",
        )
    )
    destination = OUT / "delivery-manifest.json"
    paths.discard(destination)
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}
    write_json(destination, {"files": files})
    if any(sha(ROOT / name) != expected for name, expected in files.items()):
        raise ValueError("Delivery changed during binding")
    print(f"Verified {len(files)} delivery files and {days} new numerical prediction days")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("close", "finalize"))
    args = parser.parse_args()
    if args.mode == "close":
        close()
    else:
        finalize()
