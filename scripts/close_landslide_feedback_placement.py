"""Archive the same-feedback development evidence without promoting its failed candidate."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import verify_manifest
from probe_landslide_feedback_placement import verify
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
REPORT = BASE / "feedback-placement-report-v1"
LITERATURE = BASE / "feedback-placement-literature-v1"
ANALYSIS = BASE / "feedback-placement-analysis-v1"
OUT = BASE / "manuscript-v40"
MANUSCRIPT = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
STATE_DOCS = (
    ROOT / "docs/project-status.md",
    ROOT / "docs/landslide-adaptive-review-method.md",
    ROOT / "docs/landslide-submission-roadmap.md",
)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_previous_delivery(archives=None):
    """Old mutable documents resolve to their exact pre-edit snapshots after closure."""
    manifest = read_json(BASE / "manuscript-v39/delivery-manifest.json")
    archives = archives or {}
    for name, digest in manifest["files"].items():
        path = ROOT / archives.get(name, name)
        if sha(path) != digest:
            raise ValueError("Previous delivery differs: " + name)
    return len(manifest["files"])


def write_inspection():
    transitions = csv_rows(REPORT / "per-record-transitions.csv")
    evidence = read_json(REPORT / "raw-block-evidence.json")
    cases = {
        "case_0187": "第一遍无触发，数值反馈没有获得观察机会；固定位置恢复111过程TP，自适应相对它净−111日。N在270—390日四个30日块中位数2.22→3.01→4.74→5.63mm，图形与低幅持续累积一致。第二遍已触发，净+42却仍丢81TP，不能只展示较好重复。",
        "case_0171": "平台N约62mm，270—300、330—360、390—420日块中位数62.58、61.93、62.12mm。第二遍相对固定新增31TP、丢20TP、增8FP，却将162个正确静稳日改为未知，过程净−159日；作者净−177日。撤阳性或弃判不能等同于确认静稳。",
        "case_0174": "强运动后弱尾仍累积。第二遍相对固定新增127过程TP、无新增已知FP，净+127；作者新增121TP。第一遍共享全局字段失败，新增197TP却丢70TP至未知，另200个正确静稳日至未知，过程净−43，作者净−13。",
        "case_0177": "E/U两轮运动与中间平台可辨。第一遍相对固定新增59过程TP、增10FP，净+49。第二遍共享全局字段失败，新增199TP但180个正确静稳日至未知、增10FP，净仅+13；作者净−36且175个正确静稳日至未知。",
    }
    records = []
    for cid, finding in cases.items():
        image = REPORT / (cid + "-aligned.png")
        records.append(
            {
                "case_id": cid,
                "image": image.name,
                "image_sha256": sha(image),
                "actually_viewed": True,
                "finding": finding,
                "raw_blocks": evidence[cid],
                "full_fixed_numeric_transitions": [
                    row
                    for row in transitions
                    if row["case_id"] == cid and row["control"].startswith("fixed_numeric_feedback")
                ],
            }
        )
    write_json(
        REPORT / "developer-inspection.json",
        {
            "date": "2026-10-11",
            "reviewer": "Current Codex developer; vision plus original displacement and daily numerical transitions",
            "scope": "Four actually viewed aligned figures; post-score error explanation, not new labeling or independent validation",
            "prior_exposure": "Historical methods/results/generation labels; current predictions and process reference visible in diagnostic figures",
            "labels_or_predictions_changed": False,
            "comparison": "Adaptive numerical feedback, conflict scope versus complete fixed numerical feedback, whole target",
            "decision": "STOP_FEEDBACK_PROMOTION; original core NOT_CONFIRMED unchanged",
            "records": records,
        },
    )
    text = "# 同反馈位置对照的实际视觉与数值辅助复核\n\n"
    text += "当前Codex为开发者，已知历史方法、标签及本次评分；本次四幅对齐图用于解释错误，不建立独立参考，不更改原活动/阶段标签或预测。以下净日数均相对完整固定数值复核，不是相对直接视觉。未知不获确定静稳信用。\n"
    for cid, finding in cases.items():
        text += f"\n## {cid}\n\n{finding}\n\n![{cid}实际复核图]({cid}-aligned.png)\n"
    text += "\n逐日损益见per-record-transitions.csv，30日原始N/E/U块及有效日见raw-block-evidence.json；developer-inspection.json绑定实际查看图SHA。零触发限制与平台弃判共同解释整体升级失败。此解释不引入新阈值，不按诊断例子筛选评价人口。\n"
    (REPORT / "inspection.md").write_text(text, encoding="utf-8")


def write_literature_notes():
    ledger = read_json(LITERATURE / "search-ledger.json")
    scope = {
        "date": "2026-10-11",
        "search": "Targeted official arXiv query, eight metadata/abstracts read; not a systematic review",
        "records": [record["id"] for record in ledger["records"]],
        "cgtime": {
            "version": "2608.05238v2",
            "read": "Abstract; sections 2.1–2.5 and opening of 2.6",
            "not_read_in_full": "Remaining 2.6, 2.7, remaining full paper",
            "code_confirmed": False,
            "model_executed": False,
        },
        "orca": {
            "version": "2608.08859v1",
            "read": "Abstract; complete section 4 (4.1–4.3); relevant agent.py lines 84–143 and models.py lines 74–96",
            "repository": "https://github.com/bananabicycle/ORCA-Agentic",
            "tree_sha": "ba5d2b8060ef72dee86b538f03f1fc57f340b00c",
            "not_read_in_full": "Other paper sections and entire source files",
            "model_executed": False,
            "weights_loaded": False,
            "thresholds_transplanted": False,
        },
        "bound_sources": {
            name: sha(LITERATURE / name)
            for name in (
                "search-ledger.json",
                "query.json",
                "same_feedback_placement.xml",
                "cgtime.html",
                "cgtime-fulltext.txt",
                "orca.html",
                "orca-section4-read.txt",
                "orca-agent.py",
                "orca-models.py",
                "orca-tree.json",
                "orca-code-receipt.json",
            )
        },
    }
    write_json(LITERATURE / "reading-scope.json", scope)
    notes = """# 数值反馈与观察控制的最近邻核查

本轮是官方arXiv定向检索，实际读取八条登记摘要；不是系统综述。原查询、API XML、取得时间与书目见query.json和search-ledger.json。仅两篇进一步读取相关方法，范围由reading-scope.json逐项记录；没有运行作者模型、加载权重或调用作者服务。

## CGTime：计算事实与语言描述解耦

Xinran Feng, Yi Xie, Chao Zhang, Ruikun Li, Wanyun Ling, Ziyue Li, Chenxi Liu. *Decoupling Perception from Description: Computation-Grounded Representation Alignment between Multivariate Time Series and Language*. [arXiv:2608.05238v2](https://arxiv.org/abs/2608.05238v2).

已读摘要、第2.1—2.5节及2.6开头；未完整读2.6/2.7或整篇。该方法用确定性代码计算169类统计事实，监督描述、QA与可验证奖励；因此数值事实辅助语言判断不是本研究的首创。它训练时序—语言表示，不是本研究GNSS持续活动区间定位与有限观察位置对照。本次取得HTML没有确认作者代码链接，不把arXiv/LaTeXML系统链接当公开实现。

## ORCA：轻量统计选离散上下文

Anushka Roy, Jyotirmoy Singh, Shreea Bose, Chittaranjan Hota. *Agentic Anomaly Detection with ORCA-Style Dynamic Inductive Bias Adaptation in Multimodal Wearable Time Series Data*. [arXiv:2608.08859v1](https://arxiv.org/abs/2608.08859v1). [作者仓库](https://github.com/bananabicycle/ORCA-Agentic)，树SHA ba5d2b8060ef72dee86b538f03f1fc57f340b00c，共15文件，API树未截断。

实际完整读取第4节4.1—4.3，并核对agent.py统计/选择相关84—143行及models.py注意力相关74—96行；不声称整文件或全文阅读。源码用均值波动大于50选1、平均自相关峰lag大于1.5选5，否则选3，再更新attention.max_relative_position。轻量统计驱动观察/上下文选择已有先例，不能单凭Agent包装主张创新。源码对相对距离进行clamp后仍计算完整scores与attention；所读路径未见hard neighborhood mask，不把论文的截断支持叙述当源码已保证严格裁窗或计算成本降低。

该研究面向生理多变量异常与注意力偏置，本研究面向GNSS原始持续运动与全局视觉冲突的有限位置选择。未移植作者阈值，未执行作者检测器，未以作者发布分数充当本地GNSS比较。主稿创新表述据此收窄到具体运动语义、预算和统一对照下的位置机制；完整固定比较的负结果仍约束贡献强度。
"""
    (LITERATURE / "reading-notes.md").write_text(notes, encoding="utf-8")
    write_json(
        LITERATURE / "manifest.json",
        {
            "files": {
                path.name: sha(path) for path in sorted(LITERATURE.iterdir()) if path.is_file()
            }
        },
    )


def close():
    verify(BASE / "feedback-placement-development-v1")
    for folder in (ANALYSIS, BASE / "feedback-placement-replay-v1", REPORT):
        verify_manifest(folder)
    old_count = verify_previous_delivery()
    OUT.mkdir(exist_ok=False)
    archives = {}
    for path in (MANUSCRIPT, *STATE_DOCS):
        archived = OUT / (path.stem + "-v39-original.md")
        archived.write_bytes(path.read_bytes())
        archives[path.relative_to(ROOT).as_posix()] = archived.relative_to(ROOT).as_posix()
    write_json(OUT / "v39-document-archives.json", archives)
    write_inspection()
    write_literature_notes()
    original_report = (REPORT / "report.md").read_text(encoding="utf-8")
    additions = """
## 实际读图解释及最近邻核查

本次实际查看上述四张对齐图，并用原始块中位数与逐日转换辅助复核。[逐例读图记录](inspection.md)披露开发者身份与暴露，developer-inspection.json绑定图SHA；不修改标签、阶段、预测或完整人口。0187第一遍无触发而固定位置恢复111TP；0171第二遍162个正确静稳日被改成未知；0174第二遍有弱尾收益，第一遍却因活动/静稳弃判净损失；0177失败恢复也伴随静稳弃判。因而F1上升不能单独证明可靠性改善。

[最近邻实际阅读记录](../feedback-placement-literature-v1/reading-notes.md)说明CGTime已有计算事实辅助语言、ORCA已有轻量统计选上下文。本研究不以数值反馈或Agent包装主张首创，未运行作者模型；统计阈值和注意力实现差异未移植为GNSS机制。

原report.md及其manifest.json原字节保留；此完整报告是追加读图及文献说明的派生版本。原主确认NOT_CONFIRMED与本候选STOP_FEEDBACK_PROMOTION保持。
"""
    complete_report = original_report + additions
    (REPORT / "report-complete.md").write_text(complete_report, encoding="utf-8")
    canonical = complete_report
    for cid in ("case_0171", "case_0174", "case_0177", "case_0187"):
        canonical = canonical.replace(
            f"({cid}-aligned.png)",
            f"(../artifacts/landslide-adaptive-review-2026-10-09/feedback-placement-report-v1/{cid}-aligned.png)",
        )
    canonical = canonical.replace(
        "(inspection.md)",
        "(../artifacts/landslide-adaptive-review-2026-10-09/feedback-placement-report-v1/inspection.md)",
    ).replace(
        "(../feedback-placement-literature-v1/reading-notes.md)",
        "(../artifacts/landslide-adaptive-review-2026-10-09/feedback-placement-literature-v1/reading-notes.md)",
    )
    (ROOT / "docs/landslide-feedback-placement-results.md").write_text(canonical, encoding="utf-8")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8").replace(
        "完整中文初稿v39", "完整中文初稿v40", 1
    )
    manuscript = manuscript.replace(
        "历史v38原字节完整存档。",
        "历史v38及本轮更新前v39原字节完整存档；新增同数值反馈的公平位置开发对照。",
        1,
    )
    abstract_note = "相同数值反馈的补充开发对照中，同次数位置有过程收益，但第一遍相对完整固定数值复核净损失131日，未升级反馈候选。"
    manuscript = manuscript.replace("**关键词：**", abstract_note + "\n\n**关键词：**", 1)
    manuscript = manuscript.replace(
        "## 1 引言",
        "A supplementary development control with identical numerical feedback supports a same-count placement benefit on the process target, but loses 131 definite correct days in the first repeat versus full fixed numerical review; the feedback candidate is not promoted.\n\n## 1 引言",
        1,
    )
    literature_note = "CGTime[20]已有确定性统计事实监督描述/QA，ORCA[21]已有轻量统计选择离散上下文。因此数值反馈或统计驱动的Agent控制不单独构成首创。本轮核对相关方法及ORCA作者代码，未运行作者模型；本研究差异与收益仍须由GNSS持续运动语义和统一位置对照约束。实际范围见[本轮最近邻阅读](../artifacts/landslide-adaptive-review-2026-10-09/feedback-placement-literature-v1/reading-notes.md)。\n\n"
    manuscript = manuscript.replace("## 3 任务定义", literature_note + "## 3 任务定义", 1)
    section = """### 6.4.4 同数值反馈的公平位置开发对照

原反馈消融只给自适应窗数值文字，固定窗为纯图，不能排除内容差异。补充实验仅给原固定两位置相同数值摘要，保持图、提示原字节、模型参数、共享初判、49日权限及阶段头；不新增机制或训练。全部24过程记录和预指定12作者记录继续评价。这批已参与方法选择，缓存自适应与新固定回答采集时间不同，服务漂移未排除，故只作开发证据。

| 过程人口 | 对照 | 自适应/对照F1 r0 | 净纠错 r0 | 自适应/对照F1 r1 | 净纠错 r1 |
|---|---|---|---:|---|---:|
| 完整24 | 同K早数值 | .835850/.811955 | +183 | .848253/.800690 | +171 |
| 各遍初判正常23 | 同K早数值 | .8408/.8343 | +79 | .8405/.8189 | +48 |
| 完整24 | 完整固定数值 | .835850/.829628 | −131 | .848253/.798659 | +153 |
| 各遍初判正常23 | 完整固定数值 | .8408/.8527 | −88 | .8405/.8170 | +140 |

过程同次数位置有两遍正F1与净纠错；完整固定比较第一遍却失败，正常初判人口第一遍F1也下降.0119。作者12条完整F1自适应.8926/.8987、固定.8441/.8202，净正确却−38/−166日；同K早作者净+140/−25日，排除初判失败后+11/−99。F1更高不能替代确定正确损失。准备请求前固定的开发行动条件因此给出**STOP_FEEDBACK_PROMOTION**；数值反馈不升主方法，原完整确认**NOT_CONFIRMED**不变，不选更好重复、子集或新seed救分。

实际查看四图并结合原始位移与逐日转换：0187第一遍零触发，自适应反馈无观察机会，相对固定丢111TP并改静稳；270—390日四个30日N块中位数2.22→3.01→4.74→5.63mm支持低幅累积。0171高位平台约62mm，第二遍将162个正确静稳日改未知，过程净−159。0174第二遍补弱尾127TP，第一遍失败恢复却伴70TP及200正确静稳日转未知，净−43。0177第二遍恢复199TP却有180正确静稳日转未知和10新增FP。这些是开发者评分后错误解释，不回填标签，也不能当独立观察验证。

新增74请求/74回执、367277报告token、190.938秒、0新任务失败及0重试，22相同回答复用。单遍过程自适应47/51调用、完整固定72/72、同K47/51；作者24/24对36/36。完整45/42方法保留原37/34全部结果与成本，新增控制评分逐字节离线重放一致。全部Precision/Recall/F1、事件、边界、阶段、成本及逐日转换见[同反馈位置完整结果](landslide-feedback-placement-results.md)。

"""
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript = manuscript.replace(
        "累计3651启动/3649回执、至少15799935报告token",
        "累计3725启动/3723回执、至少16167212报告token",
        1,
    )
    manuscript += """
[20] Feng, X., Xie, Y., Zhang, C., Li, R., Ling, W., Li, Z., & Liu, C. (2026). Decoupling Perception from Description: Computation-Grounded Representation Alignment between Multivariate Time Series and Language. [arXiv:2608.05238v2](https://arxiv.org/abs/2608.05238v2). 已读摘要、2.1—2.5与2.6开头；未完整读后续方法或全文，未确认作者代码，未运行模型。

[21] Roy, A., Singh, J., Bose, S., & Hota, C. (2026). Agentic Anomaly Detection with ORCA-Style Dynamic Inductive Bias Adaptation in Multimodal Wearable Time Series Data. [arXiv:2608.08859v1](https://arxiv.org/abs/2608.08859v1). 已读完整第4节及[作者仓库](https://github.com/bananabicycle/ORCA-Agentic)统计选k与相对偏置相关代码，树SHA ba5d2b8060ef72dee86b538f03f1fc57f340b00c；未执行作者模型或移植阈值。
"""
    MANUSCRIPT.write_text(manuscript, encoding="utf-8")
    (OUT / "manuscript-v40.md").write_bytes(MANUSCRIPT.read_bytes())
    update = """## 2026-10-11：同数值反馈的公平位置开发对照，初稿v40

[完整结果](landslide-feedback-placement-results.md)。本轮仅补齐原固定位置的相同数值反馈，不改提示、预算、49日权限、模型、参考、权重或seed。24过程/12作者全部保留，原37/34配置全部结果及成本不变，新增8控制，共45/42。新74请求/74回执、367277报告token、190.938秒、0新失败/0重试，复用22相同回答；累计3725启动/3723回执、至少16167212报告token，受控192/半合成60不变。

过程同K早相对净+183/+171日，正常初判敏感性+79/+48，F1两遍提高；但第一遍对完整固定数值净−131，正常初判F1差−.0119、净−88。作者对完整固定两遍净−38/−166，对同K早+140/−25。准备请求前条件未全部通过，STOP_FEEDBACK_PROMOTION；原核心NOT_CONFIRMED不变，不升反馈候选、不换seed/重复/子集救分。自适应缓存与新固定采集时间不同，服务漂移未排除，本轮仅开发证据。

实际看4图并用原始数值/逐日掩码辅助复核：0187r0零触发丢111TP，0171r1将162正确静稳日至未知，0174/0177失败恢复也伴静稳弃判。无标签或预测回填。开发者具有历史/当前结果暴露，非独立专家或现场金标准。CGTime相关方法与ORCA第4节/作者相关源码核查显示数值事实与统计选择已有先例；未运行作者模型，创新表述收窄到GNSS冲突位置语义与统一对照。

v39主稿及三状态文档在更新前精确存档；旧delivery manifest原字节保留，旧可变文档通过档案映射核验。v40公式继续Markdown LaTeX。研究goal仍active，未完成独立现场参考验证；下方为历史记录。

"""
    for path in STATE_DOCS:
        old = path.read_text(encoding="utf-8")
        title, body = old.split("\n", 1)
        path.write_text(title + "\n\n" + update + body.lstrip("\n"), encoding="utf-8")
    write_json(
        OUT / "closure.json",
        {
            "status": "closed_development_evidence; research goal active",
            "decision": read_json(ANALYSIS / "decision.json"),
            "previous_delivery_files_verified": old_count,
            "old_manifest_kept_verbatim": True,
            "old_mutable_documents_archived_verbatim": archives,
            "old_method_and_results_changed": False,
            "new_labels": False,
            "independent_expert_or_field_validation": False,
        },
    )
    print("Wrote actual inspection, scoped literature, complete report and v40")


def finalize():
    archives = read_json(OUT / "v39-document-archives.json")
    old_count = verify_previous_delivery(archives)
    verify(BASE / "feedback-placement-development-v1")
    for folder in (ANALYSIS, BASE / "feedback-placement-replay-v1", REPORT, LITERATURE):
        verify_manifest(folder)
    math = read_json(OUT / "math-check-v1/formula-audit.json")
    if math["status"] != "passed" or math["source_sha256"] != sha(MANUSCRIPT):
        raise ValueError("Current manuscript math has not been verified")
    if sha(OUT / "manuscript-v40.md") != sha(MANUSCRIPT):
        raise ValueError("Manuscript snapshot differs")
    missing_links = []
    for document in (MANUSCRIPT, ROOT / "docs/landslide-feedback-placement-results.md"):
        text = document.read_text(encoding="utf-8")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
            raise ValueError("Control character in " + str(document))
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target and not (document.parent / target.split("#")[0]).exists():
                missing_links.append(target)
    if missing_links:
        raise ValueError("Broken document links: " + repr(missing_links))
    checked_days, zero_review_paths = 0, 0
    selections = read_json(BASE / "core-lockbox-confirmation-v1/selections.json")
    for target in ("primary_process", "secondary_developer"):
        previous = read_json(BASE / "core-lockbox-analysis-v1" / target / "results.json")
        results = read_json(ANALYSIS / target / "results.json")
        if any(results[name] != result for name, result in previous.items()):
            raise ValueError("Old result dictionary changed")
        previous_rows = {
            row["method"]: row
            for row in csv_rows(BASE / "core-lockbox-analysis-v1" / target / "summary.csv")
        }
        current_rows = {row["method"]: row for row in csv_rows(ANALYSIS / target / "summary.csv")}
        if any(current_rows[name] != row for name, row in previous_rows.items()):
            raise ValueError("Old summary or cost fields changed")
        predictions = read_json(ANALYSIS / target / "predictions.json")
        for method, cases in predictions.items():
            if not method.startswith(
                (
                    "fixed_numeric_feedback",
                    "same_count_fixed_early_numeric",
                    "same_count_fixed_late_numeric",
                )
            ):
                continue
            for cid, prediction in cases.items():
                _, feature = load_case_features(BASE / "core-lockbox-data-v1", cid)
                activity = np.array(prediction["activity"])
                stage = np.array(prediction["stage"])
                if np.any(activity[~feature.observed] != -1) or np.any(
                    stage[~feature.observed] != -1
                ):
                    raise ValueError("Missing observation received definite output")
                if np.any((stage > 0) & ((activity != 1) | ~feature.supported)):
                    raise ValueError("Stage outside supported activity")
                checked_days += len(activity)
                if method.startswith("same_count_"):
                    rep = int(re.search(r"_r([01])__", method).group(1))
                    if not selections[f"{cid}-r{rep}"]["selected"]:
                        initial = predictions[f"direct_visual_r{rep}"][cid]
                        if (
                            prediction["activity"] != initial["activity"]
                            or prediction["stage"] != initial["stage"]
                        ):
                            raise ValueError("K=0 does not inherit direct prediction")
                        zero_review_paths += 1
        exports = csv_rows(REPORT / (target + "-additional-daily.csv"))
        for row in exports:
            prediction = predictions[row["method"]][row["case_id"]]
            day = int(row["day"])
            if (
                int(row["activity"]) != prediction["activity"][day]
                or int(row["stage"]) != prediction["stage"][day]
            ):
                raise ValueError("Daily export differs from prediction")
        if len(exports) != 8 * len(next(iter(predictions.values()))) * 730:
            raise ValueError("Incomplete additional daily export")
    write_json(
        OUT / "validation.json",
        {
            "status": "passed",
            "previous_delivery_verified_via_exact_archives": old_count,
            "original_results_and_summary_cost_fields": "all unchanged",
            "additional_prediction_days_checked": checked_days,
            "missing_activity_stage_and_supported_activity_gate": "passed",
            "zero_review_inheritance_paths": zero_review_paths,
            "daily_exports": "all equal to frozen predictions",
            "offline_replay": read_json(REPORT / "replay-audit.json"),
            "local_document_links": "passed",
            "latex": {key: math[key] for key in ("status", "inline_formulas", "display_formulas")},
            "readability_review": "Complete three new experiment scripts and affected call sites reviewed; frozen sources left unchanged",
        },
    )
    write_json(
        REPORT / "delivery-manifest.json",
        {
            "files": {
                path.name: sha(path)
                for path in REPORT.iterdir()
                if path.is_file() and path.name != "delivery-manifest.json"
            }
        },
    )
    folders = [
        BASE / "feedback-contribution-development-v1",
        BASE / "feedback-placement-development-v1",
        ANALYSIS,
        BASE / "feedback-placement-replay-v1",
        REPORT,
        LITERATURE,
        OUT,
    ]
    paths = {path for folder in folders for path in folder.rglob("*") if path.is_file()}
    paths.update((MANUSCRIPT, *STATE_DOCS, ROOT / "docs/landslide-feedback-placement-results.md"))
    paths.update(
        ROOT / "scripts" / name
        for name in (
            "diagnose_landslide_feedback_contribution.py",
            "probe_landslide_feedback_placement.py",
            "report_landslide_feedback_placement.py",
            "close_landslide_feedback_placement.py",
            "verify_landslide_manuscript_math.cjs",
        )
    )
    destination = OUT / "delivery-manifest.json"
    paths.discard(destination)
    write_json(
        destination,
        {"files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}},
    )
    for path in paths:
        if sha(path) != read_json(destination)["files"][path.relative_to(ROOT).as_posix()]:
            raise ValueError("Delivery changed while binding")
    print(
        f"Verified delivery: {len(paths)} files; {checked_days} prediction days; {zero_review_paths} K=0 paths"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("close", "finalize"))
    args = parser.parse_args()
    if args.mode == "close":
        close()
    else:
        finalize()
