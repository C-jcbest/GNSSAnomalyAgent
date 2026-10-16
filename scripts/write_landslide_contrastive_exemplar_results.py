"""Preserve the stopped pilot and actual visual evidence before writing v50."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import SOURCE
from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from probe_landslide_visual_invariance import calibration_labels
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha

DOCUMENTS = (
    "landslide-adaptive-review-manuscript-draft",
    "landslide-adaptive-review-method",
    "landslide-submission-roadmap",
    "project-status",
)

# Retrospective observations after actual viewing and original block inspection.
# These notes never replace the frozen evaluation references.
VISUAL_NOTES = (
    "中心N小幅波动，E接近平稳；U在中心后约14日才出现显著阶跃。后续变化不能反向确认中心全活动，逐日起止待定。",
    "E由明显累计趋平台，中心前后仍有较小增加；N接近平台，U噪声较强。弱尾可能存在，但缩略图不足以确定逐日终点。",
    "中心E仍缓慢下降，N整体下降但局部不单调，U噪声较强。可见弱累计，不能直接从训练标签转移逐日边界。",
    "NE强活动主要在中心前，中心趋平台；U仍有缓慢变化但噪声大。潜在活动尾段未必清晰可见，不能据标签强行确认视觉证据。",
    "三个分量主要噪声起伏；中心U块值升后回落，NE无连续同向累计。不确认活动；未知是弃判，不是误报。",
    "N中心前负尖峰后回原水平，E局部起伏，U围绕平台波动。无清楚中心持续运动证据；未知是弃判。",
    "中心NE持续增加、U持续下降，原块差连续且显著。支持活动；不补造原支撑，不额外赋A/S/D。",
    "中心NE持续变化且幅度大，U总体下降，块差与图一致。支持活动；不由局部导数直接赋A/S/D。",
)


def nullable(array):
    values = np.asarray(array)
    if values.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in values]
    return [nullable(row) for row in values]


def preserve_previous(archive):
    previous = BASE / "manuscript-v49/delivery-manifest.json"
    files = read_json(previous)["files"]
    checks = {name: sha(ROOT / name) == expected for name, expected in files.items()}
    if not all(checks.values()):
        raise ValueError("v49 delivery differs before archival")
    archive.mkdir(parents=True, exist_ok=False)
    for name in DOCUMENTS:
        source = ROOT / "docs" / (name + ".md")
        (archive / (name + "-v49-original.md")).write_bytes(source.read_bytes())
    (archive / "v49-delivery-manifest-original.json").write_bytes(previous.read_bytes())
    export(archive / "v49-preservation-before.json",
           {"all_match": True, "count": len(checks), "checks": checks})
    return files


def supplement_review(run, analysis, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    selected = read_json(run / "selected-queries.json")
    packets = read_json(run / "packets.json")
    predictions = read_json(analysis / "predictions.json")
    pairs = {row["case_id"]: load_case_features(SOURCE, row["case_id"]) for row in selected}
    references = calibration_labels(SOURCE, pairs)
    truth, observations = {}, []
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        packet = packets[key + ":retrieved_contrast"]
        start, stop = packet["query"]
        case, _ = pairs[row["case_id"]]
        raw = np.array([value if value is not None else [np.nan] * 3
                        for value in case.displacement_mm], dtype=float)
        table = json.loads(packet["prompt"].split("Same-observation block measurements:")[1])[1]
        blocks = np.array(table["normalized_neu_blocks"], dtype=float)
        noise = np.array(table["noise_proxy_mm"])
        medians = []
        for slot in range(13):
            first = row["day"] - 91 + 14 * slot
            values = raw[first:first + 14]
            complete = np.isfinite(values).all(axis=1)
            median = np.median(values[complete], axis=0) if complete.sum() >= 8 else np.full(3, np.nan)
            medians.append(median)
        medians = np.array(medians)
        first_valid = np.flatnonzero(np.isfinite(medians).all(axis=1))[0]
        if not np.allclose((medians - medians[first_valid]) / noise, blocks, equal_nan=True):
            raise ValueError("Block table differs from original NEU medians")
        truth[key] = np.array([int(value["activity_label"])
                               for value in references[row["case_id"]][start:stop]])
        observations.append({
            "query_id": key, "case_id": row["case_id"], "indices_half_open": [start, stop],
            "dates_inclusive": [str(case.dates[start]), str(case.dates[stop - 1])],
            "image": str(run / (key + "-retrieved_contrast.png")),
            "image_sha256": sha(run / (key + "-retrieved_contrast.png")),
            "raw_neu_mm": case.displacement_mm[row["day"] - 91:row["day"] + 91],
            "block_table": table, "original_block_medians_mm": nullable(medians),
            "center_previous_change_noise_units": nullable(blocks[6] - blocks[5]),
            "center_next_change_noise_units": nullable(blocks[7] - blocks[6]),
            "visual_observation": VISUAL_NOTES[index],
            "reference_changed": False, "phase_assigned_by_reviewer": False,
        })
    export(out / "observations.json", {
        "reviewer": "Codex actual vision plus original numerical observations",
        "timing": "Retrospective after frozen outputs and scores; historical author exposure",
        "purpose": "Error diagnosis; not independent expert or new observation gold standard",
        "unique_pilot_query_images": 8, "raw_medians_reconstructed": 104,
        "observations": observations,
    })
    decomposition, common_success = [], []
    summary = csv_rows(analysis / "summary.csv")
    for method, cases in predictions.items():
        gained = lost_wrong = lost_unknown = evaluated = answered = correct = 0
        tp = fp = fn = 0
        for key, prediction in cases.items():
            actual = truth[key]
            known = actual >= 0
            initial = np.array(predictions["base_expert"][key]["activity"])
            after = np.array(prediction["activity"])
            gained += int((known & (initial != actual) & (after == actual)).sum())
            lost_wrong += int((known & (initial == actual) & (after >= 0) & (after != actual)).sum())
            lost_unknown += int((known & (initial == actual) & (after < 0)).sum())
            evaluated += int(known.sum())
            answered += int((known & (after >= 0)).sum())
            correct += int((known & (after == actual)).sum())
            # Exclude q1 only in a clearly labeled explanatory table, never the gate.
            if key != "q1":
                tp += int(((actual == 1) & (after == 1)).sum())
                fp += int(((actual == 0) & (after == 1)).sum())
                fn += int(((actual == 1) & (after != 1)).sum())
        row = {"method": method, "reference_known_days": evaluated, "answered_days": answered,
               "correct_days": correct, "gained": gained, "lost_to_wrong": lost_wrong,
               "lost_to_unknown": lost_unknown, "net": gained - lost_wrong - lost_unknown,
               "coverage": answered / evaluated, "covered_error_rate": (answered - correct) / answered}
        reported = next(value for value in summary if value["method"] == method)
        if row["net"] != int(reported["net_vs_expert"]) or not np.isclose(row["coverage"], float(reported["coverage"])):
            raise ValueError("Abstention decomposition differs from frozen score")
        decomposition.append(row)
        common_success.append({"method": method, "tp": tp, "fp": fp, "fn": fn,
                               "f1": 2 * tp / (2 * tp + fp + fn),
                               "scope": "Descriptive seven common-success queries; not selection or gate"})
    write_csv(out / "abstention-and-corrections.csv", decomposition)
    write_csv(out / "common-success-descriptive.csv", common_success)
    failures = []
    for request_id in read_json(run / "failures.json"):
        receipt = read_json(run / "requests" / (request_id + ".response.json"))
        payload = json.loads(receipt["output"])
        failures.append({"request_id": request_id, "label_count": len(payload["labels"]),
                         "original_output": receipt["output"], "fallback": "Frozen base_expert retained",
                         "no_retry": True})
    if [row["label_count"] for row in failures] != [13, 13]:
        raise ValueError("Failure diagnosis changed")
    export(out / "output-contract-failures.json", failures)
    export(out / "validation.json", {
        "verification": "Deterministic response replay in analysis-v1 and supplement checked",
        "new_api_calls": 0, "related_tests_passed": 22, "ruff_five_original_files": "passed",
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": {
            "Simpson": "Per-query/role results retained; no every-record claim",
            "ecological": "Local windows cannot establish complete events or field performance",
            "Berkson": "Prediction-disagreement selection disclosed; not random population",
            "collider": "No adjusted regression or causal inference; selected population",
            "base_rate": "58 active/54 inactive days; artificial selected prevalence",
            "regression_to_mean": "Same frozen observations and all four controls",
            "survivorship": "All64receipts and two failures retained in main8querygate",
            "look_elsewhere": "All arms/repeats/20conditions; large historical search disclosed",
            "forking_paths": "Original protocol and layout defect retained; no rescue",
            "causation": "Only main/no-numeric images identical; other controls layout-confounded",
            "reverse_causality": "Retrospective diagnosis cannot create independent truth",
        },
        "inference_limits": "No significance/CI/p-value or repeated-date independence claim",
    })


def write():
    run = BASE / "contrastive-exemplar-pilot-v2"
    analysis = BASE / "contrastive-exemplar-analysis-v1"
    archive = BASE / "manuscript-v50"
    review = BASE / "contrastive-exemplar-visual-review-v1"
    decision = read_json(analysis / "decision.json")
    if decision["decision"] != "STOP_CONTRASTIVE_EXEMPLAR_PILOT":
        raise ValueError("Writer documents only the stopped pilot")
    previous = preserve_previous(archive)
    supplement_review(run, analysis, review)
    mechanism = r"""
JUDO[37]的正常图比较包含定位监督、领域SFT及GRPO，AD-Copilot[38]针对独立编码图像的比较瓶颈引入可学习cross-attention；并排图或检索本身已有先例。本轮另冻一个有限假设：同时取中心活动/静稳训练实例，检验能否保留弱累计并排除阶跃与平台，不将检索框架作为创新。

沿用前节十三个14日原始块与噪声归一化 $z_j(t)$，保持NaN槽位：

$$
u_j=z_{j+1}-z_j,\quad \ell_j=\log(1+\|u_j\|_2),\quad j=0,\ldots,11,
\qquad c_j=\frac{u_j^\top u_{j+1}}{\|u_j\|_2\|u_{j+1}\|_2},\quad j=0,\ldots,10.
$$

零长度或缺槽对应余弦未知，不跨缺槽差分。将12长度与11余弦组成23维 $s(t)$，在共同可用坐标集合 $J$ 上比较：

$$
d(t,e)=\frac{1}{|J|}\sum_{k\in J}[s_k(t)-s_k(e)]^2,\qquad |J|\ge12,\quad \{5,6\}\subseteq J.
$$

中心前后两增量必须共同可用，距离是检索代理，不是活动概率；每类取一个最近训练例，同距按ID。原24训练、14日中心网格、至少八支撑日、中心已观测标签全同类、至少十可用块，得到874实例。固定对照按训练类坐标中位数选代表。训练标签来自原潜在过程，不是独立观测确认，活动尾段可能与噪声平台难区分。

验证查询只由冻结base18/base_expert预测提名：四条不同记录中至少七日分歧按概率差排序，另取两静稳/两活动一致控制，不用查询参考提名。八窗各14日、原182日观察；检索正反图＋块表、三列重复目标＋块表、固定正反图＋块表、检索同图去块表，四臂各两遍64请求，无重试。只允许原强数值初判已知且原支持的查询日修订，未知保留、窗外不改，共享原阶段头及支持，未新生成视觉A/S/D。
"""
    outcome = r"""
主两遍P/R/F1均.787879/.896552/.838710，低于base_expert .981818/.931034/.955752，也未超过目标图F1 .840000。各增加四个确定正确判断、失去47个：19变错误、28变未知，净−43日；仅一记录净受益。静稳一致控制28日弃判，覆盖由1降至.75；FP/FN为14/6，未知不能称误报。已知日正确数由107/112降至64/112，已回答错误率20/84；前者将弃判计为未获得确定正确判断，后者只描述已回答风险。

固定图及去数值两遍F1均.736842，最近类数值.820000。主弱R由.785714升至1，但q3三日补检在所有视觉臂出现，未证明检索必要。主/去数值八图字节相同，数值表使q2由全静稳变为前十日活动、保留十日弱尾；整体仍低于强数值。主组两遍输出相同，不将相关日期或重复调用当独立样本。20项必要条件仅8项通过，STOP_CONTRASTIVE_EXEMPLAR_PILOT，不调窗口、检索、seed、提示、查询或图像救分。

目标图q1两遍各13标签，契约失败回退原base_expert，无重试；回退使其免于主在q1的14日误报。主表保留全八查询与失败。共同成功七查询仅作描述：主F1 .945455、目标图.840000、强数值.955752；不改变登记决定，不以删失败后的方法差异代替整体贡献。

实际八图＋原NEU/104块中位数复核：q1中心证据弱，显著U阶跃在中心后，不能反传全活动；q2活动趋平台，数值表保留部分弱尾而终点待定；q3可见E弱累计，修正各视觉臂共有；q4近平台，全静稳修一误报/损三活动日，潜在尾段视觉不清；q5/q6主要噪声波动，主未知造成覆盖损失而非活动误报；q7/q8显著累计，各臂保持活动。作者已见评分及历史标签，此为事后诊断，不新增或追改可信参考，不赋确定A/S/D。

像素核查否定原所有目标像素相同声明：tight_layout随训练面板刻度改变目标位置；目标图仅q2/q5/q8中央裁图相同，固定图八对均不同，主/去数值八整图相同。原数据、日期、归一化尺度保持，但其他差异不能全归检索。保留错误协议、原图与调用，不偷偷修图重跑。事件IoU .3/.5及边界仅14日裁窗，截断端点不是完整事件或物理起止；内部/端到端阶段来自共享数值头的活动门控。

64启动/64回执、247224报告token，API108.404秒、推理循环109.435秒（非完整预处理/端到端）；62合法/两格式失败。874实例、36原输入、邻居排名、64payload与预测重放、46720日历权限/缺测/阶段及sklearn/事件边界核验通过；22相关测试、五新增文件Ruff通过。准备v1在源码存档撞Windows长路径、0API，v2改浅层编号；39协议/图/表/调度文件字节相同，失败版保留。
"""
    raw_report = (analysis / "results.md").read_text(encoding="utf-8")
    table = raw_report[raw_report.index("| 方法"):raw_report.index("\n{\n")]
    result = ("# 平衡中心正反训练图检索：冻结复核与停止结果\n\n"
              "## Material Passport\n\n"
              "- Origin Skill: academic-research-suite / experiment-agent\n"
              "- Origin Mode: run / validate inline（沿用用户自主实验授权）\n"
              "- Origin Date: 2026-10-11\n"
              "- Verification Status: 冻结请求与预测重放通过，八图原数值实际复核；非独立评价\n"
              "- Version Label: contrastive-exemplar-pilot-v2 / analysis-v1 / manuscript-v50\n\n"
              + mechanism.strip() + "\n\n" + table + "\n\n" + outcome.strip()
              + "\n\n## 产物与复现\n\n"
              "训练库/提名/图/提示/64调用：contrastive-exemplar-pilot-v2；"
              "全指标与20条件：contrastive-exemplar-analysis-v1；八图复核、原NEU/块重建、弃判分解、"
              "失败原文及11项推断检查：contrastive-exemplar-visual-review-v1。旧参考和结果不改。\n\n"
              "```powershell\n"
              "$env:PYTHONPATH='src;scripts'\n"
              "$env:PYTHONIOENCODING='utf-8'\n"
              "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/report_landslide_contrastive_exemplars.py --run artifacts/landslide-adaptive-review-2026-10-09/contrastive-exemplar-pilot-v2 --out artifacts/landslide-adaptive-review-2026-10-09/contrastive-exemplar-user-replay\n"
              "```\n\n"
              "选不存在的新报告目录，无API重放；不要重启已完成infer。此小试不替代原完整规则/PELT/XGB/"
              "直接VLM/固定Agent比较。累计3959启动/3957回执、至少17100811报告token，受控192/半合成60不变。"
              "goal active，稳定Agent/VLM贡献与独立可信确认未完成。后续须改变有依据的识别/学习机制，"
              "而非扩展示例、提示或检索模块；通用数值合并仍是必要强控制，CPU环境不冒称完成VLM微调。\n")
    (ROOT / "docs/landslide-contrastive-exemplar-results.md").write_text(result, encoding="utf-8")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8").replace("完整中文初稿v49", "完整中文初稿v50", 1)
    manuscript = manuscript.replace(
        "更新前v48主稿及三状态文档原字节存档。新增有限训练图特征发现、三输入臂两遍固定池、强数值同预算比较及四图视觉数值复核；固定候选停止。",
        "更新前v49主稿及三状态文档原字节存档。新增平衡中心正反训练图检索、四臂两遍64请求、强数值比较及八图视觉数值复核；检索候选停止，弃判与格式失败单独披露。", 1)
    marker = "### 6.5 阶段与计算成本"
    if manuscript.count(marker) != 1:
        raise ValueError("Manuscript insertion point differs")
    addition = ("\n### 6.4.14 平衡中心正反训练图检索与数值反馈消融\n\n"
                + mechanism.strip() + "\n\n" + table + "\n\n" + outcome.strip()
                + "\n\n详见[完整结果及复现](landslide-contrastive-exemplar-results.md)。\n\n")
    manuscript = manuscript.replace(marker, addition + marker, 1).replace(
        "截至v49累计3895启动/3893回执、至少16853587报告token",
        "截至v50累计3959启动/3957回执、至少17100811报告token", 1)
    manuscript += ("\n[37] Kang, H., Lee, W., Kim, J., & Park, H. (2026). JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA. "
                   "[arXiv:2605.20284v1](https://arxiv.org/abs/2605.20284v1). [作者源码](https://github.com/woodavid31/JUDO)。"
                   "读取摘要及3.1—3.2相关段、README/树和固定SHA70194e1ceda3e9dbaea10e9f96c6f56513c90970的"
                   "eval/eval_seg_mult.py前12个图像/模板/提示命中邻近行；未完整复现或运行作者权重。\n\n"
                   "[38] Jiang, X., Guo, Y., Li, J., Liu, Y., Gao, B.-B., Deng, H., Liu, J., Zhao, H., Wang, C., & Zheng, F. (2026). "
                   "AD-Copilot: A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison. "
                   "[arXiv:2603.13779v2](https://arxiv.org/abs/2603.13779v2). [作者仓库](https://github.com/jam-cc/AD-Copilot)。"
                   "读取摘要、IV-A瓶颈及IV-B编码器相关段；取得head/tree（SHAcaf978a90abbc99095d62cfb6043ac31fa4163d0），"
                   "源码SSL获取失败保留，未读作者实现或运行其模型。\n")
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = ("\n## 2026-10-11：平衡中心正反训练图小试停止，初稿v50\n\n"
              "[完整结果](landslide-contrastive-exemplar-results.md)。874训练例、预测提名八14日窗、四臂两遍64请求。"
              "主F1 .838710低于强数值.955752和目标图.840000；净−43确定正确日（四增/19变错/28变未知），"
              "覆盖.75，仅一记录受益；弱R升至1，q3三日补检各视觉臂共有。数值表保留q2十日弱尾。"
              "8/20条件，STOP_CONTRASTIVE_EXEMPLAR_PILOT，不救分。\n\n"
              "八图/104块及NEU实际复核已保存；后续阶跃不得反传中心，弃判不得记活动误报。"
              "目标图q1两遍13标签失败回退、无重试，主表全保留，共同成功描述不改gate。"
              "主/去数值整图相同，其他对照自动排版像素变化；原错误协议、图、调用保留。"
              "作者事后暴露、局部边界与潜在标签限制明确，参考未改。874例/36输入/64payload及预测/"
              "46720日历/sklearn及事件核验、22测试/Ruff通过。\n\n"
              "新增247224token/API108.404秒/循环109.435秒；累计3959启动/3957回执、至少17100811token，"
              "受控192/半合成60不变。v49四文档原字节及392旧文件映射保留，公式LaTeX。"
              "JUDO/AD-Copilot阅读范围与源码失败披露。goal active，稳定贡献与独立可信确认未完成；"
              "后续应改变有依据的识别/学习机制，停止低价值示例/提示扩展，更强通用数值合并仍须比较。\n\n")
    for name in DOCUMENTS[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n" + status + rest, encoding="utf-8")
    mapping = {f"docs/{name}.md": archive / (name + "-v49-original.md") for name in DOCUMENTS}
    checks = {name: sha(mapping.get(name, ROOT / name)) == expected for name, expected in previous.items()}
    if not all(checks.values()):
        raise ValueError("Historical delivery changed outside archived documents")
    export(archive / "v49-preservation-after.json", {
        "all_match": True, "count": len(checks), "checks": checks,
        "archived_document_mapping": {key: str(value) for key, value in mapping.items()},
    })
    export(archive / "progress.json", {
        "status": "goal_active", "new_api_calls": 64, "new_reported_tokens": 247224,
        "decision": decision["decision"], "actual_unique_pilot_visual_reviews": 8,
        "independent_confirmation": False, "new_records": 0, "formula_format": "MarkdownLaTeX",
    })
    print("Wrote v50; preserved v49; completed eight visual/numerical diagnoses")


if __name__ == "__main__":
    write()
