"""Record actual review, source-reading limits and completed pilot receipts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs


def record():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "evidence-pooling-session-v1"
    session.mkdir(exist_ok=False)
    encoded = base / "evidence-pooling-encoding-v1"
    run = base / "evidence-pooling-calibration-v1"
    report = base / "evidence-pooling-report-v1"
    predictions = read_json(run / "predictions.json")
    review_ids = read_json(report / "analysis-plan.json")["review_ids"]
    differences = {}
    for cid in review_ids:
        with np.load(encoded / "validation" / (cid + ".npz")) as bank:
            actual = bank["activity"]
        main = np.array(predictions["continuity_pool_numeric"][cid]["activity"])
        control = np.array(predictions["uniform_pool_numeric"][cid]["activity"])
        main_error = ((main != 1) & (actual == 1)) | ((main == 1) & (actual == 0))
        control_error = ((control != 1) & (actual == 1)) | ((control == 1) & (actual == 0))
        corrected = control_error & ~main_error
        damaged = main_error & ~control_error
        differences[cid] = dict(
            corrected_days=int(corrected.sum()), damaged_days=int(damaged.sum()),
            corrected_half_open_runs=runs(corrected), damaged_half_open_runs=runs(damaged),
            reference_unchanged=True, independent_review=False,
        )
    write_json(session / "numerical-review.json", differences)
    notes = """# SCVP实际视觉与数值复核

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Scope: 16TRAIN／8已暴露VALIDATION的事后开发诊断，24保留内容未加载。
- Actual visual source: Codex的view_image调用；不是独立专家或盲评，不修改标签。
- 已实际打开comparison.png、validation-guide-example.png、case_0390-raw-review.png、case_0389-raw-review.png四幅图。
- 选择规则：对均匀池化逐记录净纠错最差／最好，不事后选择得分较好的重复。

case_0390：原N向在约50—145日和400—500日出现两次连续下降，中间为平台；E/U向亦支持运动，但噪声更大。缺测断线保持。图上主较均匀扩大第一段两侧边界，第二段判定基本重合；评分为主10FP/10FN，均匀0FP/12FN，净−8日。目测活动范围仅为定位阅读区域，不作为新确定标注。数值辅助的准确纠错及损伤区间保存在numerical-review.json，半开区间以原0基日期表示。

case_0389：E向约230—400日持续上升，与约220日等孤立尖峰可区分，活动后位移保持平台；U向相反趋势，N向运动弱于E。主比均匀更早进入既有活动，评分9FP/0FN对8FP/9FN，净+8日。图上改善有视觉合理性，但同幅图不能证明新增视觉编码有因果贡献；跳变引导控制也得到相同总TP。没有从导数提前响应赋确定活动或A/S/D。

validation-guide-example：case_0385 day358原图左侧有一次偏移，随后平台；绝对变化权重集中左侧，连续权重分散且接近均匀。该示例说明机制没有把一次幅度变化直接解释为持续活动，不能替代识别效果。其N/E/U同时显示原日位置，图像未裁掉平台或缺测。最终token已经混合上下文，权重图不是原日期的因果归因。

comparison：主F1未超过三个池化控制，事件F1亦没有新增优势；数值MLP和原全局分布树均更强。纯干扰三条FP零与混合召回改善不能抵消总体损伤。多数验证记录连续池化与原global中位余弦0.997—0.999，有效token约61—63；case_0390为0.973／34。表征变化大小与失败原因未做因果干预，不调floor救分。

首次测试由上一段工具摘要记录：单位和float32误差1.192e−7超过默认容差；只改测试atol=2e−7，算法没有修改，随后3 passed。原始首次失败stdout未另存，不能称可逐字重放的日志。本次再次执行冻结测试，stdout及实际exit另存。

强数值收益保留；SCVP六条件3/6，固定版本停止。没有新样本、独立参考或新增语言调用。下一研究必须先在强数值仍错的真实观测区域证明局部视觉纠错价值，再设计有界复核；这不是已经成立的后续收益。
"""
    (session / "actual-visual-review.md").write_text(notes, encoding="utf-8")
    reading = """# 连续引导池化：实际文献阅读范围

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，继续既定研究。
- Source: 官方arXiv abs/html与作者固定commit源码；不是全文综述或全部作者模型复现。
- 三组官方检索HTTP200，四官方abs/html请求HTTP200；搜索卡片只用于定位。

MedTVL，arXiv2608.28605：完整摘要；HTML文本约1083起185行与1266起185行，覆盖3.3文本自适应引导、3.4实例MoE融合，直到3.5对比学习起始。它用时间／视觉双路径、文本引导及实例专家，不是本项目GNSS原连续位移加权。源码start2020/MedTVL，commit 4f05ba8ff55a8387b95dcee38b0e20a6c99aa1d7，完整静态读取README.md、Models/Medtvl/model.py、Config/model_config.json；导出SHA/URL见evidence-pooling-code-source-v1/source-ledger.json。没有读main/ts_encoder，没有安装或运行。代码可见两个分支调用gate_ts、另定义gate_img，gate-fusion路径双方使用combined；未核全运行配置，不据片段宣布作者算法错误。官方元数据列KDD2026，未另核出版者，因此正文按预印本引用。

VESTA，arXiv2606.00384v2：完整摘要；HTML文本约459起150行，覆盖第3节统计建模定义、3.1概览、Propose及Algorithm1起始。统计工具支持视觉模型拟合、批评和修订，不是GNSS异常识别基线。作者链接https://github.com/wrudman/VESTA已定位，未克隆、读源码或运行，未读完全部方法或实验。

机制差异与限制：本轮只改变原冻结曲线特征图的读取，使用原14日块的相邻同向变化引导，没有复杂MoE或动态工具生成。不称引导／池化／融合通用首创；六继续条件未通过，不用最近邻论文的报告成绩为本项目背书。

当前8验证记录已用于模型／阈值选择，每族仅1条，未独立确认。所有变化限于受控开发；不基于单次点估计声称稳定性、显著性、因果或真实场景泛化。保存全部342配置，不能以最优对照替换预设主方法；日、事件、边界、阶段及成本共同披露。样本日有强相关，不能将730日当730个独立样本扩大统计强度。
"""
    (session / "reading-notes.md").write_text(reading, encoding="utf-8")
    passport = session / "passport.md"
    passport.write_text(
        "# 连续位移证据引导视觉池化研究记录\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent。\n"
        "- Version Label: SCVP pilot-v1；complete manuscript-v10。\n"
        "- Scope: 16TRAIN／8暴露VALIDATION；原96×730记录不变，保留24内容未加载。\n"
        "- Outcome: 342配置、30统一方法；六条件3/6，STOP_EVIDENCE_POOLING_PILOT。\n"
        "- Verification: 24原证据重构、216辅助编码抽样、9选中模型52560概率与240输出／评分重放。\n"
        "- Reference: 受控过程参考与开发者实际AI看图；没有独立领域标签。\n"
        "- Goal active: 稳定新Agent/VLM机制及未参与选择的确认仍未完成。\n",
        encoding="utf-8",
    )
    test_command = [sys.executable, "-m", "pytest", "tests/test_landslide_evidence_pooling.py", "-q"]
    tested = subprocess.run(test_command, cwd=root, capture_output=True, text=True, timeout=120)
    (session / "tests-stdout.txt").write_text(tested.stdout + tested.stderr, encoding="utf-8")
    if tested.returncode:
        raise RuntimeError("Frozen business tests failed;preserve output")
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v10.md"
    math_output = base / "manuscript-math-v10"
    math_command = ["node", "scripts/verify_landslide_manuscript_math.cjs", str(draft), str(math_output)]
    math = subprocess.run(math_command, cwd=root, capture_output=True, text=True, timeout=60)
    (session / "math-stdout.txt").write_text(math.stdout + math.stderr, encoding="utf-8")
    if math.returncode:
        raise RuntimeError("Manuscript LaTeX validation failed;preserve output")
    entries = [{
        "kind": "initial_instructions",
        "user_words": "需要在组合一定的方法基础上加上一点本项目的创新来提高一些指标，调研论文找一些思路，主要是异常识别，当前的项目过于专注于弱运动和异常状态（a/s/d）了",
    }]

    def receipt(step, command, paths, tokens, exit_status=0, status="passed"):
        entries.append(dict(
            kind="tool_receipt", step=step, command=command,
            exit_status=exit_status, status=status, retries_used=0,
            input_sha256={str(p): sha(p) for p in paths}, gate_tokens=tokens,
        ))

    receipt(
        "pooling-tests-first", "Historical pytest failure retained from resumed tool summary;not reexecuted",
        [session / "actual-visual-review.md"],
        ["float32 unit-sum error1.192e-7;test-only atol repair2e-7;original stdout not saved"],
        exit_status=1, status="failed",
    )
    receipt("pooling-tests", subprocess.list2cmdline(test_command),
            [root / "tests/test_landslide_evidence_pooling.py", session / "tests-stdout.txt"],
            [tested.stdout.strip()[-180:]])
    receipt("pooling-encode", "encode_landslide_evidence_pooling.py --out evidence-pooling-encoding-v1",
            [encoded / "encoding-freeze.json"], ["8526forwards;25578pool vectors;old global encodings exact"])
    receipt("pooling-train", "train_landslide_evidence_pooling.py --out evidence-pooling-calibration-v1",
            [run / "freeze.json"], ["342configurations;18trained models;30methods;STOP3/6"])
    receipt("pooling-replay", "verify_landslide_evidence_pooling.py --out evidence-pooling-replay-v1",
            [base / "evidence-pooling-replay-v1/freeze.json"],
            ["24original evidence;216sampled pools;9models52560probabilities240outputs exact"])
    receipt("pooling-report", "report_landslide_evidence_pooling.py --out evidence-pooling-report-v1",
            [report / "freeze.json", session / "actual-visual-review.md", session / "numerical-review.json"],
            ["30method scores;8family errors;fourimages actually viewed;references not rewritten"])
    receipt("pooling-draft", "write_landslide_evidence_pooling_manuscript.py",
            [draft, root / "scripts/write_landslide_evidence_pooling_manuscript.py"],
            ["Completev10;preservedv9;SCVPmethod/results/limitations;LaTeXmath"])
    receipt("pooling-math", subprocess.list2cmdline(math_command),
            [draft, math_output / "formula-audit.json"], [math.stdout.strip()])
    ledger = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite/ars/scripts/run_ledger.py")
    for index, entry in enumerate(entries):
        path = session / f"receipt-{index}.json"
        write_json(path, entry)
        subprocess.run([sys.executable, str(ledger), "append", "--passport-path", str(passport),
                        "--entry-file", str(path)], check=True, cwd=root, capture_output=True, text=True)
    steps = [entry["step"] for entry in entries if entry["kind"] == "tool_receipt"]
    claims = dict(decisions=[], expected_steps=steps,
                  steps=[dict(step=entry["step"], status=entry["status"])
                         for entry in entries if entry["kind"] == "tool_receipt"])
    write_json(session / "claims.json", claims)
    checked = subprocess.run([sys.executable, str(ledger), "report", "--passport-path", str(passport),
                              "--claims", str(session / "claims.json")],
                             check=True, cwd=root, capture_output=True, text=True)
    write_json(session / "handoff-check.json", json.loads(checked.stdout))
    print(tested.stdout.strip(), flush=True)
    print(math.stdout.strip(), flush=True)
    print(checked.stdout.strip(), flush=True)


if __name__ == "__main__":
    record()
