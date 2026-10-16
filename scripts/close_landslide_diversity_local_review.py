"""Seal the local-observation feasibility run, actual review and complete v11 draft."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close():
    base = ROOT / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "diversity-local-review-v2"
    replay = base / "diversity-local-review-replay-v1"
    report = base / "diversity-local-review-report-v2"
    for folder, name in ((run, "prepared-freeze.json"), (run, "prediction-freeze.json"),
                         (run / "analysis-v1", "freeze.json"), (replay, "freeze.json"), (report, "freeze.json")):
        verify_manifest(folder, name)
    failed = base / "diversity-local-review-v1"
    failed_source = failed / "source-before-state-failure.py"
    if sha(failed_source) != read_json(failed / "protocol.json")["source_bindings"][str(ROOT / "scripts/run_landslide_diversity_local_review.py")]:
        raise ValueError("Failed executor snapshot differs from its prepared source")
    if list((failed / "requests").glob("*.started.json")):
        raise ValueError("The pre-request failure unexpectedly contains model calls")
    draft = ROOT / "docs/gnss-deformation-manuscript-draft-2026-10-13-v11.md"
    audit = base / "manuscript-math-v11/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Final draft differs from verified LaTeX")
    session = base / "diversity-local-review-session-v1"
    session.mkdir(exist_ok=False)
    actual = """# 多形态局部VLM复核的实际观察

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；用户授权自主研究。
- Scope: 八族16TRAIN／8已暴露VALIDATION，旧有界策略的可纠错价值检验。
- Actual images: Codex通过view_image实际打开comparison.png、case_0389-raw-review.png、case_0389-window-01.png。
- Reference: 受控过程标签不修改；开发者已知机制及历史结果，不是独立领域专家。

comparison图同时显示两遍直接、固定和自适应流程。自适应视觉第一遍日F1高于固定／直接，第二遍更低；强数值初判后的自适应复核虽更好于固定，仍低于原强数值。事件IoU.3保持1不能掩盖日识别和边界损失；不挑第一遍。

case0389原NEU图：E向在孤立尖峰后持续上升，约400日后转平台；原数值初判覆盖其活动，局部复核把开始推迟且把停止提前。实际局部图90—269日中，约218日尖峰很突出，E向末端持续累积仅数毫米；N/U向噪声较大。局部判断将该末端区域大部分判为非活动／未知，不能据此修改参考为正常；确切因果原因并未单独干预。原全局图依然作为上下文输入，不能将该失败简单归因为“没有全局上下文”。

读实际响应：0389-r0-window01返回activity空、uncertain250—269；r1窗口01返回activity空、uncertain258—269。两遍都描述E向末端弱趋势，但未给确定活动；窗口03第一遍确认270—415。像素看图不能单独确定真起止日，复核只用于解释冻结错误，不赋确定A/S/D。

精确追踪：相对数值初判，第一遍减少12FP、总撤回36TP、新得2TP，净少34TP、净纠错−22；第二遍减少12FP、总撤回42TP、新得3TP，净少39TP、净纠错−27。0388损伤／新增在边界相互抵消；主要净损来自0389。未知不计正确阴性，所有缺测断线保持。完整半开日区间、撤回到未知／阴性数量见report-v2/exact-error-traces.json。首次报告曾把TP净少写成总撤回，report-v1与未封稿文字已保留，v2修统计用语，推理／评分不变。

首版执行器在更新state时用独占新建函数而失败，API开始前退出；其原源码、prepared协议／图片保留于v1。新v2只改为现有save_json保存可更新状态并补源码快照，科学协议／提示／模型／选窗未改变，57唯一请求206254token，无传输／解析失败。8条两遍全部保留；2新失败及支撑契约＋5缺测持续期测试通过，14个上游弃用warning，不是新模型失败。

结论：停止固定旧复核在这组上的扩展，先前窄受控分布的收益仍保留适用边界。后续创新需要用新增证据支持局部修订、保护已有活动证据，不能只增加裁图／提示或无依据恢复旧预测。目前没有可确认的新机制或新独立标签，24保留内容未加载，goal active。
"""
    (session / "actual-visual-review.md").write_text(actual, encoding="utf-8")
    passport = session / "passport.md"
    passport.write_text(
        "# 多形态局部观察纠错价值研究\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent。\n"
        "- Version Label: diversity-local-review-v2 / complete manuscript-v11。\n"
        "- Scope: 原16TRAIN／8已暴露VALIDATION，未新增数据，24保留内容未加载。\n"
        "- Outcome: 五流程两遍，57请求206254token；STOP_DIVERSITY_LOCAL_REVIEW。\n"
        "- Verification: 全请求payload/images、初判/选窗/共同阶段、40方法320输出及评分精确重放。\n"
        "- Reference: 受控过程与开发者AI事后分析，没有独立领域标签。\n"
        "- Goal active: 稳定Agent/VLM方法与未参与选择的新记录确认尚未完成。\n",
        encoding="utf-8",
    )
    entries = [dict(kind="initial_instructions", user_words=
                    "形变异常为主；仪器跳变、尖峰作为干扰单独识别")]

    def receipt(step, command, paths, tokens, status="passed", exit_status=0):
        entries.append(dict(kind="tool_receipt", step=step, command=command, status=status,
                            exit_status=exit_status, retries_used=0, input_paths=[str(p) for p in paths],
                            gate_tokens=tokens))

    receipt("local-executor-first", "run_landslide_diversity_local_review.py execute --out diversity-local-review-v1",
            [failed_source, failed / "prepared-freeze.json"], ["FileExistsError:state.json;0APIrequests"],
            status="failed", exit_status=1)
    receipt("local-execute", "run_landslide_diversity_local_review.py execute --out diversity-local-review-v2",
            [run / "prediction-freeze.json"], ["57requests206254tokens;all16record repeats;0transport/parser failures"])
    receipt("local-score", "run_landslide_diversity_local_review.py score --out diversity-local-review-v2",
            [run / "analysis-v1/freeze.json"], ["40methods320outputs;STOP;both repeats retained"])
    receipt("local-replay", "verify_landslide_diversity_local_review.py --out diversity-local-review-replay-v1",
            [replay / "freeze.json"], ["57payload/image/selection/stage;40methods320outputs/scores exact;no newAPI"])
    receipt("local-tests", "python -m pytest tests/test_landslide_diversity_local_review.py tests/test_landslide_observed_duration.py -q",
            [ROOT / "tests/test_landslide_diversity_local_review.py", ROOT / "tests/test_landslide_observed_duration.py"],
            ["7passed14upstreamwarnings;2new failure/support contracts;5existing duration contracts"])
    receipt("local-draft", "report_landslide_diversity_local_review.py --out diversity-local-review-report-v2",
            [report / "freeze.json", draft, session / "actual-visual-review.md"],
            ["Completev11;gross withdrawals36/42,newTP2/3,netTP loss34/39;v1 wording preserved"])
    receipt("local-math", "node scripts/verify_landslide_manuscript_math.cjs draft-v11 manuscript-math-v11",
            [draft, audit], ["48inline28display;KaTeX syntax/render passed"])
    ledger = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite/ars/scripts/run_ledger.py")
    for index, entry in enumerate(entries):
        path = session / f"receipt-{index}.json"
        write_json(path, entry)
        subprocess.run([sys.executable, str(ledger), "append", "--passport-path", str(passport),
                        "--entry-file", str(path)], check=True, capture_output=True, text=True)
    claims = dict(decisions=[], expected_steps=[entry["step"] for entry in entries[1:]],
                  steps=[dict(step=entry["step"], status=entry["status"]) for entry in entries[1:]])
    write_json(session / "claims.json", claims)
    checked = subprocess.run([sys.executable, str(ledger), "report", "--passport-path", str(passport),
                              "--claims", str(session / "claims.json")],
                             check=True, capture_output=True, text=True)
    write_json(session / "handoff-check.json", json.loads(checked.stdout))
    output = base / "manuscript-v11"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = [
        base / "manuscript-v10/manifest.json", run / "prepared-freeze.json", run / "prediction-freeze.json",
        run / "analysis-v1/freeze.json", replay / "freeze.json", report / "freeze.json", draft, audit,
        session / "actual-visual-review.md", session / "passport_run_ledger.yaml",
        ROOT / "scripts/run_landslide_diversity_local_review.py",
        ROOT / "scripts/verify_landslide_diversity_local_review.py",
        ROOT / "scripts/report_landslide_diversity_local_review.py", Path(__file__),
    ]
    write_json(output / "evidence-bindings.json", {str(p): sha(p) for p in evidence})
    write_json(output / "manifest.json", dict(status="complete_research_draft", source_sha256=sha(draft),
                                              contribution_validated=False, independent_expert_reference=False,
                                              reserved_content_loaded=False,
                                              decision=read_json(run / "analysis-v1/decision.json"),
                                              files={p.name: sha(p) for p in output.iterdir()}))
    entry = """## 2026-10-13：多形态局部VLM复核未获得稳定纠错价值，最新完整稿v11

SCVP停止后，已实际回到旧有界局部观察方法，完成同八族16TRAIN／8已暴露VALIDATION五流程两遍：直接视觉、固定视觉复核、冲突选窗视觉复核、强数值初判固定复核、强数值初判冲突选窗复核。原模型qwen3.8-flash／提示／180日图／最多两复核／原21/42日持续性选窗策略不变，所有方法共同14阳性观测／仅跨≤7原缺测解码及16TRAIN阶段头。实际57唯一请求206254token、0传输／解析失败，40统一方法。[完整报告](../artifacts/gnss-deformation-diversity-2026-10-12/diversity-local-review-report-v2/report.md)。

强数值初判F1 .945529，冲突选窗复核后.919094/.913232，事件均1；各减少12FP，却撤回36/42TP、新得2/3TP，净少34/39TP、净纠错−22/−27日。固定数值复核更差.853147/.875000。视觉直接.808470/.869565，固定.805970/.849558，自适应均.830769；只第一遍收益成立，第二遍低于两者。强数值复核各3请求对固定16，视觉自适应13/12对固定24，但省调用不能替代识别效果。STOP_DIVERSITY_LOCAL_REVIEW，停止本固定版本，不挑第一遍、不调策略／提示救分；不是新创新成立。

所有57请求的payload/图片绑定、强数值初判、选窗、共同阶段、40方法320输出和统一评分精确重放。实际打开比较图、0389原NEU与90—269日局部图：E末端少量持续趋势被复核判非活动／未知，且活动尾部提前撤回；参考不改，确切损伤半开区间及unknown/negative另存。[实际复核](../artifacts/gnss-deformation-diversity-2026-10-12/diversity-local-review-session-v1/actual-visual-review.md)。首次执行器更新state使用独占写入失败，0API；原v1源码／协议保留，新v2仅修状态进度持久化。首次报告将TP净少误写总撤回，v1报告及未封稿文字保留，v2按逐日追踪修统计用语，原推理／评分不变。2新契约＋5已有持续期测试通过，14上游弃用warning；集中Ruff通过。

[最新完整初稿v11](gnss-deformation-manuscript-draft-2026-10-13-v11.md)保留v10全文及SCVP/文献结果，补有界复核方法／两遍对照／撤回证据与成本；48行内28显示LaTeX通过，manuscript-v11完整证据绑定。当前多形态批次仍96×730、48/24/24，24保留输入／参考内容未加载，无新独立标签；AI事后看图是开发诊断。没有live进程，goal active，稳定Agent/VLM核心贡献仍缺。下一研究必须让新增观察证据支持局部修订并保护已有活动证据，先证明强数值仍错区域的纠错价值，不继续盲目加表示模块或扩无效工作流。

"""
    for name in ("project-status.md", "gnss-deformation-anomaly-method-directions-2026-10-12.md",
                 "landslide-research-plan.md", "landslide-submission-roadmap.md", "landslide-research-continuation-prompt.md"):
        path = ROOT / "docs" / name
        text = path.read_text(encoding="utf-8")
        if entry.splitlines()[0] in text:
            raise ValueError("Already synchronized: " + name)
        first, remaining = text.split("\n", 1)
        path.write_text(first + "\n\n" + entry + remaining.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("最新研究状态：")
    stop = text.index("\n\n", start)
    text = text[:start] + (
        "最新研究状态：[完整初稿v11](docs/gnss-deformation-manuscript-draft-2026-10-13-v11.md)与"
        "[多形态局部复核实验](artifacts/gnss-deformation-diversity-2026-10-12/diversity-local-review-report-v2/report.md)。"
        "整体形变异常为主、跳变／尖峰独立识别；当前批次96×730、48训练／24暴露验证／24保留。"
        "SCVP与旧有界局部复核均未获得稳定收益；强数值初判F1 .9455，VLM复核.9191/.9132。"
        "57请求206254token，40方法320输出重放通过，76个LaTeX公式通过；尚无稳定新Agent/VLM贡献或独立现场结论。"
    ) + text[stop:]
    readme.write_text(text, encoding="utf-8")
    write_json(session / "close-receipt.json", dict(manifest_sha256=sha(output / "manifest.json"),
                                                   six_entry_points_synchronized=True, source_sha256=sha(Path(__file__))))
    write_json(session / "freeze.json", {p.name: sha(p) for p in session.iterdir() if p.is_file()})
    print(checked.stdout.strip(), flush=True)
    print("Complete manuscript-v11 and local-review evidence sealed;six entry points synchronized", flush=True)


if __name__ == "__main__":
    close()
