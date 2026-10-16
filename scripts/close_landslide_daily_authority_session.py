"""Bind completed receipts, verify the ARS ledger, and sync existing entry points."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gnss_sim.artifacts import sha, write_json


def close(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "daily-authority-session-v1"
    passport = session / "passport.md"
    python = root / "artifacts/reproduction-2026-10-07/environment/Scripts/python.exe"
    ars = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite")
    ledger = ars / "ars/scripts/run_ledger.py"
    entries = [dict(kind="initial_instructions", user_words="形变异常为主；仪器跳变、尖峰作为干扰单独识别")]
    steps = (
        ("authority-pilot", "probe_landslide_daily_revision_authority.py", "64methods512outputs;visual5/5both;numericSTOP;0newAPI", base / "daily-authority-probe-v1/freeze.json"),
        ("authority-pilot-replay", "verify_landslide_daily_revision_authority.py", "64methods512outputs exact;0newAPI", base / "daily-authority-replay-v1/freeze.json"),
        ("authority-expand", "run_landslide_daily_authority_expansion.py prepare/execute/score", "48TRAIN24exposedVALIDATION;114newrequests412485tokens;STOP_RCRA_FULL_DEVELOPMENT;no retries", base / "daily-authority-expansion-v1/analysis-v1/freeze.json"),
        ("authority-expand-replay", "verify_landslide_daily_authority_expansion.py", "36methods864outputs167usedrequests exact;114newrequests verified;0replayAPI", base / "daily-authority-expansion-replay-v1/freeze.json"),
        ("authority-tests", "pytest tests/test_landslide_daily_revision_authority.py tests/test_landslide_diversity_local_review.py tests/test_landslide_observed_duration.py -q", "9passed14upstreamwarnings;concentratedRuff passed", root / "tests/test_landslide_daily_revision_authority.py"),
        ("authority-report", "report_landslide_daily_authority.py", "Paired errors and raw figures;4actualimages inspected;no relabel", base / "daily-authority-report-v1/freeze.json"),
        ("authority-literature", "search/fetch_landslide_revision_authority sources", "3officialsearches6officialpages200;boundedactualmethod/code reading;no author models run", session / "reading-notes.md"),
        ("authority-draft", "write_landslide_daily_authority_manuscript.py", "Completev12;25evidencebindings;not independent effectiveness", base / "manuscript-v12/manifest.json"),
        ("authority-math", "node scripts/verify_landslide_manuscript_math.cjs", "53inline32display LaTeX KaTeX passed;acronym advisory retained", base / "manuscript-math-v12/formula-audit.json"),
    )
    for step, command, token, path in steps:
        entries.append(dict(kind="tool_receipt", step=step, status="passed", command=command,
                            exit_status=0, gate_tokens=[token], retries_used=0,
                            input_sha256={str(path.resolve()): sha(path)}))
    for index, entry in enumerate(entries):
        entry_path = session / f"receipt-{index}.json"
        write_json(entry_path, entry)
        result = subprocess.run([str(python), str(ledger), "append", "--passport-path", str(passport),
                                 "--entry-file", str(entry_path)], capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    claims = dict(decisions=[], expected_steps=[step[0] for step in steps],
                  steps=[dict(step=step[0], status="passed") for step in steps])
    write_json(session / "claims.json", claims)
    result = subprocess.run([str(python), str(ledger), "report", "--passport-path", str(passport),
                             "--claims", str(session / "claims.json")], capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    health = json.loads(result.stdout)
    write_json(session / "handoff-check.json", health)
    if health["ledger_status"] != "ok" or health["cannot_confirm"] or health["missing"]:
        raise ValueError("Session ledger cannot confirm completion receipts")
    note = """## 2026-10-13：RCRA减少误报，完整开发扩展未通过全部条件，最新稿v12

最新候选RCRA（Raw-Conflict Revision Authority，逐日冲突授权复核）组合全局VLM、原21／42日运动冲突及最多两次局部观察；仅实际原日冲突可修订，复用partial弃权处理，未重复宣称已有49日scope／反证／事件保护或Agent框架创新。同16TRAIN／8暴露VALIDATION小试视觉日F1 .898216/.897059，两遍各5/5；数值初判净损3/15日，停止。64方法512输出精确重放后扩展全部48／24，使用共同48TRAIN阶段头与18重训强基线；不拿16TRAIN头冒充公平扩展。

[完整结果](../artifacts/gnss-deformation-diversity-2026-10-12/daily-authority-report-v1/report.md)：主F1 .753963/.741811，直接.672774/.674897，固定.659599/.634132，49邻域.756993/.737977，强数值query_motion_mass .844167。误报547/483→194/188；相对整窗减少180/218FP也少110/111TP，净纠错70/107，第一遍F1略低于整窗／邻域。IoU.3 .848485/.823529与邻域一致，IoU.5 .666667/.647059低于邻域.787879/.705882。两遍4/5与5/5，STOP_RCRA_FULL_DEVELOPMENT；不改微小差距判据、不更名控制、不打开保留集救分。相对直接确有收益，不能写所有模块／事件／场景都改善。

新增114请求412485token、零传输／解析失败，原8回答复用；167使用请求绑定、36方法864输出及评分精确重放，重放新增API零。9业务测试通过、14上游弃用warning，集中Ruff通过。主解码前414/373日期变化、权限外传播4/0；不能称最终权限外绝不改变。实际看0401平台纠错、0399局部活动恢复受限，AI开发诊断不重标、不称独立专家。3官方检索／6官方原文200，实际读VLM4TS／Refining／DQS有关方法和已有固定commit VLM4TS源码，不冒称作者模型复现。

[完整初稿v12](gnss-deformation-manuscript-draft-2026-10-13-v12.md)新增权限LaTeX、完整强比较和消融、代价及取舍；53行内32显示公式KaTeX通过，缩写advisory保留，25证据绑定与ARS10条账本核验通过。96×730、48/24/24不变，保留24内容未加载，没有live实验进程，goal active。

下一步：先核对已有单向修订／事件保护，检验为何原冲突拒绝整窗可恢复活动，提出有明确差异的新增／撤回证据授权；用同回答小试及同日数控制验证，不直接改停止版邻域／阈值／提示救分。新机制通过开发条件后才冻结保留／新生成记录确认；稳定Agent/VLM贡献、独立评价及投稿条件尚未完成。

"""
    paths = (
        root / "docs/project-status.md",
        root / "docs/landslide-research-plan.md",
        root / "docs/landslide-submission-roadmap.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        first, remaining = text.split("\n", 1)
        path.write_text(first + "\n\n" + note + remaining.lstrip("\n"), encoding="utf-8")
    for filename in ("landslide-research-continuation-prompt.md", "landslide-manuscript-working-draft.md"):
        path = root / "docs" / filename
        text = path.read_text(encoding="utf-8")
        first, remaining = text.split("\n", 1)
        short = (
            "当前续接以[完整初稿v12](gnss-deformation-manuscript-draft-2026-10-13-v12.md)"
            "及[daily-authority-session-v1](../artifacts/gnss-deformation-diversity-2026-10-12/daily-authority-session-v1/passport.md)为准。"
            "RCRA小试两遍通过，完整48/24主F1 .753963/.741811、强数值.844167，"
            "第一遍未胜49邻域、严格事件受损，STOP_RCRA_FULL_DEVELOPMENT。"
            "114新请求、36方法864输出重放已完成，保留24未加载；"
            "先检验新增／撤回不同证据授权与已有机制差异，不改停止版救分。"
            "53行内32显示LaTeX通过，goal active，没有live进程。\n\n"
        )
        path.write_text(first + "\n\n" + short + remaining.lstrip("\n"), encoding="utf-8")
    path = root / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("最新研究状态："):
            lines[index] = (
                "最新研究状态：[完整初稿v12](docs/gnss-deformation-manuscript-draft-2026-10-13-v12.md)"
                "与[RCRA完整开发扩展](artifacts/gnss-deformation-diversity-2026-10-12/daily-authority-report-v1/report.md)。"
                "整体形变异常为主、跳变／尖峰独立识别；96×730、48训练／24暴露验证／24保留。"
                "主F1 .7540/.7418高于直接.6728/.6749，误报547/483→194/188，"
                "但第一遍未胜49邻域且IoU.5受损，强数值.8442；停止固定版，未成立独立稳定贡献。"
                "114新请求412485token，36方法864输出重放、85个LaTeX公式通过；24保留内容未加载。\n"
            )
            break
    else:
        raise ValueError("Existing README research entry missing")
    path.write_text("".join(lines), encoding="utf-8")
    write_json(session / "close-receipt.json", dict(
        status="progress", research_work_completed=True, goal_complete=False, live_processes=0,
        manuscript_sha256=sha(root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v12.md"),
        math_sha256=sha(base / "manuscript-math-v12/formula-audit.json"),
        ledger_health=health, reserved_content_loaded=False,
        corrected_readonly_command_errors="Initial .json passport path corrected to actual passport.md; missing guessed paths and literal forty in read commands corrected; no experimental state retry",
    ))
    write_json(session / "freeze.json", {p.name: sha(p) for p in session.iterdir() if p.is_file() and p.suffix != ".lock"})
    write_json(base / "manuscript-v12/close-freeze.json", {
        str(p.resolve()): sha(p) for p in (
            base / "manuscript-v12/manifest.json", base / "manuscript-math-v12/formula-audit.json",
            session / "freeze.json", Path(__file__),
        )
    })
    print(dict(ledger=health["ledger_status"], entries=health["entries"], completed_steps=health["backed"],
               next="Novel mechanism evidence before heldout", goal_complete=False), flush=True)


if __name__ == "__main__":
    close(Path(__file__).resolve().parents[1])
