"""Save research receipts and synchronize the latest manuscript entry points."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "signed-event-session-v1"
    passport = session / "passport.md"
    python = root / "artifacts/reproduction-2026-10-07/environment/Scripts/python.exe"
    ars = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite")
    ledger = ars / "ars/scripts/run_ledger.py"
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v13.md"
    for path, expected in read_json(base / "manuscript-v13/evidence-bindings.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Manuscript evidence changed: " + path)
    math = read_json(base / "manuscript-math-v13/formula-audit.json")
    if math["status"] != "passed" or math["source_sha256"] != sha(draft):
        raise ValueError("Matching successful math rendering required")
    entries = [dict(kind="initial_instructions", user_words="形变异常为主；仪器跳变、尖峰作为干扰单独识别")]
    steps = (
        ("signed-probe", "probe_landslide_signed_event_authority.py",
         "48methods1152outputs;STOP;0newAPI;fixed protocol retained", base / "signed-event-authority-probe-v1/freeze.json", 0),
        ("signed-replay", "verify_landslide_signed_event_authority.py",
         "48methods1152outputs exact;independent authority/seed/budget/abstention/decoder;inherited payload binding",
         base / "signed-event-authority-replay-v1/freeze.json", 0),
        ("signed-report", "report_landslide_signed_event_authority.py",
         "Completev2;v1 incomplete report interface failure retained;raw0395and0408 inspected;no relabel",
         base / "signed-event-authority-report-v2/freeze.json", 0),
        ("signed-tests", "pytest signed_event_revision and observed_duration -q",
         "9passed0.66s;source unchanged after test;focused Ruff passed",
         root / "tests/test_landslide_signed_event_revision.py", 0),
        ("local-motion-literature", "search/fetch_landslide_local_motion_sources.py",
         "3officialsearches4officialpages200;bounded actual reading;not systematic review",
         session / "reading-notes.md", 0),
        ("timerlm-code-access", "fetch_landslide_timerlm_code.py",
         "FirstGitHubAPI403rate limit;no source files/commit retrieved;not retried",
         base / "timerlm-code-index-v1/repository.json", 1),
        ("target-fidelity", "probe_landslide_visual_target_fidelity.py",
         "9628TRAIN14630VALIDATION;STOP_LINEAR_TARGET_EVIDENCE;no activity references used;weights replay1e-10",
         base / "visual-target-fidelity-v1/freeze.json", 0),
        ("signed-manuscript", "write_landslide_signed_event_manuscript.py",
         "Completev13;35evidencebindings;new negative mechanisms and bounded sources",
         base / "manuscript-v13/manifest.json", 0),
        ("signed-math", "node scripts/verify_landslide_manuscript_math.cjs",
         "59inline35display LaTeX KaTeX passed;not semantic formula validation",
         base / "manuscript-math-v13/formula-audit.json", 0),
    )
    for step, command, token, path, exit_status in steps:
        entries.append(dict(
            kind="tool_receipt", step=step, status="passed" if exit_status == 0 else "failed",
            command=command, exit_status=exit_status, gate_tokens=[token], retries_used=0,
            input_sha256={str(path.resolve()): sha(path)},
        ))
    for index, entry in enumerate(entries):
        path = session / f"receipt-{index}.json"
        write_json(path, entry)
        result = subprocess.run(
            [str(python), str(ledger), "append", "--passport-path", str(passport), "--entry-file", str(path)],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    claims = dict(
        decisions=[], expected_steps=[step[0] for step in steps],
        steps=[dict(step=step[0], status="passed" if step[4] == 0 else "failed") for step in steps],
    )
    write_json(session / "claims.json", claims)
    result = subprocess.run(
        [str(python), str(ledger), "report", "--passport-path", str(passport), "--claims", str(session / "claims.json")],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    health = json.loads(result.stdout)
    write_json(session / "handoff-check.json", health)
    if health["ledger_status"] != "ok" or health["cannot_confirm"] or health["missing"]:
        raise ValueError("Ledger cannot confirm research receipts")
    note = (
        "## 2026-10-13：SERA事件授权与视觉目标事实探针停止，完整稿v13\n\n"
        "按用户定义继续整体形变异常，跳变／尖峰另识别且可同现；A/S/D仍是子分析。"
        "已探索旧单向新增、事件保护、有符号块证据、窄／宽图、竞争曲线和PELT分段，"
        "不把已有实现再包装创新。\n\n"
        "SERA（Signed Event Revision Authority，方向分离事件授权）只新检验撤回由原日冲突、"
        "新增由局部阳性候选的有序累计证据和新增种子授权。"
        "[完整结果](../artifacts/gnss-deformation-diversity-2026-10-12/signed-event-authority-report-v2/report.md)"
        "主F1 .738574/.728938，对RCRA .753963/.741811；新增105/123FP、补47/58TP、"
        "撤24/19TP，净−82/−84。新增权限162/199日，主与whole／同预算center输出完全相同。"
        "IoU.3也下降，IoU.5只第二遍恢复条件通过；STOP_SIGNED_EVENT_AUTHORITY，"
        "不救门槛／提示或扩本版。原36＋12流程共48方法1152输出精确重放，0新API／记录，"
        "不是1152条数据。实际看0395起变前54FP／2TP、0408净0；参考未改。"
        "9业务测试及集中Ruff通过。\n\n"
        "读新MORA／TimeRLM官方摘要与限定方法、实验设置；3搜索／4原文200，"
        "TimeRLM源码API403保留未重试，未读作者源码或复现。"
        "[固定视觉目标事实探针](../artifacts/gnss-deformation-diversity-2026-10-12/visual-target-fidelity-v1/report.md)"
        "以9628TRAIN／14630开发查询、原512图表示重建六轴相邻变化，未读取活动标签。"
        "主MAE .899760高于zero .715288／mean .719693／shuffle .817881，"
        "平均R²−.038338，STOP_LINEAR_TARGET_EVIDENCE；导出权重1e-10重算通过。"
        "只否定固定线性读出，不是异常F1或全部视觉不可用。\n\n"
        "[完整初稿v13](gnss-deformation-manuscript-draft-2026-10-13-v13.md)包含真实负结果、"
        "全部强对照及适用限制，59行内35显示LaTeX通过，35证据绑定与ARS账本核验。"
        "96×730、48/24/24不变，保留24内容未加载，无live进程，goal active。"
        "下一研究先获得与旧方法不同且可验证的局部事实读取／定位依据，"
        "再冻结一个最小异常机制；不继续排列权限、重复旧块图或以开发集更名救分。"
        "尚未成立稳定Agent/VLM贡献或独立投稿效果结论。\n\n"
    )
    for filename in ("project-status.md", "landslide-research-plan.md", "landslide-submission-roadmap.md"):
        path = root / "docs" / filename
        first, remaining = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + note + remaining.lstrip("\n"), encoding="utf-8")
    short = (
        "当前续接以[完整初稿v13](gnss-deformation-manuscript-draft-2026-10-13-v13.md)和"
        "[signed-event-session-v1](../artifacts/gnss-deformation-diversity-2026-10-12/signed-event-session-v1/passport.md)"
        "为准。SERA F1 .738574/.728938低于RCRA、净损82/84，"
        "同预算center／whole输出相同，STOP；48方法1152输出重放完成。"
        "固定视觉目标事实MAE .899760高于零.715288、平均R²−.038338，也STOP。"
        "0新API／数据，59行内35显示LaTeX通过；保留24内容未加载。"
        "先获取独特局部证据价值，不重做已停止块图／分段／竞争／权限排列。"
        "goal active，无live进程，尚无独立稳定Agent/VLM贡献。\n\n"
    )
    for filename in ("landslide-research-continuation-prompt.md", "landslide-manuscript-working-draft.md"):
        path = root / "docs" / filename
        first, remaining = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + short + remaining.lstrip("\n"), encoding="utf-8")
    path = root / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("最新研究状态："):
            lines[index] = (
                "最新研究状态：[完整初稿v13](docs/gnss-deformation-manuscript-draft-2026-10-13-v13.md)"
                "与[SERA完整结果](artifacts/gnss-deformation-diversity-2026-10-12/signed-event-authority-report-v2/report.md)。"
                "整体形变异常为主、跳变／尖峰独立识别；96×730、48训练／24暴露验证／24保留。"
                "SERA F1 .7386/.7289对RCRA .7540/.7418净损82/84日；"
                "固定目标事实线性读出也停止。48方法1152输出重放、94个LaTeX公式通过，"
                "0新API／记录；保留24内容未加载，稳定Agent/VLM贡献未成立。\n"
            )
            break
    else:
        raise ValueError("README research entry missing")
    path.write_text("".join(lines), encoding="utf-8")
    write_json(session / "close-receipt.json", dict(
        status="research_progress", goal_complete=False, live_processes=0, ledger_health=health,
        manuscript_sha256=sha(draft), reserved_content_loaded=False, new_api_calls=0,
        implementation_review="Complete affected functions/callers read;frozen old mechanisms not modified",
        corrections="Initial verifier assumed an absent replay path;fixed to explicit bound replay CLI before successful run. "
                    "Initial report score_table call missed methods argument;incompletev1 preserved,completev2 generated. "
                    "One JS patch parse failed before any mutation. Read-only guessed paths corrected. "
                    "No model request,reference,method threshold or outcome retry.",
        optional_source_failure="TimeRLM GitHubAPI403 recorded,not retried;source not read",
    ))
    write_json(session / "freeze.json", {
        item.name: sha(item) for item in session.iterdir() if item.is_file() and item.suffix != ".lock"
    })
    write_json(base / "manuscript-v13/close-freeze.json", {
        str(item.resolve()): sha(item) for item in (
            base / "manuscript-v13/manifest.json", base / "manuscript-math-v13/formula-audit.json",
            session / "freeze.json", Path(__file__),
        )
    })
    verify_manifest(session, "freeze.json")
    print(dict(ledger=health["ledger_status"], entries=health["entries"], receipts_confirmed=health["backed"],
               goal_complete=False, manuscript=str(draft)), flush=True)


if __name__ == "__main__":
    close(Path(__file__).resolve().parents[1])
