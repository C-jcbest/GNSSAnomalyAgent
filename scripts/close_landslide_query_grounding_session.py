"""Record concrete QSG research receipts and update the current manuscript links."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "query-grounding-session-v1"
    passport = session / "passport.md"
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v14.md"
    python = root / "artifacts/reproduction-2026-10-07/environment/Scripts/python.exe"
    ars = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite")
    ledger = ars / "ars/scripts/run_ledger.py"
    old = base / "signed-event-session-v1"
    resume = subprocess.run(
        [str(python), str(ledger), "report", "--passport-path", str(old / "passport.md"), "--claims", str(old / "claims.json")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    old_health = json.loads(resume.stdout)
    if old_health["ledger_status"] != "ok" or old_health["cannot_confirm"] or old_health["missing"]:
        raise ValueError("Previous settled research receipts cannot be confirmed")
    write_json(session / "resume-handoff-check.json", old_health)
    for path, expected in read_json(base / "manuscript-v14/evidence-bindings.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Manuscript evidence changed: " + path)
    math = read_json(base / "manuscript-math-v14/formula-audit.json")
    if math["status"] != "passed" or math["source_sha256"] != sha(draft):
        raise ValueError("Matching current LaTeX rendering required")
    files = [
        "scripts/run_landslide_query_shape_grounding.py", "scripts/verify_landslide_query_shape_grounding.py",
        "scripts/report_landslide_query_shape_grounding.py", "scripts/write_landslide_query_grounding_manuscript.py",
        "scripts/close_landslide_query_grounding_session.py", "scripts/search_landslide_query_grounding_sources.py",
        "scripts/fetch_landslide_query_grounding_sources.py", "src/gnss_sim/landslide_query_shape_grounding.py",
        "tests/test_landslide_query_shape_grounding.py",
    ]
    lint = subprocess.run([str(root / ".venv/Scripts/ruff.exe"), "check", *files], cwd=root,
                          capture_output=True, text=True, encoding="utf-8")
    write_json(session / "focused-lint.json", dict(command=["ruff", "check", *files], exit_status=lint.returncode,
                                                 stdout=lint.stdout, stderr=lint.stderr))
    if lint.returncode:
        raise ValueError("Focused lint failed")
    steps = (
        ("qsg-training", "run_landslide_query_shape_grounding.py", "14models/266configs/STOP;training handle86609 exit0", base / "query-shape-grounding-v1/freeze.json"),
        ("qsg-replay", "verify_landslide_query_shape_grounding.py", "handle82000 exit0;14weights/266trials/600outputs exact;no optimizer retraining", base / "query-shape-grounding-replay-v1/freeze.json"),
        ("qsg-diagnostics", "report_landslide_query_shape_grounding.py", "25methods/eightfamilies/paired traces/zero and mean teacher/AP;no relabel", base / "query-shape-grounding-report-v1/freeze.json"),
        ("qsg-visual-reading", "view_image0403/0395", "actual rawNEU inspected;net-51/+14;posthoc developer analysis", session / "actual-visual-review.md"),
        ("qsg-literature", "search/fetch and bounded primary reading", "3official searches/4primary pages200;ViSTA/UTOPYA not reproduced", session / "reading-notes.md"),
        ("qsg-statistics", "ARS prescribed fallacy scan", "11/11 checked;descriptive exposed-development inference", session / "statistical-interpretation.md"),
        ("qsg-lint", "ruff focused check", "all affected source passed;9business tests already passed before training,no production changes since", session / "focused-lint.json"),
        ("qsg-manuscript", "write_landslide_query_grounding_manuscript.py", "completev14/43evidencebindings/current negative results", base / "manuscript-v14/manifest.json"),
        ("qsg-math", "node verify_landslide_manuscript_math.cjs", "65inline/37display passed;missing slash corrected by source review,syntax alone insufficient", base / "manuscript-math-v14/formula-audit.json"),
    )
    entries = [dict(kind="initial_instructions", user_words="形变异常为主；仪器跳变、尖峰作为干扰单独识别")]
    for step, command, details, path in steps:
        entries.append(dict(kind="tool_receipt", step=step, status="passed", command=command,
                            exit_status=0, gate_tokens=[details], retries_used=0,
                            input_sha256={str(path.resolve()): sha(path)}))
    for index, entry in enumerate(entries):
        path = session / f"receipt-{index}.json"
        write_json(path, entry)
        subprocess.run([str(python), str(ledger), "append", "--passport-path", str(passport), "--entry-file", str(path)],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    claims = dict(decisions=[], expected_steps=[step[0] for step in steps],
                  steps=[dict(step=step[0], status="passed") for step in steps])
    write_json(session / "claims.json", claims)
    result = subprocess.run([str(python), str(ledger), "report", "--passport-path", str(passport),
                             "--claims", str(session / "claims.json")],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    health = json.loads(result.stdout)
    write_json(session / "handoff-check.json", health)
    if health["ledger_status"] != "ok" or health["cannot_confirm"] or health["missing"]:
        raise ValueError("Current research receipts cannot be confirmed")
    note = (
        "## 2026-10-13：QSG-VN局部图形监督完整试验停止，完整稿v14\n\n"
        "形变异常为主，仪器跳变／尖峰另识别且可同现；弱运动与A/S/D仍是子分析。"
        "QSG-VN只新增图像分支局部三组NEU图形变化监督，复用小CNN与原数值输入，"
        "不是语言VLM或Agent。[完整结果](../artifacts/gnss-deformation-diversity-2026-10-12/"
        "query-shape-grounding-report-v1/report.md)：主F1 .711368低于匹配去监督.744262、"
        "纯数值.792971与强query_motion_mass .844167。对plain去25FP／加3FP／补6TP／撤96TP，"
        "净−68日。7继续条件仅4通过，STOP_QUERY_SHAPE_GROUNDING，不调loss／seed／teacher救分。\n\n"
        "14模型、266校准机会、25方法600输出全部保存；72观测和教师、TRAIN尺度、全部权重／选择／"
        "概率／活动／阶段及评分精确重放，未重跑优化器。实际看0403较弱首段丢失净−51，"
        "0395少前扩净+14，参考未改。事实读出MAE .082160小于零.084492和TRAIN均值.084534，"
        "平均R² .047938，但AP主.765724仍低于plain .770000，不能把重建改善当检测贡献。"
        "9教师与缺测业务测试此前通过，完成后集中Ruff通过。\n\n"
        "定向读ViSTA／UTOPYA摘要与方法概览、复用VAN-AD规定段落；未复现作者模型。"
        "旧TimeRLM源码403仅是该轮访问限制，更早已读固定commit源码，历史记录不抹去。"
        "[完整初稿v14](gnss-deformation-manuscript-draft-2026-10-13-v14.md)保留历史负结果，"
        "65行内37显示公式LaTeX/KaTeX通过、43证据绑定，ARS指定11谬误全检查。"
        "96×730、48／24／24不变，0新API／记录／编码，24保留内容未加载。"
        "训练与核查句柄已exit0，无本轮live进程；goal active，稳定Agent/VLM贡献和独立验证未完成。\n\n"
    )
    for filename in ("project-status.md", "landslide-research-plan.md", "landslide-submission-roadmap.md"):
        path = root / "docs" / filename
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + note + rest.lstrip("\n"), encoding="utf-8")
    short = (
        "当前以[完整稿v14](gnss-deformation-manuscript-draft-2026-10-13-v14.md)及"
        "QSG完整结果为准。固定局部图形监督F1 .7114低于去监督.7443，净−68；"
        "虽平均图形R² .0479，未转为活动收益，STOP。14权重266校准600输出精确重放，"
        "65行内37显示LaTeX通过。96×730、48／24／24不变，保留内容未加载。"
        "不要重启训练、改停止版本或将较好控制更名创新；下一机制应先证明独特且可纠错的"
        "活动信息，不能重复已停止残差、空间读出、竞争拟合或权限排列。"
        "目标active，无本轮live进程，核心创新和独立效果尚未成立。\n\n"
    )
    for filename in ("landslide-research-continuation-prompt.md", "landslide-manuscript-working-draft.md"):
        path = root / "docs" / filename
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + short + rest.lstrip("\n"), encoding="utf-8")
    path = root / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("最新研究状态："):
            lines[index] = (
                "最新研究状态：[完整初稿v14](docs/gnss-deformation-manuscript-draft-2026-10-13-v14.md)"
                "与[QSG完整结果](artifacts/gnss-deformation-diversity-2026-10-12/query-shape-grounding-report-v1/report.md)。"
                "局部图形监督F1 .7114低于去监督.7443、强数值.8442，净−68日，固定版停止；"
                "14模型266配置600输出重放、102公式通过。96×730、48／24／24不变，"
                "保留内容未加载；稳定Agent/VLM贡献尚未成立。\n"
            )
            break
    else:
        raise ValueError("Current README research entry is missing")
    path.write_text("".join(lines), encoding="utf-8")
    write_json(session / "close-receipt.json", dict(
        status="research_progress", goal_complete=False, live_processes=0, ledger_health=health,
        manuscript_sha256=sha(draft), evidence_bindings=43, reserved_content_loaded=False, new_api_calls=0,
        review="Read affected complete functions/callers;preserved frozen training code and actual replay snapshot",
        construction_corrections="Replay E731 lambda changed to def after handle terminated. First draft equation lost backslashes "
        "during patch serialization;corrected source/draft/snapshot before final and checked rendered math again. "
        "Read-only CSV first lacked utf-8-sig,corrected;no experiment/reference/threshold change.",
        verification_limits="No optimizer retraining;9business tests inherited from pretraining completion,not rerun for report-only edits",
    ))
    write_json(session / "freeze.json", {item.name: sha(item) for item in session.iterdir()
                                         if item.is_file() and item.suffix != ".lock"})
    write_json(base / "manuscript-v14/close-freeze.json", {str(path.resolve()): sha(path) for path in (
        base / "manuscript-v14/manifest.json", base / "manuscript-math-v14/formula-audit.json", session / "freeze.json", Path(__file__),
    )})
    verify_manifest(session, "freeze.json")
    print(dict(ledger=health["ledger_status"], entries=health["entries"], receipts=health["backed"],
               goal_complete=False, manuscript=str(draft)), flush=True)


if __name__ == "__main__":
    close(Path(__file__).resolve().parents[1])
