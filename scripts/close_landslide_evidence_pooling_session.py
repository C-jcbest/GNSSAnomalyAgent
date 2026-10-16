"""Seal the complete pooling study and synchronize its current entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "evidence-pooling-session-v1"
    manifests = []
    for name, manifest in (
        ("evidence-pooling-encoding-v1", "encoding-freeze.json"),
        ("evidence-pooling-calibration-v1", "freeze.json"),
        ("evidence-pooling-replay-v1", "freeze.json"),
        ("evidence-pooling-report-v1", "freeze.json"),
        ("evidence-pooling-literature-v1", "freeze.json"),
        ("evidence-pooling-primary-v1", "freeze.json"),
        ("evidence-pooling-code-source-v1", "freeze.json"),
    ):
        verify_manifest(base / name, manifest)
        manifests.append(base / name / manifest)
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v10.md"
    audit = base / "manuscript-math-v10/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Draft differs from the verified math source")
    replay = read_json(base / "evidence-pooling-replay-v1/verification.json")
    if replay["status"] != "VERIFIED":
        raise ValueError("Exact replay is required")
    output = base / "manuscript-v10"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = manifests + [
        base / "manuscript-v9/manifest.json", draft, audit,
        session / "actual-visual-review.md", session / "numerical-review.json",
        session / "reading-notes.md", session / "passport_run_ledger.yaml",
        root / "scripts/write_landslide_evidence_pooling_manuscript.py",
        root / "scripts/record_landslide_evidence_pooling_session.py",
        root / "scripts/close_landslide_evidence_pooling_session.py",
    ]
    write_json(output / "evidence-bindings.json", {str(p): sha(p) for p in evidence})
    write_json(output / "manifest.json", dict(
        status="complete_research_draft", source_sha256=sha(draft),
        contribution_validated=False, reserved_content_loaded=False,
        independent_expert_reference=False,
        decision=read_json(base / "evidence-pooling-calibration-v1/decision.json"),
        files={p.name: sha(p) for p in output.iterdir()},
    ))
    entry = """## 2026-10-13：连续位移引导池化未获总体增益，最新完整稿v10

已完成SCVP（Sustained-Change Visual Pooling，连续位移证据引导视觉池化）：用相邻同向14日块变化的调和量读取原冻结MobileCLIP2-S0特征图，原图、投影、全局512及共同数值保留。固定八族16TRAIN／8暴露VALIDATION，七神经／两树342配置，加21原基线共30统一方法；四主要融合臂1198维、同训练中心／12epoch／宽16或32，每个数值方法拿到同87信息。主日／事件F1 .920335/.888889；均匀.921081/.888889、打乱.922246/.888889、跳变.923239/.888889；数值MLP.932377/1、原全局分布树.945529/1。主多13TP也多16FP，对三个控制净−3/−4/−3日，六预设条件仅3项成立，STOP_EVIDENCE_POOLING_PILOT。混合召回.8100高于均匀.7692，纯干扰三条FP零；这些局部收益不能替代整体结论。不扩本版，不调floor／位置／epoch／seed或更名对照救分。[完整报告](../artifacts/gnss-deformation-diversity-2026-10-12/evidence-pooling-report-v1/report.md)。

24原记录／guide／三权重及TRAIN缩放逐值重构，216辅助编码抽样最大误差1.19e−7，9选中模型52560概率、30方法240输出及活动／事件／边界／阶段评分精确重放。编码准备8526前向、25578辅助向量、430.504秒，包含消融，不称部署成本；0新语言请求／新记录。3业务测试及8源文件集中Ruff通过。首次float32单位和容差测试失败及测试专属修复已披露，原始首次stdout未另存，不冒称逐字日志；当前成功stdout保留。实际查看比较图、guide图与最差／最好原NEU图0390／0389，净−8／+8日的精确变化区间另存。[实际复核](../artifacts/gnss-deformation-diversity-2026-10-12/evidence-pooling-session-v1/actual-visual-review.md)。最终token已混合上下文，不把8×8权重当精确日期或因果注意力。

已读MedTVL与VESTA官方摘要／限定方法段，静态完整读取MedTVL三个作者源码文件，commit 4f05ba8ff55a8387b95dcee38b0e20a6c99aa1d7，未运行作者模型；VESTA源码仅定位未读取。[阅读边界](../artifacts/gnss-deformation-diversity-2026-10-12/evidence-pooling-session-v1/reading-notes.md)。[最新完整初稿v10](gnss-deformation-manuscript-draft-2026-10-13-v10.md)保留v9全文与历史结果，增加SCVP方法／消融／负结果和最近邻，48行内／27显示LaTeX通过。manuscript-v10绑定冻结结果、阅读与实际复核，v9保持原字节。

主任务仍是整体形变活动，跳变／尖峰独立识别且可同现，A/S/D是活动内部子分析。当前多形态批次96×730、48／24／24不变，24保留输入／参考内容未加载；AI看图为开发者诊断，不称独立领域参考。没有live进程，goal active。数值分布有信息收益，但稳定Agent/VLM新贡献与新记录确认仍缺；下一研究先证明局部观察可以纠正强数值仍错的具体活动／边界，再设计有限复核，不继续增加没有纠错依据的全局表示变换。

"""
    for name in (
        "project-status.md", "gnss-deformation-anomaly-method-directions-2026-10-12.md",
        "landslide-research-plan.md", "landslide-submission-roadmap.md",
        "landslide-research-continuation-prompt.md",
    ):
        path = root / "docs" / name
        text = path.read_text(encoding="utf-8")
        if entry.splitlines()[0] in text:
            raise ValueError("Already synchronized: " + name)
        first, remaining = text.split("\n", 1)
        path.write_text(first + "\n\n" + entry + remaining.lstrip("\n"), encoding="utf-8")
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("最新研究状态：")
    stop = text.index("\n\n", start)
    text = (
        text[:start]
        + "最新研究状态：[完整初稿v10](docs/gnss-deformation-manuscript-draft-2026-10-13-v10.md)与"
        "[SCVP池化实验](artifacts/gnss-deformation-diversity-2026-10-12/evidence-pooling-report-v1/report.md)。"
        "整体形变异常为主，跳变／尖峰独立识别；当前多形态批次96条730日，48训练／24暴露验证／24保留。"
        "SCVP小试九臂342配置、30统一方法，日F1 .9203低于均匀.9211及强数值.9455，六条件3/6，固定版本停止。"
        "原证据、9选中模型及240输出／评分重放通过，75个LaTeX公式通过，尚无稳定新Agent/VLM贡献或独立现场结论。"
        + text[stop:]
    )
    readme.write_text(text, encoding="utf-8")
    # The manuscript binds the ledger before closing; avoid a circular hash.
    write_json(session / "close-receipt.json", dict(
        draft_sha256=sha(draft), manifest_sha256=sha(output / "manifest.json"),
        sources_and_results_verified=True, six_entry_points_synchronized=True,
        source=sha(Path(__file__)), reserved_content_loaded=False,
    ))
    write_json(session / "freeze.json", {p.name: sha(p) for p in session.iterdir() if p.is_file()})
    ledger = Path("C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite/ars/scripts/run_ledger.py")
    checked = subprocess.run([
        sys.executable, str(ledger), "report", "--passport-path", str(session / "passport.md"),
        "--claims", str(session / "claims.json"),
    ], check=True, capture_output=True, text=True)
    print(checked.stdout.strip(), flush=True)
    print("Complete manuscript-v10 sealed; six entry points synchronized; frozen evidence unchanged", flush=True)


if __name__ == "__main__":
    close()
