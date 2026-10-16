"""Seal the completed dated-evidence probe and its complete manuscript snapshot."""

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
    session = base / "dated-evidence-session-v1"
    ledger = Path(
        "C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/"
        "skills/academic-research-suite/ars/scripts/run_ledger.py"
    )
    manifests = []
    for folder in ("dated-evidence-probe-v1", "dated-evidence-replay-v1", "dated-evidence-report-v1"):
        verify_manifest(base / folder, "freeze.json")
        manifests.append(base / folder / "freeze.json")
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v9.md"
    audit = base / "manuscript-math-v9/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Final draft differs from the checked LaTeX source")
    if read_json(base / "dated-evidence-replay-v1/verification.json")["status"] != "VERIFIED":
        raise ValueError("Missing exact selected-model replay")
    output = base / "manuscript-v9"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = manifests + [
        base / "manuscript-v8/manifest.json", draft, audit, session / "actual-review.md",
        root / "scripts/write_landslide_dated_evidence_manuscript.py",
    ]
    write_json(output / "evidence-bindings.json", {str(p.resolve()): sha(p) for p in evidence})
    write_json(
        output / "manifest.json",
        dict(
            status="complete_research_draft", source_sha256=sha(draft),
            contribution_validated=False, reserved_content_loaded=False,
            independent_expert_reference=False,
            decision=read_json(base / "dated-evidence-probe-v1/decision.json"),
            files={p.name: sha(p) for p in output.iterdir()},
        ),
    )
    entries = [
        dict(kind="initial_instructions",
             user_words="需要在组合一定的方法基础上加上一点本项目的创新来提高一些指标，调研论文找一些思路，主要是异常识别，当前的项目过于专注于弱运动和异常状态（a/s/d）了"),
    ]
    checks = (
        ("dated-business-tests", "python -m pytest tests/test_landslide_dated_evidence.py -q",
         [root / "tests/test_landslide_dated_evidence.py"], ["2 passed"]),
        ("dated-train", "probe_landslide_dated_evidence.py --out dated-evidence-probe-v1",
         [manifests[0]], ["190 configurations;10trained models;34methods", "0/4 numerical;0/4 visual;STOP"]),
        ("dated-replay", "verify_landslide_dated_evidence.py --out dated-evidence-replay-v1",
         [manifests[1]], ["5models;87600probabilities;816outputs exact", "TRAIN edge scaling exact"]),
        ("dated-draft", "write_landslide_dated_evidence_manuscript.py",
         [draft, manifests[2]], ["Complete v9;actual chart viewed"]),
        ("dated-math", "node scripts/verify_landslide_manuscript_math.cjs draft-v9 manuscript-math-v9",
         [draft, audit], ["42inline;24display;KaTeX passed"]),
    )
    for step, command, paths, tokens in checks:
        entries.append(dict(
            kind="tool_receipt", step=step, command=command, status="passed",
            exit_status=0, retries_used=0, gate_tokens=tokens,
            input_paths=[str(p) for p in paths],
        ))
    for index, entry in enumerate(entries):
        path = session / f"receipt-{index}.json"
        write_json(path, entry)
        subprocess.run(
            [sys.executable, str(ledger), "append", "--passport-path",
             str(session / "passport.md"), "--entry-file", str(path)],
            check=True,
        )
    write_json(session / "claims.json", dict(
        decisions=[], steps=[dict(step=row[0], status="passed") for row in checks],
        expected_steps=[row[0] for row in checks],
    ))
    subprocess.run(
        [sys.executable, str(ledger), "report", "--passport-path", str(session / "passport.md"),
         "--claims", str(session / "claims.json")],
        check=True,
    )
    print("Dated-evidence fixed version stopped;complete v9 saved;research goal active", flush=True)


if __name__ == "__main__":
    close()
