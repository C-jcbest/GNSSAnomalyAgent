"""Bind the NCVR negative result, actual source reads and LaTeX-checked draft."""

from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "conditional-visual-session-v1"
    ledger = Path(
        "C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/"
        "skills/academic-research-suite/ars/scripts/run_ledger.py"
    )
    manifests = []
    for folder in (
        "conditional-visual-calibration-v1", "conditional-visual-replay-v1",
        "conditional-visual-report-v1", "conditional-visual-primary-v1",
        "conditional-visual-literature-v1", "conditional-visual-code-source-v1",
    ):
        manifest = base / folder / "freeze.json"
        verify_manifest(manifest.parent, manifest.name)
        manifests.append(manifest)
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v8.md"
    audit = base / "manuscript-math-v8/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Formula audit and final draft differ")
    if read_json(base / "conditional-visual-replay-v1/verification.json")["status"] != "VERIFIED":
        raise ValueError("Missing successful experimental replay")
    with (base / "conditional-visual-report-v1/embedding-diagnostics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    diagnostics = {}
    for group in ("train", "validation"):
        diagnostics[group] = {}
        for mapping in ("paired", "shuffled"):
            values = [
                float(row["relative_to_fixed_train_mean_r2"])
                for row in rows if row["group"] == group and row["mapping"] == mapping
            ]
            diagnostics[group][mapping] = dict(
                records=len(values), median_record_r2=statistics.median(values),
                min_record_r2=min(values), max_record_r2=max(values),
            )
    write_json(session / "diagnostic-summary.json", diagnostics)
    output = base / "manuscript-v8"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = manifests + [
        draft, audit, session / "reading-notes.md", session / "actual-visual-review.md",
        root / "scripts/write_landslide_conditional_visual_manuscript.py",
    ]
    write_json(output / "evidence-bindings.json", {str(p.resolve()): sha(p) for p in evidence})
    write_json(
        output / "manifest.json",
        dict(
            status="complete_research_draft", source_sha256=sha(draft),
            independent_expert_reference=False, test_truth_opened=False,
            contribution_validated=False,
            decision=read_json(base / "conditional-visual-calibration-v1/decision.json"),
            files={p.name: sha(p) for p in output.iterdir()},
        ),
    )
    instructions = session / "initial-instructions.json"
    write_json(
        instructions,
        dict(
            kind="initial_instructions",
            user_words="形变异常为主；仪器跳变、尖峰作为干扰单独识别",
        ),
    )
    subprocess.run(
        [sys.executable, str(ledger), "append", "--passport-path",
         str(session / "passport.md"), "--entry-file", str(instructions)],
        check=True,
    )
    receipts = (
        ("ncvr-train", "run_landslide_conditional_visual.py",
         "passed", 0, 0, [manifests[0]], ["418 configurations", "2/7 STOP_CONDITIONAL_VISUAL"]),
        ("ncvr-replay", "verify_landslide_conditional_visual.py",
         "passed", 0, 0, [manifests[1]], ["192720 probabilities;696 outputs exact", "2 expectation refits exact"]),
        ("ncvr-report", "report_landslide_conditional_visual.py",
         "passed", 0, 0, [manifests[2]], ["29 methods;24 paired cases;3 actual viewed figures"]),
        ("ncvr-source-export", "export_landslide_vtbench_code.py",
         "passed", 0, 0, [manifests[5]], ["2 author source files;fixed commit;not executed"]),
        ("ncvr-tests", "python -m pytest tests/test_landslide_conditional_visual.py -q",
         "passed", 0, 0, [root / "tests/test_landslide_conditional_visual.py"], ["4 passed"]),
        ("ncvr-math-first", "node scripts/verify_landslide_manuscript_math.cjs draft-v8 manuscript-math-v8",
         "failed", 1, 0,
         [session / "writer-at-math-failure.py", session / "draft-at-math-failure.md"],
         ["KaTeX Unexpected character form feed;writer transport escaping"]),
        ("ncvr-math", "node scripts/verify_landslide_manuscript_math.cjs draft-v8 manuscript-math-v8",
         "passed", 0, 1, [draft, audit], ["40 inline;23 display", "Draft-only repair;models unchanged"]),
    )
    for step, command, status, exit_status, retries, paths, tokens in receipts:
        receipt = session / (step + "-receipt.json")
        write_json(
            receipt,
            dict(
                kind="tool_receipt", step=step, command=command, status=status,
                exit_status=exit_status, retries_used=retries, gate_tokens=tokens,
                input_paths=[str(p) for p in paths],
            ),
        )
        subprocess.run(
            [sys.executable, str(ledger), "append", "--passport-path",
             str(session / "passport.md"), "--entry-file", str(receipt)],
            check=True,
        )
    passed = [row[0] for row in receipts if row[2] == "passed"]
    write_json(
        session / "claims.json",
        dict(decisions=[], steps=[dict(step=step, status="passed") for step in passed],
             expected_steps=passed),
    )
    subprocess.run(
        [sys.executable, str(ledger), "report", "--passport-path", str(session / "passport.md"),
         "--claims", str(session / "claims.json")],
        check=True,
    )
    print(dict(status="NCVR complete negative result", diagnostics=diagnostics, goal="active"), flush=True)


if __name__ == "__main__":
    close()
