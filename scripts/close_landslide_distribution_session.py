"""Bind the new draft, actual receipts and research decisions into the local run ledger."""

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
    session = base / "distribution-session-v1"
    skill = Path(
        "C:/Users/JC/.codex/plugins/cache/ars-codex/ars-codex/3.23.0/skills/academic-research-suite/ars/scripts/run_ledger.py"
    )
    manifests = []
    for folder, filename in (
        ("motion-mass-probe-v1", "freeze.json"),
        ("motion-mass-replay-v2", "freeze.json"),
        ("motion-mass-report-v2", "freeze.json"),
        ("distribution-view-encoding-v2", "encoding-freeze.json"),
        ("distribution-view-calibration-v1", "freeze.json"),
        ("distribution-view-replay-v1", "freeze.json"),
        ("distribution-view-report-v1", "freeze.json"),
        ("morphology-primary-v1", "freeze.json"),
        ("morphology-code-source-v1", "freeze.json"),
    ):
        verify_manifest(base / folder, filename)
        manifests.append(base / folder / filename)
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v7.md"
    audit = base / "manuscript-math-v7/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Math audit belongs to a different draft")
    if read_json(base / "distribution-view-replay-v1/verification.json")["status"] != "VERIFIED":
        raise ValueError("Missing successful pilot replay")
    output = base / "manuscript-v7"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = manifests + [
        draft,
        audit,
        session / "reading-notes.md",
        session / "actual-visual-review.md",
        root / "scripts/write_landslide_distribution_manuscript.py",
    ]
    write_json(output / "evidence-bindings.json", {str(p.resolve()): sha(p) for p in evidence})
    write_json(
        output / "manifest.json",
        dict(
            status="complete_research_draft",
            source_sha256=sha(draft),
            independent_expert_reference=False,
            test_truth_opened=False,
            contribution_validated=False,
            pilot_decision=read_json(base / "distribution-view-calibration-v1/decision.json"),
            files={p.name: sha(p) for p in output.iterdir()},
        ),
    )
    failed = (
        (
            "source-api",
            "morphology-source-api",
            "fetch_landslide_morphology_code.py",
            "GitHub API403 rate limit exceeded",
            [base / "morphology-code-metadata-v1/failure.json"],
        ),
        (
            "source-clone",
            "morphology-first-tshape-clone",
            "git clone --depth1 --filter=blob:none --no-checkout CSTCloudOps/TShape into morphology-author-code-v1",
            "OpenSSL SSL_ERROR_SYSCALL",
            [session / "reading-notes.md"],
        ),
        (
            "source-main",
            "morphology-tshape-main-fetch",
            "git show HEAD:main.py in morphology-author-code-v2/TShape",
            "Promisor blob SSL fetch failed; file not read",
            [session / "reading-notes.md"],
        ),
        (
            "mass-report",
            "motion-mass-first-report",
            "report_landslide_motion_mass.py --data data --out motion-mass-report-v1",
            "FileNotFoundError: actual input directory is data-v1",
            [base / "motion-mass-report-v1/failure.json"],
        ),
        (
            "encoding",
            "distribution-first-encoding",
            "encode_landslide_motion_distribution.py --out distribution-view-encoding-v1",
            "FileExistsError: progress CSV reused x creation mode",
            [
                base / "distribution-view-encoding-v1/failure.json",
                base / "distribution-view-encoding-v1/source-at-failure.py",
            ],
        ),
    )
    for name, step, command, error, paths in failed:
        receipt = session / (name + "-failure-receipt.json")
        write_json(
            receipt,
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                status="failed",
                exit_status=1,
                retries_used=0,
                gate_tokens=[error],
                input_paths=[str(p) for p in paths],
            ),
        )
        subprocess.run(
            [
                sys.executable,
                str(skill),
                "append",
                "--passport-path",
                str(session / "passport.md"),
                "--entry-file",
                str(receipt),
            ],
            check=True,
        )
    passed = (
        (
            "tests",
            "distribution-business-tests",
            "python -m pytest tests/test_landslide_motion_mass.py tests/test_landslide_motion_distribution_view.py -q",
            [
                root / "src/gnss_sim/landslide_motion_mass.py",
                root / "tests/test_landslide_motion_mass.py",
                root / "src/gnss_sim/landslide_motion_distribution_view.py",
                root / "tests/test_landslide_motion_distribution_view.py",
            ],
            ["7 passed"],
            0,
        ),
        (
            "mass-replay",
            "motion-mass-final-replay",
            "verify_landslide_motion_mass.py --out motion-mass-replay-v2",
            manifests[:2],
            [
                "72 evidence/5selected models/87600probabilities/18methods432outputs exact",
                "Method-keyed summary fix; numerical outputs unchanged",
            ],
            1,
        ),
        (
            "source",
            "morphology-exact-source-export",
            "export_landslide_morphology_code.py --out morphology-code-source-v1",
            [manifests[-1]],
            ["6 source files at2exact commits", "No author model execution"],
            1,
        ),
        (
            "encoding",
            "distribution-final-encoding",
            "encode_landslide_motion_distribution.py --out distribution-view-encoding-v2",
            [manifests[3]],
            [
                "24fixed original records",
                "Ordered and permuted view embeddings complete",
                "Same TRAIN-only numeric scaling",
            ],
            1,
        ),
        (
            "replay",
            "distribution-selected-replay",
            "verify_landslide_motion_distribution.py --out distribution-view-replay-v1",
            manifests[3:6],
            [
                "96auxiliary encoding samples",
                "10selected models/58400probabilities/21methods168outputs exact",
                "Identical duplicate refit verified",
            ],
            0,
        ),
        (
            "writer",
            "distribution-draft-write",
            "python scripts/write_landslide_distribution_manuscript.py",
            [draft],
            ["Complete v7 Markdown, actual results and limitations"],
            0,
        ),
        (
            "math",
            "distribution-latex-check",
            "node scripts/verify_landslide_manuscript_math.cjs draft-v7 manuscript-math-v7",
            [draft, audit],
            ["KaTeX passed", "Source hash bound"],
            0,
        ),
    )
    for name, step, command, paths, tokens, retries in passed:
        receipt = session / (name + "-receipt.json")
        write_json(
            receipt,
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                status="passed",
                exit_status=0,
                retries_used=retries,
                gate_tokens=tokens,
                input_paths=[str(p) for p in paths],
            ),
        )
        subprocess.run(
            [
                sys.executable,
                str(skill),
                "append",
                "--passport-path",
                str(session / "passport.md"),
                "--entry-file",
                str(receipt),
            ],
            check=True,
        )
    claims = dict(
        decisions=[],
        steps=[dict(step=row[1], status="passed") for row in passed],
        expected_steps=[row[1] for row in passed],
    )
    write_json(session / "claims.json", claims)
    subprocess.run(
        [
            sys.executable,
            str(skill),
            "report",
            "--passport-path",
            str(session / "passport.md"),
            "--claims",
            str(session / "claims.json"),
        ],
        check=True,
    )
    print(
        "Draft-v7 and completed experiment receipts bound; overall research goal remains active",
        flush=True,
    )


if __name__ == "__main__":
    close()
