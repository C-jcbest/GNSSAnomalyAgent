"""Bind draft v6 and actual run receipts without changing experimental outputs."""

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "shared-frame-session-v1"
    manifests = []
    for folder, manifest in (
        ("shared-frame-pairs-v1", "encoding-freeze.json"),
        ("shared-frame-calibration-v1", "freeze.json"),
        ("shared-frame-replay-v1", "freeze.json"),
        ("shared-frame-report-v1", "freeze.json"),
        ("shared-frame-analysis-v1", "freeze.json"),
        ("shared-frame-literature-v1", "freeze.json"),
        ("local-persistence-context-probe-v1", "freeze.json"),
    ):
        verify_manifest(base / folder, manifest)
        manifests.append(base / folder / manifest)
    if read_json(base / "shared-frame-replay-v1/verification.json")["status"] != "VERIFIED":
        raise ValueError("Selected-output replay is not complete")
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v6.md"
    audit = base / "manuscript-math-v6/formula-audit.json"
    if read_json(audit)["source_sha256"] != sha(draft):
        raise ValueError("Math check belongs to another draft")
    output = base / "manuscript-v6"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = manifests + [
        draft,
        audit,
        session / "reading-notes.md",
        session / "actual-visual-review.md",
        base / "shared-frame-analysis-v1/raw-observation-blocks.json",
        root / "scripts/write_landslide_shared_frame_manuscript.py",
    ]
    write_json(output / "evidence-bindings.json", {str(p): sha(p) for p in evidence})
    write_json(
        output / "manifest.json",
        dict(
            status="complete_research_draft",
            source_sha256=sha(draft),
            independent_expert_reference=False,
            test_truth_opened=False,
            contribution_validated=False,
            stopped_candidate="SC-PP",
            acronym_check="not_newly_run; inherited_v5_partial_advisory_findings",
            files={p.name: sha(p) for p in output.iterdir()},
        ),
    )
    write_json(
        session / "instructions.json",
        read_json(base / "local-persistence-session-v1/instructions.json"),
    )
    failed = (
        (
            "writer-failure",
            "shared-frame-draft-first-attempt",
            "python scripts/write_landslide_shared_frame_manuscript.py with PYTHONPATH=src;scripts",
            "ModuleNotFoundError: open_clip; no draft created, models unchanged",
        ),
        (
            "math-failure",
            "shared-frame-math-first-attempt",
            "node scripts/verify_landslide_manuscript_math.cjs draft-v6 manuscript-math-v6",
            "ENOENT: preceding draft-generation failure left no Markdown source",
        ),
    )
    for name, step, command, error in failed:
        write_json(
            session / (name + "-receipt.json"),
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                status="failed",
                exit_status=1,
                retries_used=0,
                gate_tokens=[error],
            ),
        )
    passed = (
        (
            "tests",
            "shared-frame-business-tests",
            "python -m pytest tests/test_landslide_shared_frame.py -q",
            [
                root / "src/gnss_sim/landslide_shared_frame.py",
                root / "tests/test_landslide_shared_frame.py",
            ],
            ["4 passed"],
            0,
        ),
        (
            "replay",
            "shared-frame-pair-model-replay",
            "verify_landslide_shared_frame.py --data data-v1 --encoded quality-encoding-v1 --pairs shared-frame-pairs-v1 --parent-pairs local-persistence-pairs-v1 --physical-replay local-persistence-replay-v1 --parent disagreement-calibration-v1 --run shared-frame-calibration-v1 --out shared-frame-replay-v1",
            manifests[:3],
            [
                "127 paired charts",
                "9 selected models",
                "157680 probabilities",
                "22 methods/528 record outputs exact",
            ],
            0,
        ),
        (
            "writer",
            "shared-frame-draft-write",
            "python scripts/write_landslide_shared_frame_manuscript.py with bundled vision runtime on PYTHONPATH",
            [draft],
            ["v6 complete Markdown generated", "raw frozen result table retained"],
            1,
        ),
        (
            "math",
            "shared-frame-latex-check",
            "node scripts/verify_landslide_manuscript_math.cjs docs/gnss-deformation-manuscript-draft-2026-10-13-v6.md artifacts/gnss-deformation-diversity-2026-10-12/manuscript-math-v6",
            [draft, audit],
            ["36 inline", "17 display", "KaTeX passed"],
            1,
        ),
    )
    for name, step, command, paths, tokens, retries in passed:
        write_json(
            session / (name + "-receipt.json"),
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                input_paths=[str(p) for p in paths],
                status="passed",
                exit_status=0,
                retries_used=retries,
                gate_tokens=tokens,
            ),
        )
    write_json(
        session / "claims.json",
        dict(
            decisions=[],
            steps=[dict(step=row[1], status="passed") for row in passed],
            expected_steps=[row[1] for row in passed],
        ),
    )
    print("Bound manuscript-v6, actual failed/passed receipts; SC-PP remains stopped", flush=True)


if __name__ == "__main__":
    close()
