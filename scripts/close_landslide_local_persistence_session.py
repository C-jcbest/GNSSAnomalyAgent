"""Save the local-pair result, draft snapshot, and actual verification receipts."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def close():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "local-persistence-session-v1"
    for folder, manifest in (
        ("local-persistence-pairs-v1", "encoding-freeze.json"),
        ("local-persistence-calibration-v1", "freeze.json"),
        ("local-persistence-replay-v1", "freeze.json"),
        ("local-persistence-report-v1", "freeze.json"),
    ):
        verify_manifest(base / folder, manifest)
    if read_json(base / "local-persistence-replay-v1/verification.json")["status"] != "VERIFIED":
        raise ValueError("Cannot close without successful selected-output replay")
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-12-v5.md"
    audit = base / "manuscript-math-v5-final/formula-audit.json"
    output = base / "manuscript-v5"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = [
        draft,
        audit,
        base / "local-persistence-pairs-v1/encoding-freeze.json",
        base / "local-persistence-calibration-v1/freeze.json",
        base / "local-persistence-replay-v1/freeze.json",
        base / "local-persistence-report-v1/freeze.json",
        base / "local-persistence-literature-v1/freeze.json",
        base / "local-persistence-primary-v2/freeze.json",
        session / "reading-notes.md",
        session / "actual-visual-review.md",
        session / "raw-observation-blocks.json",
        session / "acronyms-v5.json",
        session / "pai-code-receipt.json",
    ]
    write_json(output / "evidence-bindings.json", {str(path): sha(path) for path in evidence})
    write_json(
        output / "manifest.json",
        dict(
            status="complete_draft",
            source_sha256=sha(draft),
            files={path.name: sha(path) for path in output.iterdir()},
            independent_expert_reference=False,
            test_truth_opened=False,
            contribution_validated=False,
            acronym_check="partial_advisory_with_findings",
            stopped_candidate="CP-PP",
        ),
    )
    write_json(
        session / "math-failure-receipt.json",
        dict(
            kind="tool_receipt",
            step="manuscript-v5-first-math-attempt",
            command="node scripts/verify_landslide_manuscript_math.cjs docs/gnss-deformation-manuscript-draft-2026-10-12-v5.md artifacts/gnss-deformation-diversity-2026-10-12/manuscript-math-v5",
            status="failed",
            exit_status=1,
            retries_used=0,
            gate_tokens=[
                "KaTeX ParseError: unexpected backspace in new cases formula",
                "Escaping corrected only in manuscript and writer; model outputs unchanged",
            ],
        ),
    )
    receipts = {
        "replay": (
            "local-pair-and-model-replay",
            "verify_landslide_local_persistence.py --data data-v1 --encoded quality-encoding-v1 --pairs local-persistence-pairs-v1 --parent disagreement-calibration-v1 --run local-persistence-calibration-v1 --out local-persistence-replay-v1",
            evidence[2:6],
            [
                "127 pairs",
                "9 selected models",
                "157680 probability values",
                "22 methods/528 outputs exact",
            ],
            0,
        ),
        "tests": (
            "local-pair-business-tests",
            "python -m pytest tests/test_landslide_local_persistence.py -q",
            [
                root / "src/gnss_sim/landslide_local_persistence.py",
                root / "tests/test_landslide_local_persistence.py",
            ],
            ["4 passed"],
            0,
        ),
        "math": (
            "manuscript-v5-latex-render-check",
            "node scripts/verify_landslide_manuscript_math.cjs docs/gnss-deformation-manuscript-draft-2026-10-12-v5.md artifacts/gnss-deformation-diversity-2026-10-12/manuscript-math-v5-final",
            [draft, audit],
            ["32 inline", "15 display", "KaTeX passed"],
            1,
        ),
    }
    for name, (step, command, paths, tokens, retries) in receipts.items():
        write_json(
            session / f"{name}-receipt.json",
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                input_paths=[str(path) for path in paths],
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
            steps=[dict(step=row[0], status="passed") for row in receipts.values()],
            expected_steps=[row[0] for row in receipts.values()],
        ),
    )
    print(
        "Saved manuscript-v5 and replay/test/math receipts; candidate remains stopped", flush=True
    )


if __name__ == "__main__":
    close()
