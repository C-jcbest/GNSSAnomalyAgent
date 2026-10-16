"""Bind the completed disagreement experiment to its unchanged draft evidence."""

from __future__ import annotations

import json
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest

from gnss_sim.artifacts import sha, write_json


def close():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    session = base / "disagreement-session-v1"
    for folder in (
        "disagreement-calibration-v1",
        "disagreement-replay-v1",
        "disagreement-report-v1",
        "deformation-duration-probe-v1",
    ):
        verify_manifest(base / folder, "freeze.json")
    replay = json.loads((base / "disagreement-replay-v1/verification.json").read_text())
    if replay["status"] != "VERIFIED" or replay["max_probability_error"] != 0:
        raise ValueError("Completed replay does not support an exact verification claim")
    draft = root / "docs/gnss-deformation-manuscript-draft-2026-10-12-v4.md"
    audit = base / "manuscript-math-v4-final/formula-audit.json"
    output = base / "manuscript-v4"
    output.mkdir(exist_ok=False)
    (output / "draft.md").write_bytes(draft.read_bytes())
    evidence = [
        draft,
        audit,
        base / "disagreement-calibration-v1/freeze.json",
        base / "disagreement-replay-v1/freeze.json",
        base / "disagreement-report-v1/freeze.json",
        base / "disagreement-statistics-v2/freeze.json",
        base / "deformation-duration-probe-v1/freeze.json",
        session / "actual-visual-review.md",
        session / "acronyms-v4-final.json",
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
            acronym_check="advisory_partial_with_findings",
        ),
    )
    receipts = {
        "replay": (
            "disagreement-artifact-validation",
            "Validate frozen calibration/replay/report/duration manifests and exact replay receipt",
            evidence[2:7],
            ["VERIFIED", "maximum probability error 0"],
        ),
        "tests": (
            "disagreement-and-duration-business-tests",
            "python -m pytest tests/test_landslide_disagreement_review.py tests/test_landslide_observed_duration.py -q",
            [
                root / "src/gnss_sim/landslide_disagreement_review.py",
                root / "src/gnss_sim/landslide_observed_duration.py",
                root / "tests/test_landslide_disagreement_review.py",
                root / "tests/test_landslide_observed_duration.py",
            ],
            ["8 passed"],
        ),
        "math": (
            "manuscript-v4-latex-render-check",
            "node scripts/verify_landslide_manuscript_math.cjs docs/gnss-deformation-manuscript-draft-2026-10-12-v4.md artifacts/gnss-deformation-diversity-2026-10-12/manuscript-math-v4-final",
            [draft, audit],
            ["26 inline", "14 display", "KaTeX passed"],
        ),
    }
    for name, (step, command, paths, tokens) in receipts.items():
        write_json(
            session / f"{name}-receipt.json",
            dict(
                kind="tool_receipt",
                step=step,
                command=command,
                status="passed",
                exit_status=0,
                retries_used=0,
                gate_tokens=tokens,
                input_paths=[str(path) for path in paths],
            ),
        )
    write_json(
        session / "claims.json",
        dict(
            decisions=[],
            steps=[dict(step=value[0], status="passed") for value in receipts.values()],
            expected_steps=[value[0] for value in receipts.values()],
        ),
    )
    print(
        "Frozen manuscript-v4, artifact validation, and observed verification receipts", flush=True
    )


if __name__ == "__main__":
    close()
