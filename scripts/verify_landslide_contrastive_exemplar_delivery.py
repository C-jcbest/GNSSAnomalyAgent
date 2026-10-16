"""Seal or verify v50 without changing frozen inference or historical delivery."""

from __future__ import annotations

import argparse
from pathlib import Path

from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha


def verify_evidence():
    run = BASE / "contrastive-exemplar-pilot-v2"
    analysis = BASE / "contrastive-exemplar-analysis-v1"
    archive = BASE / "manuscript-v50"
    review = BASE / "contrastive-exemplar-visual-review-v1"
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original experiment source or model changed")
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Frozen analysis changed")
    state = read_json(run / "state.json")
    if state["status"] != "inference_complete" or state["attempts"] != 64 or state["failures"] != 2:
        raise ValueError("Inference terminal state differs")
    cost = read_json(analysis / "cost.json")
    if cost["reported_tokens"] != 247224 or cost["new_requests"] != 64 or cost["new_receipts"] != 64:
        raise ValueError("Cost differs")
    decision = read_json(analysis / "decision.json")
    if decision["decision"] != "STOP_CONTRASTIVE_EXEMPLAR_PILOT" or decision["passed"] != 8:
        raise ValueError("Registered stop decision changed")
    observations = read_json(review / "observations.json")
    if observations["raw_medians_reconstructed"] != 104 or len(observations["observations"]) != 8:
        raise ValueError("Actual visual/numerical review incomplete")
    previous = read_json(archive / "v49-delivery-manifest-original.json")["files"]
    mapping = read_json(archive / "v49-preservation-after.json")["archived_document_mapping"]
    for name, expected in previous.items():
        path = Path(mapping[name]) if name in mapping else ROOT / name
        if sha(path) != expected:
            raise ValueError("Historical v49 evidence changed")
    if sha(archive / "v49-delivery-manifest-original.json") != sha(BASE / "manuscript-v49/delivery-manifest.json"):
        raise ValueError("Original v49 manifest bytes changed")
    for source, folder in (
        ("landslide-adaptive-review-manuscript-draft", "formula-verification"),
        ("landslide-contrastive-exemplar-results", "result-formula-verification"),
    ):
        audit = read_json(archive / folder / "formula-audit.json")
        if audit["status"] != "passed" or audit["source_sha256"] != sha(ROOT / "docs" / (source + ".md")):
            raise ValueError("LaTeX render audit does not match current Markdown")
    return {"old_delivery_files": len(previous), "requests": 64, "reviewed_query_images": 8,
            "reported_tokens": 247224, "decision": decision["decision"]}


def seal():
    archive = BASE / "manuscript-v50"
    result = verify_evidence()
    if (archive / "delivery-manifest.json").exists():
        raise ValueError("Seal only a new delivery; use verification thereafter")
    for folder in ("contrastive-exemplar-literature-v1", "contrastive-exemplar-visual-review-v1"):
        directory = BASE / folder
        export(directory / "delivery-freeze.json", {
            "files": {path.relative_to(directory).as_posix(): sha(path)
                      for path in directory.rglob("*") if path.is_file()},
            "scope": "Includes later reading/diagnostics; earlier partial manifests retained unchanged",
        })
    export(archive / "verification.json", {"status": "passed", **result})
    folders = (
        "contrastive-exemplar-literature-v1", "contrastive-exemplar-pilot-v1",
        "contrastive-exemplar-pilot-v2", "contrastive-exemplar-analysis-v1",
        "contrastive-exemplar-visual-review-v1", "manuscript-v50",
    )
    paths = [path for folder in folders for path in (BASE / folder).rglob("*") if path.is_file()]
    paths += [ROOT / "docs" / (name + ".md") for name in (
        "landslide-adaptive-review-manuscript-draft", "landslide-adaptive-review-method",
        "landslide-submission-roadmap", "project-status", "landslide-contrastive-exemplar-results",
    )]
    paths += [ROOT / path for path in (
        "src/gnss_sim/landslide_contrastive_exemplars.py",
        "scripts/probe_landslide_contrastive_exemplars.py",
        "scripts/report_landslide_contrastive_exemplars.py",
        "scripts/write_landslide_contrastive_exemplar_results.py",
        "scripts/verify_landslide_contrastive_exemplar_delivery.py",
        "scripts/verify_landslide_manuscript_math.cjs",
        "tests/test_landslide_contrastive_exemplars.py",
        "tests/test_landslide_contrastive_review_output.py",
        "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v49/delivery-manifest.json",
    )]
    export(archive / "delivery-manifest.json",
           {"files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}})
    print("Sealed", len(set(paths)), "v50 files;", result)


def verify():
    result = verify_evidence()
    files = read_json(BASE / "manuscript-v50/delivery-manifest.json")["files"]
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Delivered file changed: " + name)
    print("Verified", len(files), "v50 files;", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        verify()
