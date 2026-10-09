"""Prepare a prediction-free review page or validate a separately completed review."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_stage_numeric import load_case  # noqa: E402
from landslide_stage_numeric import verify_inference as verify_numeric_run  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide_stage_reference import (  # noqa: E402
    StageReferencePacket,
    cache_support_inventory,
    compile_reference,
    reference_template,
)
from gnss_sim.visual import _unique_object  # noqa: E402

SOURCE = ROOT / "artifacts/landslide-stage-numeric-2026-10-08/run-v1"
DEFAULT_KIT = ROOT / "artifacts/landslide-stage-reference-2026-10-08/kit-v1"
CODE_FILES = (
    "src/gnss_sim/landslide_stage_reference.py", "scripts/landslide_stage_reference.py",
    "scripts/landslide_stage_reference_page.html", "tests/test_landslide_stage_reference.py",
    "scripts/check_landslide_stage_reference_ui.cjs",
    "docs/landslide-stage-reference-protocol.md",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def load_contexts(public):
    records = read_json(public / "source/source-manifest.json")["cases"]
    entries = {entry["case_id"]: entry for entry in read_json(public / "source/activity-manifest.json")["cases"]}
    contexts = []
    for record in records:
        case, review, rows, digest, motions = load_case(public, record, entries)
        contexts.append({"case": case, "review": review, "rows": rows,
                         "activity_sha256": digest, "motions": motions, "record": record})
    return contexts


def prepare(kit):
    verify_numeric_run()
    old_public = SOURCE / "public"
    contexts = load_contexts(old_public)
    records = read_json(old_public / "source/source-manifest.json")["cases"]
    entries = read_json(old_public / "source/activity-manifest.json")["cases"]
    entries_by_id = {entry["case_id"]: entry for entry in entries}
    names = {"source/source-manifest.json", "source/activity-manifest.json", "source/activity-freeze.json"}
    for record in records:
        names.add("source/" + record["input"])
        names.update("source/" + record["diagnostics"][str(w)]["data"] for w in (31, 61, 91))
        names.update(view["image"] for view in record["views"])
        names.update(record["diagnostics"][str(w)]["image"] for w in (31, 61, 91))
        names.update("source/activity-source/" + entries_by_id[record["case_id"]][key]
                     for key in ("review", "daily"))
    kit.mkdir(parents=True, exist_ok=False)
    public = kit / "public"
    public.mkdir()
    for name in sorted(names):
        target = public / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_public / name, target)
        if sha(target) != sha(old_public / name):
            raise ValueError("Review input copy differs from frozen source")
    guide = ROOT / "docs/landslide-stage-reference-protocol.md"
    shutil.copyfile(guide, public / "review-guide.md")
    manifest = {
        "schema_version": "landslide-stage-reference-kit-v1", "status": "unreviewed_preparation_only",
        "source_numeric_manifest_sha256": sha(SOURCE / "manifest.json"),
        "source_numeric_freeze_sha256": sha(SOURCE / "inference-freeze.json"),
        "input_files_sha256": {path.relative_to(public).as_posix(): sha(path)
                               for path in public.rglob("*") if path.is_file()},
        "source_code_sha256": {name: sha(ROOT / name) for name in CODE_FILES},
        "cases": len(contexts), "interiors": sum(s["state"] == "activity"
                                                  for c in contexts for s in c["review"]["spans"]),
        "independent_reference_exists": False, "model_calls": 0,
    }
    write_json(public / "bundle-manifest.json", manifest)
    manifest_sha = sha(public / "bundle-manifest.json")
    template = reference_template(contexts, manifest_sha)
    write_json(public / "review-template.json", template)
    write_json(public / "review-schema.json", StageReferencePacket.model_json_schema())
    display = []
    for context in contexts:
        case, record = context["case"], context["record"]
        display.append({"case_id": case.case_id, "dates": [str(day) for day in case.dates],
                        "observed": [value is not None for value in case.displacement_mm],
                        "raw_images": [view["image"] for view in record["views"]],
                        "auxiliary_images": [record["diagnostics"][str(w)]["image"] for w in (31, 61, 91)],
                        "support": [cache_support_inventory(case, context["motions"], s["start"], s["stop"])
                                    for s in context["review"]["spans"] if s["state"] == "activity"]})
    data = json.dumps({"template": template, "cases": display}, ensure_ascii=False)
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    html = (ROOT / "scripts/landslide_stage_reference_page.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(html.replace("__PAGE_DATA__", data), encoding="utf-8")
    for name in CODE_FILES:
        target = kit / "source-code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(kit / "delivery-audit.json", {
        "source_manifest_sha256": manifest_sha,
        "public_files_sha256": {path.relative_to(public).as_posix(): sha(path)
                                for path in public.rglob("*") if path.is_file()},
        "reference_status": "unreviewed", "model_calls": 0,
    })
    with zipfile.ZipFile(kit / "review-kit.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(public.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(public).as_posix())
    verify_bundle(kit)
    print(json.dumps({"kit": str(kit), "cases": len(contexts), "interiors": manifest["interiors"],
                      "status": "unreviewed preparation; no reference labels", "model_calls": 0}))


def verify_bundle(kit):
    public = kit / "public"
    manifest = read_json(public / "bundle-manifest.json")
    delivery = read_json(kit / "delivery-audit.json")
    if sha(public / "bundle-manifest.json") != delivery["source_manifest_sha256"]:
        raise ValueError("Review source manifest changed")
    for name, digest in delivery["public_files_sha256"].items():
        if sha(public / name) != digest:
            raise ValueError(f"Review delivery changed: {name}")
    for name, digest in manifest["source_code_sha256"].items():
        if sha(ROOT / name) != digest or sha(kit / "source-code" / name) != digest:
            raise ValueError(f"Review implementation changed: {name}")
    contexts = load_contexts(public)
    expected = reference_template(contexts, delivery["source_manifest_sha256"])
    if expected != read_json(public / "review-template.json"):
        raise ValueError("Review template contains prefilled decisions or changed provenance")
    return contexts, delivery["source_manifest_sha256"]


def compile_file(kit, review_path, output):
    contexts, digest = verify_bundle(kit)
    payload = read_json(review_path)
    packet, daily, summary, inventory = compile_reference(payload, contexts, digest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "review.json", packet)
    write_json(output / "daily-reference.json", daily)
    write_json(output / "summary.json", summary)
    write_json(output / "support-inventory.json", inventory)
    with (output / "daily-reference.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    write_json(output / "reference-freeze.json", {
        "source_manifest_sha256": digest, "submitted_review_sha256": sha(review_path),
        "files_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        "formal_reference_verified": False,
    })
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "compile"))
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.kit.resolve())
    elif args.mode == "verify":
        contexts, _ = verify_bundle(args.kit.resolve())
        print(f"Verified {len(contexts)} cases; template remains UNREVIEWED")
    else:
        if args.review is None or args.output is None:
            parser.error("compile requires --review and a new --output directory")
        compile_file(args.kit.resolve(), args.review.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
