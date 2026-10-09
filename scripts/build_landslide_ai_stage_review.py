"""Assemble explicitly authored AI image review notes; never infer stage labels."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from landslide_stage_reference import DEFAULT_KIT, ROOT, compile_file, read_json, verify_bundle

from gnss_sim.artifacts import sha, write_json

DEFAULT_REVIEW = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1"
NOTE_FILES = ("observations.json", "observations-05-08.json", "observations-09-12.json")
SEGMENT_FIELDS = (
    "start", "stop", "feature", "raw_evidence", "rate_evidence",
    "nonzero_evidence", "steady_evidence", "unknown_reason",
)


def load_authored_notes(review_directory):
    cases = {}
    for name in NOTE_FILES:
        for case_id, note in read_json(review_directory / name)["cases"].items():
            if case_id in cases:
                raise ValueError(f"Duplicate authored case: {case_id}")
            cases[case_id] = note
    return cases


def assemble_packet(template, notes, reviewed_at):
    """Keep frozen source identities and copy only explicit review decisions."""
    if set(notes) != {case["case_id"] for case in template["cases"]}:
        raise ValueError("Authored notes must cover every source case")
    packet = json.loads(json.dumps(template))
    packet["reviewer"] = {
        "name": "Codex / current conversation AI image review / 2026-10-08",
        "background": (
            "AI直接查看本轮原图与辅助图；已接触历史预测摘要并延续作者活动复核。"
            "不是现场地质专家，不是独立盲审。本轮未读取生成阶段标签或封存测试。"
        ),
        "reviewer_type": "ai",
        "prior_predictions_seen": True,
        "prior_generation_labels_seen": False,
        "participated_in_activity_review": True,
    }
    for case in packet["cases"]:
        note = notes[case["case_id"]]
        expected = [(it["start"], it["stop"]) for it in case["interiors"]]
        supplied = [(it["start"], it["stop"]) for it in note["interiors"]]
        if expected != supplied:
            raise ValueError(f"Authored notes changed activity interiors: {case['case_id']}")
        for interior, authored in zip(case["interiors"], note["interiors"], strict=True):
            if not note["raw_views"]:
                raise ValueError("Activity decisions require actual raw image inspection")
            segments = []
            for decision in authored["segments"]:
                unknown_fields = set(decision) - set(SEGMENT_FIELDS)
                if unknown_fields:
                    raise ValueError(f"Unexpected authored fields: {unknown_fields}")
                segments.append({
                    field: decision.get(field, None if field == "unknown_reason" else "")
                    for field in SEGMENT_FIELDS
                })
            interior.update(
                review_status="reviewed",
                raw_activity_status=authored["raw_activity_status"],
                raw_activity_evidence=authored["raw_activity_evidence"],
                raw_views_seen=True,
                auxiliary_views_seen=bool(note["auxiliary_windows"]),
                reviewed_at=reviewed_at,
                segments=segments,
            )
    return packet


def build_review_page(source_public, public, packet, summary):
    """Reuse the frozen workbench in a separate delivery, with disclosed AI answers."""
    html = (source_public / "index.html").read_text(encoding="utf-8")
    pattern = r'(<script id="page-data" type="application/json">)(.*?)(</script>)'
    match = re.search(pattern, html, re.DOTALL)
    if match is None:
        raise ValueError("Source workbench has no embedded page payload")
    data = json.loads(match.group(2))
    data["template"] = packet
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = html[:match.start(2)] + encoded + html[match.end(2):]
    html = html.replace("长期位移阶段复核工作台", "长期位移 · AI开发标注与证据")
    html = html.replace("GNSS · CONDITIONAL STAGE REVIEW", "GNSS · AI DEVELOPMENT REVIEW")
    html = html.replace("先确认原始位移，再判断运动阶段", "原图先行：已填写的阶段开发标注")
    original_notice = (
        "这是未审复核工作台，默认没有阶段答案。活动范围来自已冻结的作者复核；"
        "本页只研究其内部阶段。填写、校验和来源哈希不会自动证明复核独立或正确。当前没有正式准确率。"
    )
    updated_notice = (
        "这是已接触历史预测的AI开发标注，不能作为独立参考或准确率证明。"
        "21个活动内部已填写，原始空复核包仍保持未审。"
        f"确定阶段有观测日{summary['stage_evaluable_days']}日；其余明确未知或缺测。"
        "边界是人工视觉粗定位，转折带保留未知；全记录居中证据只支持离线回顾。"
    )
    if html.count(original_notice) != 1:
        raise ValueError("Source page disclosure changed; inspect before publishing")
    html = html.replace(original_notice, updated_notice)
    html = html.replace("gnss-stage-reference-v1:", "gnss-ai-stage-review-v1:")
    html = html.replace("空填写模板", "原始空填写模板")
    html = html.replace(
        '<div class="links">',
        '<div class="links">'
        '<a href="compiled/review.json">AI标注JSON</a>'
        '<a href="compiled/daily-reference.csv">逐日标签CSV</a>'
        '<a href="compiled/summary.json">覆盖与身份声明</a>'
        '<a href="inspection-ledger.json">实际看图清单</a>',
        1,
    )
    html = html.replace(
        "完整文件还需使用Python编译器核验、冻结；不会直接写入旧实验或自动生成评分。",
        "本页初始AI标注已经Python编译、单独冻结；任何网页修改需重新下载编译，不会回写该冻结结果或生成正式评分。",
    )
    (public / "index.html").write_text(html, encoding="utf-8")


def build(review_directory, kit):
    if (review_directory / "compiled").exists() or (review_directory / "public").exists():
        raise FileExistsError("This review already has compiled output; use a new version")
    contexts, source_digest = verify_bundle(kit)
    notes = load_authored_notes(review_directory)
    reviewed_at = datetime.now(timezone.utc).isoformat()
    packet = assemble_packet(read_json(kit / "public/review-template.json"), notes, reviewed_at)
    submission = review_directory / "submitted-review.json"
    with submission.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    compile_file(kit, submission, review_directory / "compiled")
    summary = read_json(review_directory / "compiled/summary.json")
    if summary["reference_use"] != "development_only" or summary["formal_reference_verified"]:
        raise ValueError("AI review must remain development-only")

    viewed_images = {}
    for case_id, note in notes.items():
        names = [f"images/{case_id}-raw-{view}.png" for view in note["raw_views"]]
        names.extend(f"images/{case_id}-aux-{window}.png" for window in note["auxiliary_windows"])
        for name in names:
            viewed_images[name] = sha(kit / "public" / name)
    ledger = {
        "reviewer": packet["reviewer"],
        "reviewed_at": reviewed_at,
        "source_manifest_sha256": source_digest,
        "images_actually_inspected_sha256": viewed_images,
        "raw_views": sum(len(note["raw_views"]) for note in notes.values()),
        "auxiliary_views": sum(len(note["auxiliary_windows"]) for note in notes.values()),
        "process": "Each active case: raw views inspected before its auxiliary images; labels explicitly authored afterward",
        "numerical_followup": "Read-only cached rate checks after visual decisions; two candidate decisions downgraded to unknown before freeze",
        "tool": "functions.exec -> tools.view_image; direct current conversation image inspection",
        "image_delivery": "Auxiliary images displayed at 1792x1407 after resize from 2100x1650",
        "model_identity": "Codex session; exact runtime model identifier not exposed",
        "inference_api_requests": 0,
        "generation_exposure_scope": "No generation labels read during this AI review; historical predictions were already exposed",
        "formal_reference_verified": False,
        "reference_use": "development_only",
        "boundary_policy": "Authored half-open day bands; not calibrated exact changepoints",
        "annotation_units": "3D motion speed stages; signed component movement is evidence, not the stage target",
    }
    write_json(review_directory / "inspection-ledger.json", ledger)
    public = review_directory / "public"
    shutil.copytree(kit / "public", public)
    shutil.copytree(review_directory / "compiled", public / "compiled")
    shutil.copyfile(review_directory / "inspection-ledger.json", public / "inspection-ledger.json")
    for name in NOTE_FILES:
        shutil.copyfile(review_directory / name, public / name)
    shutil.copyfile(review_directory / "decision-corrections.json", public / "decision-corrections.json")
    build_review_page(kit / "public", public, packet, summary)

    daily = read_json(review_directory / "compiled/daily-reference.json")
    original_rows = {(context["case"].case_id, day): row
                     for context in contexts for day, row in enumerate(context["rows"])}
    activity_reason_counts = Counter()
    for row in daily:
        key = (row["case_id"], row["day_index"])
        if row["activity_label"] != original_rows[key]["activity_label"]:
            raise ValueError("Compiled AI reference changed frozen activity")
        if row["activity_label"] == 1:
            activity_reason_counts[row["reference_reason"]] += 1
        if row["stage_evaluable"] and row["feature"] == "unknown":
            raise ValueError("Unknown date entered stage evaluation")
    write_json(review_directory / "delivery-audit.json", {
        "status": "passed",
        "rows": len(daily),
        "activity_reason_counts": dict(activity_reason_counts),
        "public_files_sha256": {path.relative_to(public).as_posix(): sha(path)
                                for path in public.rglob("*") if path.is_file()},
        "source_kit_unchanged": verify_bundle(kit)[1] == source_digest,
        "browser_rendering": "not_verified",
        "performance_scores": None,
    })
    write_json(review_directory / "review-freeze.json", {
        "created_at": reviewed_at,
        "source_manifest_sha256": source_digest,
        "notes_sha256": {name: sha(review_directory / name) for name in NOTE_FILES},
        "corrections_sha256": sha(review_directory / "decision-corrections.json"),
        "implementation_sha256": sha(Path(__file__)),
        "submitted_review_sha256": sha(submission),
        "compiled_freeze_sha256": sha(review_directory / "compiled/reference-freeze.json"),
        "delivery_audit_sha256": sha(review_directory / "delivery-audit.json"),
        "reference_use": "development_only", "formal_reference_verified": False,
    })
    print(json.dumps({"review": str(review_directory), "viewed_images": len(viewed_images),
                      "reason_counts": dict(activity_reason_counts)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-directory", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    args = parser.parse_args()
    build(args.review_directory.resolve(), args.kit.resolve())


if __name__ == "__main__":
    main()
