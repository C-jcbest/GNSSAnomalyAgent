"""Replay AI drafts and build an input/output review page; no accuracy scoring."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_evaluation import runs  # noqa: E402
from gnss_sim.landslide_reference_draft import (  # noqa: E402
    FEATURE_COLORS,
    activity_prompt,
    parse_activity_draft,
    parse_stage_draft,
    stage_prompt,
    unknown_rows,
)
from gnss_sim.landslide_trace import request_evidence  # noqa: E402
from gnss_sim.visual import _unique_object  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def closed_boundary_conflicts(payload) -> list[dict]:
    conflicts = []
    for episode in payload.get("episodes", []):
        for stable in payload.get("stable_ranges", []):
            start = max(episode["possible_start"], stable["start"])
            end = min(episode["possible_end"], stable["end"])
            if start <= end:
                conflicts.append({"days": [start, end],
                                  "episode_possible": [episode["possible_start"], episode["possible_end"]],
                                  "stable": [stable["start"], stable["end"]]})
    return conflicts


def page(summary, records, packets, public):
    def esc(value):
        return html.escape(str(value))

    status_labels = {
        "valid_ai_draft": "契约通过 · AI 草案", "failed_unknown": "失败 · 保留未知",
        "skipped_failed_activity": "活动失败，跳过", "not_applicable_no_confirmed_activity": "无确认活动，不适用",
        "failed_unknown_stages": "阶段失败 · 保留未知",
    }
    table_rows, cards = [], []
    for result in summary["cases"]:
        case_id = result["case_id"]
        stages = sum(result["feature_counts"].get(feature, 0) for feature in FEATURE_COLORS)
        table_rows.append(f'<tr><td><a href="#{case_id}">{case_id}</a></td>'
                          f'<td>{esc(status_labels[result["activity_status"]])}</td>'
                          f'<td>{esc(status_labels[result["stage_status"]])}</td>'
                          f'<td>{result["activity_counts"].get("1",0)}</td><td>{result["activity_counts"].get("0",0)}</td>'
                          f'<td>{result["activity_counts"].get("-1",0)}</td><td>{stages}</td></tr>')
        calls = []
        for packet in packets:
            if packet["case_id"] != case_id:
                continue
            receipt = read_json(public / "requests" / (packet["request_id"] + ".response.json"))
            sent_images = ''.join(
                f'<a href="{esc(name)}" target="_blank"><img loading="lazy" src="{esc(name)}" '
                f'alt="实际输入图 {esc(Path(name).name)}"><span>{esc(Path(name).name)}</span></a>'
                for name in packet["images"])
            calls.append(f'<details class="request"><summary>{esc(packet["request_id"])} · '
                         f'{esc(receipt["status"])} · {esc(receipt.get("finish_reason"))}</summary>'
                         f'<p>实际图片顺序如下，点击查看全尺寸。原始图片及哈希保留在请求记录。</p>'
                         f'<div class="inputs">{sent_images}</div><h3>完整实际提示词</h3>'
                         f'<pre>{esc(packet["prompt"])}</pre><h3>模型原始回答（未修补）</h3>'
                         f'<pre>{esc(receipt.get("output",""))}</pre>'
                         f'<p>返回模型：{esc(receipt.get("returned_model"))}；token：'
                         f'{esc(receipt.get("usage",{}).get("total_tokens"))}；'
                         f'服务时长：{receipt["seconds"]:.3f} 秒。'
                         f'<a href="requests/{esc(packet["request_id"])}.started.json">请求记录</a> · '
                         f'<a href="requests/{esc(packet["request_id"])}.response.json">原始响应</a></p></details>')
        record = records[case_id]
        raw_link = record["views"][0]["image"]
        errors = json.dumps(result["errors"], ensure_ascii=False, indent=2)
        download = ''
        if result.get("review_file"):
            download = f'<a href="{esc(result["review_file"])}">结构化 AI 审查</a> · '
        cards.append(f'<section id="{case_id}" class="case"><h2>{case_id}</h2>'
                     f'<p>活动：{esc(status_labels[result["activity_status"]])}；'
                     f'阶段：{esc(status_labels[result["stage_status"]])}。</p>'
                     f'<p>{download}<a href="drafts/{case_id}.daily.json">逐日草案 JSON</a> · '
                     f'<a href="{esc(raw_link)}" target="_blank">冻结原始位移图</a></p>'
                     f'<p class="notice">下图为未送模型的 AI 草案覆盖图。蓝底表示原图层确认活动；'
                     f'L/A/S/D 使用不同色带。白色可能是静稳、未知或缺测，请以逐日标签区别。</p>'
                     f'<a href="{esc(result["overlay"])}" target="_blank"><img loading="lazy" '
                     f'src="{esc(result["overlay"])}" alt="{case_id} AI 草案覆盖图"></a>'
                     f'<details><summary>契约失败记录与逐日计数</summary><pre>{esc(errors)}</pre>'
                     f'<pre>{esc(json.dumps({"activity":result["activity_counts"],"feature":result["feature_counts"]},ensure_ascii=False,indent=2))}</pre></details>'
                     f'{"".join(calls)}</section>')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI 活动／阶段草案审查</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f0;color:#20342f;max-width:1200px;margin:0 auto;padding:30px;line-height:1.65}}
h1{{font-size:30px}}h2{{font-size:23px}}h3{{font-size:16px}}a{{color:#286556}}.case,.summary{{background:white;border:1px solid #d5ded3;border-radius:8px;padding:24px;margin:24px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dbe3d7}}.table{{overflow:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5ef;padding:18px;font-size:13px;line-height:1.7}}img{{width:100%;height:auto;display:block}}
.inputs{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.inputs span{{font-size:12px}}summary{{cursor:pointer;padding:10px 0}}.request{{border-top:1px solid #dbe3d7;margin-top:16px}}
.notice{{border-left:3px solid #a27836;padding-left:14px}}.count{{display:inline-block;background:#e4ece3;padding:10px 16px;margin:4px}}
@media(max-width:700px){{body{{padding:15px}}.case,.summary{{padding:15px}}.inputs{{grid-template-columns:1fr}}}}</style></head><body>
<header><p>GNSS / OBSERVATION REVIEW TRACE</p><h1>原始位移活动 → 冻结范围 → 活动内阶段</h1>
<p class="notice">本页是单模型 AI 辅助开发草案，未经独立专家复核。契约通过只说明格式与层级合法，不能说明判断正确，也不作为性能真值。</p></header>
<div><span class="count">12 条开发记录</span><span class="count">{summary["calls"]} 次实际调用</span>
<span class="count">{summary["total_tokens"]} token</span><span class="count">新准确率评分：未进行</span></div>
<section class="summary"><h2>完整记录与失败</h2><p>闭区间端点冲突不自动修补；活动失败整条未知并跳过阶段。
无确认活动时阶段不适用；阶段失败保留第一层活动。未审／边界待定与缺测不默认为稳定。</p>
<div class="table"><table><thead><tr><th>记录</th><th>活动状态</th><th>阶段状态</th><th>活动日</th><th>静稳日</th><th>未知日</th><th>确定阶段日</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
<p><a href="daily-draft-labels.csv">下载全记录逐日 AI 草案 CSV</a> · <a href="diagnostics.json">契约诊断／核查</a> ·
<a href="protocol.json">本轮协议</a></p></section>
<section class="summary"><h2>需要重点复核的三类问题</h2><p>四条复杂记录存在闭区间共享端点，整个原图审查被拒绝；不能当作模型认为其没有活动。
部分回答把后段小变化统称为平台，弱活动支持尚未确认。另有阶段说明声称减速不属于允许类别，和实际提示词相矛盾。
以下实际答案均完整保留，作者诊断另存，不修改标签。</p></section>{"".join(cards)}</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/landslide-reference-draft-2026-10-08/run-v1")
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    out, public = args.run, args.run / "public"
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["evidence_sha256"].items():
        assert sha(out / name) == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    source = ROOT / manifest["source_packet"]
    for name, expected in manifest["source_packet_sha256"].items():
        assert sha(source / name) == expected, name
    summary = read_json(public / "drafts.json")
    packets = read_json(public / "packets.json")
    records = {row["case_id"]: row for row in read_json(public / "source-manifest.json")["cases"]}
    activity_pass = read_json(out / "activity-pass-freeze.json")
    for name, expected in activity_pass["reviews_sha256"].items():
        assert sha(public / "drafts" / name) == expected
    reviews, rows_by_case, failures = {}, {}, {}
    asset_dir = out / "request-replay-assets"
    verified_associations = 0
    saw_stages = False
    activity_calls = 0
    for packet in packets:
        case_id = packet["case_id"]
        case = LandslideInput.model_validate_json((public / records[case_id]["input"]).read_bytes())
        motion = LandslideDiagnostics.model_validate_json(
            (public / records[case_id]["diagnostics"]["61"]["data"]).read_bytes(),
        )
        receipt = read_json(public / "requests" / (packet["request_id"] + ".response.json"))
        started = read_json(public / "requests" / (packet["request_id"] + ".started.json"))
        images = request_evidence(started, public / "images", asset_dir)
        assert [image["name"] for image in images] == [Path(name).name for name in packet["images"]]
        verified_associations += len(images)
        assert started["prompt"] == packet["prompt"]
        if packet["layer"] == "activity":
            assert not saw_stages
            activity_calls += 1
            assert all("-raw-" in name for name in packet["images"])
            assert packet["prompt"] == activity_prompt(case_id, len(case.dates) - 1, packet["images"])
        else:
            saw_stages = True
            assert activity_calls == len(records)
            assert len(reviews) == len(activity_pass["reviews_sha256"])
            assert packet["prompt"] == stage_prompt(reviews[case_id], len(case.dates) - 1, packet["images"])
        payload = json.loads(receipt["output"], object_pairs_hook=_unique_object)
        try:
            if packet["layer"] == "activity":
                review, rows = parse_activity_draft(payload, case, records[case_id], motion, started["model"])
            else:
                review, rows = parse_stage_draft(payload, reviews[case_id], case, motion)
            reviews[case_id] = review
            rows_by_case[case_id] = rows
        except ValueError as error:
            assert packet["layer"] == "activity"
            failures[case_id] = {"error": str(error), "boundary_conflicts": closed_boundary_conflicts(payload)}
            rows_by_case[case_id] = unknown_rows(case)
    with (public / "daily-draft-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    assert len(saved_rows) == 13140
    offset = 0
    failed_observed_days = 0
    missing_days = 0
    for result in summary["cases"]:
        case_id = result["case_id"]
        rows = rows_by_case[case_id]
        assert rows == read_json(public / f"drafts/{case_id}.daily.json")
        case = LandslideInput.model_validate_json((public / records[case_id]["input"]).read_bytes())
        missing_days += sum(row is None for row in case.displacement_mm)
        for day, row in enumerate(rows):
            csv_row = {**saved_rows[offset + day], "activity_label":int(saved_rows[offset + day]["activity_label"]),
                       "day_index":int(saved_rows[offset + day]["day_index"])}
            assert csv_row == row
            if case.displacement_mm[day] is None:
                assert row["activity_label"] == -1 and row["feature"] == "unknown"
            if row["feature"] in FEATURE_COLORS:
                assert row["activity_label"] == 1
        offset += len(rows)
        assert result["activity_counts"] == dict(Counter(str(row["activity_label"]) for row in rows))
        assert result["feature_counts"] == dict(Counter(row["feature"] for row in rows))
        for feature in FEATURE_COLORS:
            mask = np.asarray([row["feature"] == feature for row in rows])
            assert result["mask_spans"][feature] == [[start, stop - 1] for start, stop in runs(mask)]
        activity = np.asarray([row["activity_label"] == 1 for row in rows])
        assert result["mask_spans"]["activity"] == [[start, stop - 1] for start, stop in runs(activity)]
        assert sha(public / result["overlay"]) == result["overlay_sha256"]
        if case_id in failures:
            failed_observed_days += sum(row is not None for row in case.displacement_mm)
    diagnostic = {"status":"VERIFIED actual request hashes and draft replay; not independent labels",
                  "calls":len(packets), "image_associations":verified_associations,
                  "all_transport_returned_stop":all(read_json(public/"requests"/(row["request_id"]+".response.json"))["finish_reason"]=="stop" for row in packets),
                  "failed_activity":failures, "failed_observed_days":failed_observed_days,
                  "all_daily_masks_replayed":True, "rows":len(saved_rows),
                  "observed_unknown_days":summary["activity_counts"]["-1"]-missing_days,
                  "performance_scores":None}
    verification_path = args.verification_output or public / "diagnostics.json"
    write_json(verification_path, diagnostic)
    (public / "index.html").write_text(page(summary, records, packets, public), encoding="utf-8")
    print(f"VERIFIED {len(packets)} requests, {verified_associations} image associations; review page built")


if __name__ == "__main__":
    main()
