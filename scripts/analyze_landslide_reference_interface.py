"""Replay paired interfaces and display coverage/disagreement without accuracy claims."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_landslide_reference_interface import verify_previous_run  # noqa: E402

from gnss_sim.artifacts import sha  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_observation_review import activity_review_sha256  # noqa: E402
from gnss_sim.landslide_reference_draft import (  # noqa: E402
    activity_prompt,
    parse_activity_draft,
    unknown_rows,
)
from gnss_sim.landslide_reference_interface import (  # noqa: E402
    paired_activity_counts,
    parse_partition_draft,
    partition_prompt,
    partition_sha256,
    render_activity_pair,
)
from gnss_sim.landslide_trace import request_evidence  # noqa: E402
from gnss_sim.visual import _unique_object  # noqa: E402

ARMS = ("episode_v1", "partition_v2")
ARM_NAMES = {"episode_v1": "原接口 v1", "partition_v2": "单分区 v2"}

CONTENT_FINDINGS = [
    {"case_id": "case_0008", "kind": "weak_change_unresolved",
     "raw_image": "images/case_0008-raw-3.png",
     "finding": "v2将520–1094全列静稳，声称所有分量平台且无持续变化；685–1094局部原图的N可见数毫米缓变，E/U支持较弱，需复核。",
     "limit": "同会话作者图像复核，未裁定阳性日界，不按机制补标签。"},
    {"case_id": "case_0007", "kind": "plateau_evidence_mismatch",
     "raw_image": "images/case_0007-raw-2.png",
     "finding": "v2以E持续上升描述450–599活动，但局部图E约510–599主要平台、600附近回落；N/U可能另有小变化，不能仅据E概括全段。",
     "limit": "具体分量证据不支持覆盖整段的文字；不是对全部日期的独立负类判定。"},
    {"case_id": "case_0009", "kind": "task_attribution_drift",
     "raw_image": "images/case_0009-raw-0.png",
     "finding": "v2用离散step-like事件解释598–669持续活动，并以E/U缺少强相关而无法确认landslide activity解释670之后未知。实际任务是可观测持续位移，单阶跃不单独确认，且不要求先确定滑坡成因。",
     "limit": "提示／回答语义冲突可核查；区间内其他分量变化仍需原图判断，不据此自动改全段标签。"},
    {"case_id": "case_0011", "kind": "late_episode_unresolved",
     "raw_image": "images/case_0011-raw-3.png",
     "finding": "v2把940–1094全列未知，理由包括记录突然结束。原图末段包含再次上升与后续平台，不能以统一的截尾叙述代替逐段复核。",
     "limit": "未知未自动改活动；正式起止和活动内阶段等待可信参考。"},
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def build_page(summary, analysis, packets, public):
    def esc(value):
        return html.escape(str(value))

    headline_rows = []
    for arm in ARMS:
        item = analysis["arms"][arm]
        headline_rows.append(f'<tr><td>{ARM_NAMES[arm]}</td><td>{item["valid_records"]}/12</td>'
                             f'<td>{item["activity_days"]}</td><td>{item["stationary_days"]}</td>'
                             f'<td>{item["observed_unknown_days"]}</td>'
                             f'<td>{item["observed_coverage"]:.2%}</td></tr>')
    results = {(row["case_id"], row["arm"]): row for row in summary["results"]}
    findings_html = ''.join(
        f'<li><a href="#{item["case_id"]}">{item["case_id"]}</a>：{esc(item["finding"])} '
        f'<a href="{item["raw_image"]}" target="_blank">原图证据</a>。{esc(item["limit"])}</li>'
        for item in analysis["posthoc_content_findings"]
    )
    table_rows = []
    cards = []
    for case_id, pair in analysis["paired_cases"].items():
        status_cells = []
        sections = []
        for arm in ARMS:
            result = results[case_id, arm]
            status = "通过 · AI草案" if result["status"] == "valid_ai_draft" else "失败 · 全条未知"
            status_cells.append(f'<td>{status}<br>活动 {result["activity_counts"].get("1", 0)} 日</td>')
            packet = next(row for row in packets if row["case_id"] == case_id and row["arm"] == arm)
            receipt = read_json(public / "requests" / (packet["request_id"] + ".response.json"))
            inputs = ''.join(f'<a href="{esc(name)}" target="_blank"><img loading="lazy" src="{esc(name)}" '
                             f'alt="实际输入 {esc(Path(name).name)}"><span>{esc(Path(name).name)}</span></a>'
                             for name in packet["images"])
            details = json.dumps({"status": result["status"], "error": result["error"],
                                  "activity_counts": result["activity_counts"]}, ensure_ascii=False, indent=2)
            sections.append(f'<details class="request"><summary>{ARM_NAMES[arm]} · {status} · '
                            f'{esc(packet["request_id"])}</summary><p>同四张原图，按以下顺序实际发送；点击查看全尺寸。</p>'
                            f'<div class="inputs">{inputs}</div><h3>完整实际提示词</h3><pre>{esc(packet["prompt"])}</pre>'
                            f'<h3>原始回答（未修补）</h3><pre>{esc(receipt.get("output", "未返回完整回答"))}</pre>'
                            f'<h3>契约检查</h3><pre>{esc(details)}</pre><p>返回模型：{esc(receipt.get("returned_model"))}；'
                            f'finish：{esc(receipt.get("finish_reason"))}；token：{receipt.get("usage", {}).get("total_tokens", 0)}。'
                            f'<a href="requests/{esc(packet["request_id"])}.started.json">请求凭据</a> · '
                            f'<a href="requests/{esc(packet["request_id"])}.response.json">完整响应</a> · '
                            f'<a href="drafts/{esc(packet["request_id"])}.daily.json">逐日草案</a></p></details>')
        table_rows.append(f'<tr><td><a href="#{case_id}">{case_id}</a></td>{"".join(status_cells)}'
                          f'<td>{pair["state_disagreement_days"]}</td></tr>')
        cards.append(f'<section id="{case_id}" class="case"><h2>{case_id}</h2>'
                     f'<p>下图使用原图轴限与同一逐日数组，未送给模型。蓝色活动、浅绿静稳、黄色有观测未知、灰色缺测。</p>'
                     f'<a href="images/{case_id}-paired-activity.png" target="_blank"><img loading="lazy" '
                     f'src="images/{case_id}-paired-activity.png" alt="{case_id} 两组活动标签对照"></a>'
                     f'<details><summary>逐日状态转移（不是错误计数）</summary>'
                     f'<pre>{esc(json.dumps(pair, ensure_ascii=False, indent=2))}</pre></details>{"".join(sections)}</section>')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>原图活动接口同期对照</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f0;color:#20342f;max-width:1200px;margin:0 auto;padding:30px;line-height:1.65}}
h1{{font-size:30px}}h2{{font-size:23px}}h3{{font-size:16px}}a{{color:#286556}}section{{background:white;border:1px solid #d5ded3;border-radius:8px;padding:24px;margin:24px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dbe3d7}}.table{{overflow:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5ef;padding:18px;font-size:13px;line-height:1.7}}img{{width:100%;height:auto;display:block}}
.inputs{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.inputs span{{font-size:12px}}summary{{cursor:pointer;padding:10px 0}}.request{{border-top:1px solid #dbe3d7;margin-top:16px}}
.notice{{border-left:3px solid #a27836;padding-left:14px}}
@media(max-width:700px){{body{{padding:15px}}section{{padding:15px}}.inputs{{grid-template-columns:1fr}}}}</style></head><body>
<header><p>GNSS / PAIRED RAW ACTIVITY INTERFACES</p><h1>同原图：活动区间接口同期对照</h1>
<p>12 条完整开发记录 × 2 接口 × 单遍 = {summary["calls"]} 次真实调用；{summary["total_tokens"]} token；
{summary["service_seconds"]:.3f} 秒服务时长。交替平衡调用顺序；无重试；未进行阶段判别。</p>
<p class="notice">两组都是 AI 开发草案，缺少独立专家参考。接口通过、观测覆盖和两组一致性均不是准确率；
这批结果不能证明视觉优于数值方法或识别已改善。封存测试未解封。</p></header>
<section><h2>同期结果</h2><div class="table"><table><thead><tr><th>配置</th><th>契约通过</th><th>活动日</th><th>静稳日</th><th>有观测未知日</th><th>观测覆盖</th></tr></thead><tbody>{"".join(headline_rows)}</tbody></table></div>
<p>共同成功记录 {analysis["common_success_records"]}/12；全组差异 {analysis["all_observed_disagreement_days"]} 个观测日，
共同成功记录差异 {analysis["common_success_disagreement_days"]} 日。差异包含活动、静稳与未知之间的变化。</p>
<p>共同成功记录的观测覆盖：v1 {analysis["common_success_coverage"]["episode_v1"]:.2%}，v2 {analysis["common_success_coverage"]["partition_v2"]:.2%}。
v2在这部分仅净少赋值3日；全组覆盖增加主要来自恢复原接口失败记录，不能解释为内容质量提高。</p>
<p><a href="daily-activity-drafts.csv">下载两组逐日 CSV</a> · <a href="analysis.json">完整分析／核查</a> · <a href="protocol.json">固定配置</a> ·
<a href="http://127.0.0.1:18778/">历史 AI 草案与阶段问题</a> · <a href="http://127.0.0.1:18777/">无标注原始记录</a></p></section>
<section><h2>内容复核仍发现的问题</h2><p>以下为调用完成后的作者诊断，未发给模型、未改草案、不是独立专家标签。</p><ul>{findings_html}</ul></section>
<section><h2>全部记录</h2><div class="table"><table><thead><tr><th>记录</th><th>原接口 v1</th><th>单分区 v2</th><th>差异观测日</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
<p>v1 保留四端点活动对象＋独立静稳对象；v2 使用连续半开分区，明确填写未知。结构、示例和输出义务改变可能影响判读。</p></section>{"".join(cards)}</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/landslide-reference-interface-2026-10-08/run-v2")
    args = parser.parse_args()
    out, public = args.run, args.run / "public"
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["evidence_sha256"].items():
        assert sha(out / name) == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert sha(ROOT / name) == sha(out / "source" / name) == expected, name
    source = ROOT / manifest["source_packet"]
    for name, expected in manifest["source_packet_sha256"].items():
        assert sha(source / name) == expected, name
    verify_previous_run(ROOT / manifest["historical_run"])
    packets = read_json(public / "packets.json")
    summary = read_json(public / "drafts.json")
    results = {(row["case_id"], row["arm"]): row for row in summary["results"]}
    records = {row["case_id"]: row for row in read_json(public / "source-manifest.json")["cases"]}
    freeze = read_json(out / "activity-pass-freeze.json")
    for name, expected in freeze["drafts_sha256"].items():
        assert sha(public / "drafts" / name) == expected
    assert sha(public / "packets.json") == read_json(out / "run-ledger-start.json")["request_plan_sha256"]
    assert len(packets) == len(results) == 24
    with (public / "daily-activity-drafts.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 26280
    rows_by_case = {case_id: {} for case_id in records}
    arms = {arm: {"valid_records": 0, "failed_records": [], "counts": Counter()} for arm in ARMS}
    offset = 0
    for index, packet in enumerate(packets):
        case_id, arm = packet["case_id"], packet["arm"]
        case_number = index // 2 + 1
        expected_order = ARMS if case_number % 2 else tuple(reversed(ARMS))
        assert case_id == f"case_{case_number:04d}" and arm == expected_order[index % 2]
        record, result = records[case_id], results[case_id, arm]
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        motion = LandslideDiagnostics.model_validate_json((public / record["diagnostics"]["61"]["data"]).read_bytes())
        started = read_json(public / "requests" / (packet["request_id"] + ".started.json"))
        receipt = read_json(public / "requests" / (packet["request_id"] + ".response.json"))
        assets = request_evidence(started, public / "images", out / "request-replay-assets")
        assert [item["name"] for item in assets] == [Path(view["image"]).name for view in record["views"]]
        assert packet["images"] == [view["image"] for view in record["views"]]
        prompt_function = activity_prompt if arm == "episode_v1" else partition_prompt
        assert started["prompt"] == packet["prompt"] == prompt_function(case_id, len(case.dates) - 1, packet["images"])
        rows = unknown_rows(case)
        try:
            if receipt["status"] != "returned_json":
                raise RuntimeError("Saved transport did not return JSON")
            payload = json.loads(receipt["output"], object_pairs_hook=_unique_object)
            if arm == "episode_v1":
                review, rows = parse_activity_draft(payload, case, record, motion, started["model"])
                review_hash = activity_review_sha256(review)
                review_data = review.model_dump(mode="json")
            else:
                review_data, rows = parse_partition_draft(payload, case, record, started["model"])
                review_hash = partition_sha256(review_data)
            assert result["status"] == "valid_ai_draft"
            assert result["activity_sha256"] == review_hash
            assert read_json(public / result["review_file"]) == review_data
            arms[arm]["valid_records"] += 1
        except (RuntimeError, ValueError, TypeError, KeyError):
            assert result["status"] == "failed_unknown"
            rows = unknown_rows(case)
            arms[arm]["failed_records"].append({"case_id": case_id, "error": result["error"]})
        assert rows == read_json(public / f"drafts/{packet['request_id']}.daily.json")
        for row in rows:
            actual_csv = csv_rows[offset]
            actual_csv["day_index"] = int(actual_csv["day_index"])
            actual_csv["activity_label"] = int(actual_csv["activity_label"])
            assert actual_csv == {"arm": arm, **row}
            offset += 1
        arms[arm]["counts"].update(str(row["activity_label"]) for row in rows)
        assert result["activity_counts"] == dict(Counter(str(row["activity_label"]) for row in rows))
        assert all(row["feature"] == ("none" if row["activity_label"] == 0 else "unknown") for row in rows)
        rows_by_case[case_id][arm] = rows
    observed_days = sum(record["observed_days"] for record in records.values())
    missing_days = sum(record["missing_days"] for record in records.values())
    for data in arms.values():
        counts = data.pop("counts")
        data.update(activity_days=counts["1"], stationary_days=counts["0"],
                    unknown_days=counts["-1"], observed_unknown_days=counts["-1"] - missing_days,
                    observed_coverage=(counts["1"] + counts["0"]) / observed_days)
    paired = {}
    common_ids = []
    endpoint_conflicts = {}
    plot = read_json(public / "source-protocol.json")["plot"]
    for case_id, record in records.items():
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        observed = [row is not None for row in case.displacement_mm]
        rows = rows_by_case[case_id]
        pair = paired_activity_counts(rows["episode_v1"], rows["partition_v2"], observed)
        paired[case_id] = pair
        if all(results[case_id, arm]["status"] == "valid_ai_draft" for arm in ARMS):
            common_ids.append(case_id)
        if results[case_id, "episode_v1"]["status"] == "failed_unknown":
            receipt = read_json(public / "requests" / (case_id + "-episode_v1.response.json"))
            if receipt["status"] == "returned_json":
                payload = json.loads(receipt["output"], object_pairs_hook=_unique_object)
                conflicts = []
                for episode in payload.get("episodes", []):
                    for stable in payload.get("stable_ranges", []):
                        start = max(episode["possible_start"], stable["start"])
                        end = min(episode["possible_end"], stable["end"])
                        if start <= end:
                            conflicts.append([start, end])
                endpoint_conflicts[case_id] = conflicts
        path = public / f"images/{case_id}-paired-activity.png"
        strip = render_activity_pair(case, record, rows, path, plot)
        assert strip["label_image"] == [
            [row["activity_label"] + 1 if observed[day] else 3 for day, row in enumerate(rows[arm])]
            for arm in ARMS
        ]
    common_observed = sum(paired[case_id]["observed_days"] for case_id in common_ids)
    common_transitions = Counter()
    for case_id in common_ids:
        common_transitions.update(paired[case_id]["transitions_episode_to_partition"])
    common_coverage = {}
    for arm in ARMS:
        known = sum(row["activity_label"] != -1 for case_id in common_ids for row in rows_by_case[case_id][arm])
        common_coverage[arm] = known / common_observed if common_observed else None
    analysis = {"status": "VERIFIED request and daily replay; no independent accuracy reference",
                "arms": arms, "paired_cases": paired, "observed_days_per_arm": observed_days,
                "missing_days_per_arm": missing_days, "common_success_records": len(common_ids),
                "common_success_case_ids": common_ids,
                "common_success_observed_days": common_observed,
                "common_success_coverage": common_coverage,
                "common_success_transitions": dict(common_transitions),
                "v1_endpoint_conflicts": endpoint_conflicts,
                "posthoc_content_findings": CONTENT_FINDINGS,
                "all_observed_disagreement_days": sum(row["state_disagreement_days"] for row in paired.values()),
                "common_success_disagreement_days": sum(paired[case_id]["state_disagreement_days"] for case_id in common_ids),
                "verified_requests": 24, "verified_image_associations": 96, "verified_csv_rows": 26280,
                "performance_scores": None, "stages": None, "sealed_test_read": False,
                "total_tokens": summary["total_tokens"], "service_seconds": summary["service_seconds"],
                "all_returned_stop": all(read_json(public / "requests" / (packet["request_id"] + ".response.json")).get("finish_reason") == "stop" for packet in packets)}
    # Presentation/diagnostics are regenerable; frozen inference files were checked above.
    (public / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (public / "index.html").write_text(build_page(summary, analysis, packets, public), encoding="utf-8")
    audit = {
        "analysis_source_sha256": sha(Path(__file__)),
        "new_presentation_sha256": {path.relative_to(public).as_posix(): sha(path)
                                    for path in [public / "index.html", public / "analysis.json",
                                                 *sorted((public / "images").glob("*-paired-activity.png"))]},
        "evidence_unchanged": all(sha(out / name) == expected for name, expected in manifest["evidence_sha256"].items()),
    }
    (out / "analysis-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"arms": arms, "common_success": len(common_ids),
                      "disagreement_days": analysis["all_observed_disagreement_days"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
