"""Offline native-label scoring and a read-only HTML comparison; never invokes models."""
from __future__ import annotations

import html
import json
from pathlib import Path

from gnss_sim.artifacts import sha, write_json
from gnss_sim.detection import PROTOCOL, validate
from gnss_sim.metrics import evaluate
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult


def load_rows(path, task):
    model = PointResult if task == "point" else RangeResult
    if not path.exists():
        return {}
    rows = [model.model_validate_json(line) for line in path.read_bytes().splitlines() if line]
    if len({r.case_id for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction identity")
    return {r.case_id: r for r in rows}


def pretty(value):
    return "N/A" if value is None else f"{value:.4f}"


def export_html(out, report):
    labels = {"normal": "Normal", "global_extremum": "Global Extremum",
              "trend": "Trend", "mean_shift": "Transient Mean Shift"}
    table = []
    for method in ("numerical", "visual"):
        for task in ("point", "range"):
            r = report["summary"][method][task]
            m = r["daily"]
            table.append(f'<tr><th>{"数值" if method == "numerical" else "视觉"} / {task.title()}</th>'
                         + ''.join(f'<td>{pretty(m[k])}</td>' for k in ('precision', 'recall', 'f1'))
                         + f'<td>{m["tp"]} / {m["fp"]} / {m["fn"]}</td>'
                         + ''.join(f'<td>{pretty(r["affiliation"][k])}</td>'
                                   for k in ('precision', 'recall', 'f1'))
                         + f'<td>{r["normal"]["predicted_axis_days"]}</td><td>{r["failed_cases"]}</td></tr>')
    cards = []
    for row in report["cases"]:
        events = row["truth"]
        target = '；'.join(f'{e["axis"]} {e["task"]} [{e["start_index"]},{e["end_index"]}]'
                          for e in events) or '无异常'
        predictions = []
        for method in ("numerical", "visual"):
            for task in ("point", "range"):
                p = row["predictions"][method][task]
                text = json.dumps(p['predictions'], ensure_ascii=False) if p else '未产生结果'
                status = p['status'] if p else 'missing'
                predictions.append(f'<tr><th>{method} / {task}</th><td>{status}</td><td><code>{html.escape(text)}</code></td></tr>')
        cards.append(f'<details><summary>{row["case_id"]} · {labels[row["case_type"]]} · '
                     f'{row["background_group"]}　GT: {html.escape(target)}</summary>'
                     f'<table>{"".join(predictions)}</table>'
                     f'<img loading="lazy" src="data:image/png;base64,{row["image_base64"]}" '
                     f'alt="{row["case_id"]} 原始无标签检测图"></details>')
    visual = report['cost']['visual']
    grouped = []
    for kind in labels:
        for method in ('numerical', 'visual'):
            for task in ('point', 'range'):
                row = report['by_type'][kind][method][task]
                m = row['daily']
                grouped.append(f'<tr><th>{labels[kind]}</th><td>{method}</td><td>{task.title()}</td>'
                               + ''.join(f'<td>{pretty(m[k])}</td>'
                                         for k in ('precision', 'recall', 'f1'))
                               + f'<td>{m["tp"]}/{m["fp"]}/{m["fn"]}</td>'
                               + ''.join(f'<td>{pretty(row["affiliation"][k])}</td>'
                                         for k in ('precision', 'recall', 'f1')) + '</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>检测结果 · GNSS LAB</title><style>
body{margin:0;background:#f3f5f2;color:#233b42;font:15px/1.7 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 22px}
h1{font-size:30px;margin:8px 0}h2{font-size:21px;margin-top:32px}.eyebrow{letter-spacing:2px;color:#147d72;font-size:12px}
.note{padding:18px 22px;background:#e5eeea;border-left:4px solid #147d72}.table{overflow:auto}table{width:100%;border-collapse:collapse;background:white;margin:16px 0}
th,td{text-align:left;padding:12px;border-bottom:1px solid #dce4df;white-space:nowrap}th{font-weight:600}td code{white-space:normal;overflow-wrap:anywhere}
details{background:white;margin:12px 0;padding:16px;border:1px solid #dce4df}summary{cursor:pointer;font-weight:600}img{width:100%;height:auto}
a{color:#147d72}.muted{color:#65746e}footer{margin-top:30px;color:#65746e}</style><main>
<p class="eyebrow">GNSS LAB / DETECTION REPORT</p><h1>数值 × 视觉：检测结果</h1>
<p>同一批观测、同一套逐轴原生真值；全部方法处理全部输入。</p>
<div class="note">合成数据开发测试。数值使用独立 Normal 数据校准的参数；视觉使用 qwen3.8-flash，thinking=false。Point 仅孤立异常，Range 评价活动区间。数据采用合成背景，不是现场 GNSS 实测，也不是独立泛化测试。</div>
<p class="muted">Precision 是报警中命中的比例，Recall 是真值中被检出的比例，F1 综合两者。Affiliation 衡量预测与真值在时间上的接近程度；点按一天的区间计算。以下 Affiliation 三项分别在有真值的样例分量上取平均，不代替正常误报统计。</p>
<h2>总体结果</h2><div class="table"><table><thead><tr><th>方法 / 任务</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>Affiliation Precision</th><th>Affiliation Recall</th><th>Affiliation F1</th><th>Normal 误报轴日</th><th>失败例</th></tr></thead><tbody>'''
    page += ''.join(table) + '</tbody></table></div>'
    page += (f'<p class="muted">视觉实际请求 {visual.get("calls", 0)} 次，结构纠正 {visual.get("corrections", 0)} 次，'
             f'tokens {visual.get("total_tokens", 0):,}；金额未返回。失败/缺失阳性保留 FN，失败正常例不计为干净。'
             '无阳性组 F1 为 N/A，仍报告误报。Point ±3日仅作定位诊断，不替代精确日得分。</p>')
    page += '<h2>按形态查看</h2><div class="table"><table><tr><th>类型</th><th>方法</th><th>任务</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>Affiliation Precision</th><th>Affiliation Recall</th><th>Affiliation F1</th></tr>' + ''.join(grouped) + '</table></div>'
    page += '<h2>逐例核对</h2><p>日期使用 0–364 索引，区间两端均含。展开可查看原始无真值检测图，以及每个方法的实际输出。</p>' + ''.join(cards)
    page += '<footer>原始响应、请求账本、输入与报告保存在本次运行目录。本页为离线只读报告，不触发模型调用。</footer></main></html>'
    target = out / 'comparison.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(page)


def run(out):
    import base64

    if (out / "evaluation.json").exists() or (out / "comparison.html").exists():
        raise FileExistsError("Evaluation already exists")

    _, cases = validate(out)
    saved = json.loads((out / 'registered.json').read_bytes())
    if sha(out / 'evaluation-registration.json') != saved['evaluation_registration_sha256']:
        raise ValueError('Evaluation registration changed')
    registration = json.loads((out / 'evaluation-registration.json').read_bytes())
    for path, digest in registration['artifacts'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen evaluation data changed')
    dataset = Path(registration['dataset'])
    metadata = json.loads((dataset / 'manifest.json').read_bytes())['cases']
    truths = {c.case_id: CaseTruth.model_validate_json(
        (dataset / 'cases' / c.case_id / 'truth.json').read_bytes()) for c in cases}
    predictions = {m: {t: load_rows(out / m / f'{t}.jsonl', t) for t in ('point', 'range')}
                   for m in ('numerical', 'visual')}
    summary = {m: {t: evaluate(truths, predictions[m][t], t) for t in ('point', 'range')}
               for m in predictions}
    by_type = {}
    for kind in ('normal', 'global_extremum', 'trend', 'mean_shift'):
        selected = {r['case_id']: truths[r['case_id']] for r in metadata if r['case_type'] == kind}
        by_type[kind] = {m: {t: evaluate(selected, {cid: p for cid, p in predictions[m][t].items()
                                                 if cid in selected}, t) for t in ('point', 'range')}
                         for m in predictions}
    rows = []
    for entry in metadata:
        cid = entry['case_id']
        rows.append({**entry, 'truth': [e.model_dump(mode='json') for e in truths[cid].events],
                     'predictions': {m: {t: predictions[m][t][cid].model_dump(mode='json')
                                        if cid in predictions[m][t] else None
                                        for t in ('point', 'range')} for m in predictions}})
    cost = {m: json.loads((out / m / 'run.json').read_bytes()) if (out / m / 'run.json').exists()
            else {'status': 'missing'} for m in predictions}
    report = {'protocol': PROTOCOL, 'scope': registration['scope'],
              'summary': summary, 'by_type': by_type, 'cases': rows, 'cost': cost}
    write_json(out / 'evaluation.json', report)
    for row in rows:
        row['image_base64'] = base64.b64encode(
            (out / 'inputs/images' / f'{row["case_id"]}.png').read_bytes()).decode('ascii')
    export_html(out, report)
    return summary
