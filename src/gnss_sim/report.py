"""Offline native-label scoring and a read-only HTML comparison; never invokes models."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from gnss_sim.artifacts import sha, write_json
from gnss_sim.detection import PROTOCOL, validate
from gnss_sim.metrics import AXES, evaluate, range_days, target_days
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult

METHODS = ("numerical", "visual", "union", "intersection")
METHOD_LABELS = dict(zip(METHODS, ("数值 N", "视觉 V", "并集 U", "交集 I")))


def intervals(days):
    result = []
    for day in sorted(days):
        if result and day == result[-1][1] + 1:
            result[-1][1] = day
        else:
            result.append([day, day])
    return result


def combine(left, right, case_ids, task, method):
    """Observation-derived predictions only; both branches must succeed."""
    if method not in ("union", "intersection") or task not in ("point", "range"):
        raise ValueError("Invalid combination")
    model = PointResult if task == "point" else RangeResult
    results = {}
    for cid in case_ids:
        a, b = left.get(cid), right.get(cid)
        valid = all(row is not None and row.status == "success" for row in (a, b))
        pred = {axis: [] for axis in AXES}
        if valid:
            for axis in AXES:
                av, bv = getattr(a.predictions, axis), getattr(b.predictions, axis)
                x, y = (set(av), set(bv)) if task == "point" else (range_days(av), range_days(bv))
                days = x | y if method == "union" else x & y
                pred[axis] = sorted(days) if task == "point" else intervals(days)
        results[cid] = model(case_id=cid, method=method,
                             status="success" if valid else "failed", predictions=pred)
    return results


def paired_analysis(truths, predictions, repeats=2000, seed=2026093003,
                    methods=METHODS, comparisons=None):
    """Descriptive paired cluster bootstrap; fixed draws shared by every method."""
    groups = sorted({t.background_group for t in truths.values()})
    group_index = {g: i for i, g in enumerate(groups)}
    draws = np.random.default_rng(seed).integers(len(groups), size=(repeats, len(groups)))
    output = {"bootstrap_unit": "background_group", "groups": len(groups),
              "repeats": repeats, "seed": seed, "level": .95,
              "interpretation": "exploratory percentile intervals; no multiplicity adjustment",
              "tasks": {}}
    for task in ("point", "range"):
        values = {}
        for method in methods:
            counts = np.zeros((len(groups), 3))
            aff = np.zeros((len(groups), 4))
            normal = np.zeros((len(groups), 3))
            for cid, truth in truths.items():
                row = predictions[method][task].get(cid)
                score = evaluate({cid: truth}, {cid: row} if row is not None else {}, task)
                i = group_index[truth.background_group]
                counts[i] += [score["daily"][k] for k in ("tp", "fp", "fn")]
                a = score["affiliation"]
                aff[i] += [(a[k] or 0) * a["positive_axes"] for k in
                           ("precision", "recall", "f1")] + [a["positive_axes"]]
                n = score["normal"]
                normal[i] += [n["alarm_cases"], n["failed_cases"], n["cases"]]
            c = counts[draws].sum(axis=1)
            a = aff[draws].sum(axis=1)
            n = normal[draws].sum(axis=1)
            def divide(x, y, default):
                return np.divide(x, y, out=np.full_like(x, default), where=y != 0)
            values[method] = {
                "precision": divide(c[:, 0], c[:, 0] + c[:, 1], 0.),
                "recall": divide(c[:, 0], c[:, 0] + c[:, 2], np.nan),
                "f1": divide(2*c[:, 0], 2*c[:, 0]+c[:, 1]+c[:, 2], np.nan),
                **{f"affiliation_{k}": divide(a[:, j], a[:, 3], np.nan)
                   for j, k in enumerate(("precision", "recall", "f1"))},
                "normal_alarm_fraction_all": divide(n[:, 0], n[:, 2], np.nan),
                "normal_failure_fraction": divide(n[:, 1], n[:, 2], np.nan)}
        def bounds(v):
            v = v[np.isfinite(v)]
            return {"low": float(np.quantile(v, .025)) if len(v) else None,
                    "high": float(np.quantile(v, .975)) if len(v) else None,
                    "valid_replicates": len(v)}
        pairs = comparisons or (("visual", "numerical"), ("union", "numerical"),
                                ("intersection", "numerical"), ("union", "visual"),
                                ("intersection", "visual"))
        complement = {k: 0 for k in ("paired_success_cases", "excluded_cases",
                                    "shared_tp", "numerical_only_tp", "visual_only_tp",
                                    "shared_fp", "numerical_only_fp", "visual_only_fp")}
        for cid, truth in truths.items():
            rows = [predictions[m][task].get(cid) for m in METHODS[:2]]
            if not all(r is not None and r.status == "success" for r in rows):
                complement["excluded_cases"] += 1
                continue
            complement["paired_success_cases"] += 1
            for axis in AXES:
                pred = [getattr(r.predictions, axis) for r in rows]
                x, y = [set(v) if task == "point" else range_days(v) for v in pred]
                gt = target_days(truth, task, axis)
                for label, days in (("shared", x & y), ("numerical_only", x-y),
                                    ("visual_only", y-x)):
                    complement[label + "_tp"] += len(days & gt)
                    complement[label + "_fp"] += len(days - gt)
        output["tasks"][task] = {
            "method_intervals": {m: {k: bounds(v) for k, v in vs.items()}
                                 for m, vs in values.items()},
            "paired_differences": {f"{a}_minus_{b}": {
                k: bounds(values[a][k] - values[b][k]) for k in values[a]}
                for a, b in pairs}, "complementarity_axis_days": complement}
    return output


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


def export_html(out, report, methods=METHODS, method_labels=METHOD_LABELS):
    labels = {"normal": "Normal", "global_extremum": "Global Extremum",
              "trend": "Trend", "mean_shift": "Transient Mean Shift"}
    table = []
    for method in methods:
        for task in ("point", "range"):
            r = report["summary"][method][task]
            m = r["daily"]
            table.append(f'<tr><th>{method_labels[method]} / {task.title()}</th>'
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
        for method in methods:
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
        for method in methods:
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
    page += '<p>U/I按每轴日期集合取并集/交集，不调用模型；任一路失败或缺失均传播为融合失败。配对背景组置信区间与互补统计见 evaluation.json 的 paired_analysis。</p>'
    page += (f'<p class="muted">视觉实际请求 {visual.get("calls", 0)} 次，结构纠正 {visual.get("corrections", 0)} 次，'
             f'tokens {visual.get("total_tokens", 0):,}；金额未返回。失败/缺失阳性保留 FN，失败正常例不计为干净。'
             '无阳性组 F1 为 N/A，仍报告误报。Point ±3日仅作定位诊断，不替代精确日得分。</p>')
    if 'langchain_agent' in methods or 'localization_agent' in methods or 'window_agent' in methods:
        page += ('<h2>模型调用成本</h2><p>复核列仅为新增成本；完整融合还依赖一次独立视觉检测。'
                 '复核组共享候选及原始全年图，每任务两轮语义请求、最多一次结构纠正。'
                 '无候选直接空预测。自主查询与固定复查的第一轮功能不同，不能仅由差值证明自主性。'
                 '候选全部保留是相同候选支持集合的无模型对照。'
                 '边界收缩智能体仅允许在原Range候选内收缩，不增加候选或向外扩展。</p>'
                 '<table><tr><th>方法</th><th>请求数</th><th>Tokens</th><th>耗时/秒</th></tr>')
        for name in (m for m in methods if m in report['cost'] and m != 'numerical'):
            c = report['cost'][name]
            page += (f'<tr><th>{method_labels[name]}</th><td>{c["calls"]}</td>'
                     f'<td>{c["total_tokens"]}</td><td>{c["wall_seconds"]:.1f}</td></tr>')
        page += '</table>'
        if 'localization_agent' in methods:
            page += ('<p>定位工具智能体与工具固定评分使用相同的观测边界方案；'
                     '定位工具允许在候选起止各±31日内搜索，不能新增候选区域。'
                     '噪声尺度来自独立Normal校准，近优边界跨度不是置信区间。'
                     '本轮同时改变专门工具和方案ID输出，结论针对完整策略。'
                     'Point沿用原筛选逻辑，不将差异归因于Range定位。</p>')
        if 'hypothesis_point_agent' in methods:
            page += ('<p>Point双解释实验：只改变Point判定说明，比较真实孤立异常与正常背景偶然尖峰。'
                     '四个Point复核组共享全年与局部图、候选和轮数；数值工具及阈值不变。'
                     '主比较是新旧工具智能体Point F1，并要求本批真点检出数不下降。'
                     '四组Range均复制同一窗口智能体的冻结结果，含失败，不能作为Point改进证据。</p>')
        elif 'local_point_agent' in methods:
            page += ('<p>Point局部图实验：全年图保留，仅增加每个数值候选前后14日原始观测图。'
                      '仅视觉和数值工具复核各有全年/局部对照，提示与轮数相同。'
                      '两局部组的Range分别共享对应全年组的冻结结果，含失败。'
                      '主比较为局部数值智能体减全年数值智能体的Point F1，'
                      '并检查真实点误删、Recall、正常误报及失败。</p>')
        elif 'window_agent' in methods:
            page += ('<p>窗口搜索实验只改变端点搜索域：旧定位组分别在原端点±31日搜索；'
                     '新组允许两个端点在同一观测上下文内任意移动，拟合、评分和模型提示相同。'
                     '主比较为新旧智能体，另比较新旧工具固定评分及相同方案下的模型确认。'
                     '新旧定位智能体共用同一次Point筛选结果，含失败，不再次请求Point。</p>')
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
    for method in METHODS[2:]:
        predictions[method] = {t: combine(predictions['numerical'][t], predictions['visual'][t],
                                         truths, t, method) for t in ('point', 'range')}
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
    for method in METHODS[2:]:
        cost[method] = {'status': 'offline', 'additional_model_calls': 0,
                        'requires': ['numerical', 'visual']}
    report = {'protocol': PROTOCOL, 'scope': registration['scope'],
              'summary': summary, 'by_type': by_type, 'cases': rows, 'cost': cost,
              'paired_analysis': paired_analysis(truths, predictions)}
    write_json(out / 'evaluation.json', report)
    for row in rows:
        row['image_base64'] = base64.b64encode(
            (out / 'inputs/images' / f'{row["case_id"]}.png').read_bytes()).decode('ascii')
    export_html(out, report)
    return summary
