"""Export each anomaly type independently from frozen scores, without type averages."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import EVENT_TYPES

TITLES = {'spike': '单点异常', 'step': '永久阶跃', 'transient': '暂态异常', 'deformation': '持续形变'}


def per_type_rows(summary):
    rows = []
    for method, scores in summary['methods'].items():
        typed = scores['original'].get('types')
        if typed is None:
            scope = 'generic_only'
        elif method.endswith('_pure'):
            scope = 'transient_only'
        else:
            scope = 'full_system'
        for kind in EVENT_TYPES:
            evaluated = typed is not None and (scope != 'transient_only' or kind == 'transient')
            row = {'method': method, 'scope': scope, 'kind': kind, 'evaluated': evaluated}
            daily = typed[kind]['daily_micro'] if evaluated else {}
            for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn'):
                row[key] = daily.get(key)
            ranges = scores.get('range') or {}
            metrics = ranges.get(kind) if evaluated else None
            affiliation = metrics['affiliation'] if metrics is not None else {}
            row.update(aff_precision=affiliation.get('precision'), aff_recall=affiliation.get('recall'),
                aff_f1=affiliation.get('station_macro_f1'),
                aff_positive_stations=affiliation.get('positive_stations'),
                negative_station_fp_days=metrics.get('negative_station_fp_days') if metrics is not None else None)
            range_anomaly = kind in ('transient', 'deformation')
            row['primary_metric'] = 'affiliation_f1' if range_anomaly else 'daily_f1'
            for key in ('precision', 'recall', 'f1'):
                row['primary_' + key] = row['aff_' + key] if range_anomaly else row[key]
            rows.append(row)
    return rows


def format_score(value):
    return '—' if value is None else f'{value:.4f}'


def export(summary_path, output):
    summary = read_json(summary_path)
    rows = per_type_rows(summary)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'per-type.json', {'experiment_id': summary['experiment_id'],
        'stations': summary['stations'], 'reporting_policy': 'point-daily-range-affiliation-v2',
        'source_summary': str(summary_path), 'source_summary_sha256': sha256(summary_path),
        'new_model_requests': 0, 'rows': rows})
    with (output / 'per-type.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {method: {row['kind']: row for row in rows if row['method'] == method}
        for method in summary['methods']}
    lines = ['# ' + summary['experiment_id'] + '：分类型统计', '',
        f"原批次{summary['stations']}站；读取冻结结果，新增模型请求0。每个异常类型独立统计，不报告跨类型平均。", '',
        '## 各类型主指标', '',
        '| 方法 | 输出范围 | 单点日F1 | 阶跃日F1 | 暂态Aff-F1 | 形变Aff-F1 |',
        '|---|---|---:|---:|---:|---:|']
    scope_names = {'full_system': '四类输出', 'transient_only': '仅暂态输出', 'generic_only': '仅通用异常'}
    for method, typed in grouped.items():
        scope = scope_names[typed[EVENT_TYPES[0]]['scope']]
        lines.append('| ' + method + ' | ' + scope + ' | ' + ' | '.join(
            format_score(typed[kind]['primary_f1']) for kind in EVENT_TYPES) + ' |')
    for kind in EVENT_TYPES:
        lines.extend(['', '## ' + TITLES[kind]])
        if kind in ('transient', 'deformation'):
            lines.extend(['', '| 方法 | Aff-P | Aff-R | Aff-F1 | 该类阳性站 | 无该类异常站FP日 |',
                '|---|---:|---:|---:|---:|---:|'])
            for method, typed in grouped.items():
                row = typed[kind]
                values = [format_score(row[key]) for key in ('aff_precision', 'aff_recall', 'aff_f1')]
                values.extend('—' if row[key] is None else str(row[key])
                    for key in ('aff_positive_stations', 'negative_station_fp_days'))
                lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
        lines.extend(['', '辅助日级统计：' if kind in ('transient', 'deformation') else '主评价：日级统计。', '',
            '| 方法 | 日Precision | 日Recall | 日F1 | TP | FP | FN |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for method, typed in grouped.items():
            row = typed[kind]
            counts = ['—' if row[key] is None else str(row[key]) for key in ('tp', 'fp', 'fn')]
            lines.append('| ' + method + ' | ' + ' | '.join(
                [format_score(row[key]) for key in ('precision', 'recall', 'f1')] + counts) + ' |')
    lines.extend(['', '“—”表示不适用。仅暂态输出的方法不评价其他三类，通用检测器不编造类型结果。',
        'Aff只对有该类真值的站聚合，需与该类全站日级指标及负站误报并读。',
        '形变沿用数值输出时不能将其成绩归为新增视觉作用；原冻结准入决定不因展示修正改变。'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print({'experiment_id': summary['experiment_id'], 'stations': summary['stations'],
        'methods': len(grouped), 'per_type_rows': len(rows), 'new_model_requests': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.summary, args.out)
