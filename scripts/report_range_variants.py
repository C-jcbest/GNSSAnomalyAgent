"""Display already scored range variants without cross-type averages or re-inference."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json

TYPES = ('spike', 'step', 'transient', 'deformation')
TITLES = {'spike': '单点异常', 'step': '永久阶跃', 'transient': '暂态异常', 'deformation': '持续形变'}


def score(value):
    return '—' if value is None else f'{value:.4f}'


def report(folder):
    source = folder / 'per-type.json'
    rows = read_json(source)
    summary = read_json(folder / 'summary.json')
    grouped = {method: {r['kind']: r for r in rows if r['method'] == method}
        for method in summary['methods']}
    output = folder / 'per-type-display'
    output.mkdir(exist_ok=False)
    with (output / 'metrics.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# ' + summary['experiment_id'] + '：按细分异常评价', '',
        f"本批{summary['stations']}站；只读取既定逐类结果，新增请求0，不计算跨异常类别平均。", '',
        '| 方法 | 单点日F1 | 阶跃日F1 | 暂态Aff-F1 | 形变Aff-F1 |',
        '|---|---:|---:|---:|---:|']
    for method, typed in grouped.items():
        lines.append('| ' + method + ' | ' + ' | '.join(score(typed[k]['aff_f1']
            if k in ('transient', 'deformation') else typed[k]['f1']) for k in TYPES) + ' |')
    for kind in TYPES:
        lines.extend(['', '## ' + TITLES[kind], ''])
        if kind in ('transient', 'deformation'):
            lines.extend(['| 方法 | Aff-P | Aff-R | Aff-F1 | 阳性站 | 阴性站FP日 |',
                '|---|---:|---:|---:|---:|---:|'])
            for method, typed in grouped.items():
                row = typed[kind]
                values = [score(row[key]) for key in ('aff_precision', 'aff_recall', 'aff_f1')]
                values += ['—' if row[key] is None else str(row[key])
                    for key in ('positive_stations', 'negative_station_fp_days')]
                lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
            lines.extend(['', '辅助日级统计：', ''])
        lines.extend(['| 方法 | 日P | 日R | 日F1 | TP | FP | FN |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for method, typed in grouped.items():
            row = typed[kind]
            values = [score(row[key]) for key in ('precision', 'recall', 'f1')]
            values += ['—' if row[key] is None else str(row[key]) for key in ('tp', 'fp', 'fn')]
            lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
    lines.extend(['', 'Aff-F1是该类阳性站逐站F1均值，阴性站不入Aff；需与误报和日指标并读。',
        '“—”表示不适用。通用检测器不编造类型分数；条件复核输出有候选输入，不称无候选纯视觉。',
        'EXP39/40图文消融共享上游视觉候选，只隔离复核阶段模态；数值对照完全不使用视觉。',
        'EXP40的单点、阶跃、持续形变由冻结数值方法输出，不能算作新视觉贡献。',
        '开发诊断、开发确认与独立测试分开解释；本组方法最优不等于领域SOTA。'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'provenance.json', {'summary_sha256': sha256(folder / 'summary.json'),
        'per_type_sha256': sha256(source), 'new_model_requests': 0,
        'policy': 'point daily F1; separate range affiliation F1 plus daily F1'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='+', type=Path)
    for folder in parser.parse_args().folders:
        report(folder)
