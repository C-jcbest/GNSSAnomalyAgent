"""Export all scored types and paired-review sensitivity without new inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from report_exp36 import compare
from report_range_variants import TYPES, score

from gnss_research.files import read_json, sha256, write_json


def assess(phase):
    folder = Path(f'results/exp41-{phase}-v1')
    summary = read_json(folder / 'summary.json')
    rows = read_json(folder / 'per-type.json')
    grouped = {method: {row['kind']: row for row in rows if row['method'] == method}
        for method in summary['methods']}
    with (folder / 'metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# EXP41 BMV-GNSS：{phase}逐类结果', '',
        f"{summary['stations']}站；各类型分别统计，无跨类型平均。", '',
        '| 方法 | 单点日F1 | 阶跃日F1 | 暂态Aff-F1 | 形变Aff-F1 |',
        '|---|---:|---:|---:|---:|']
    for method, typed in grouped.items():
        values = [score(typed[kind]['aff_f1' if kind in ('transient', 'deformation') else 'f1'])
            for kind in TYPES]
        lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
    for kind in TYPES:
        lines.extend(['', '## ' + kind, '',
            '| 方法 | 日P | 日R | 日F1 | TP | FP | FN | Aff-P | Aff-R | Aff-F1 | 阳性站 | 阴性FP日 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for method, typed in grouped.items():
            row = typed[kind]
            values = [score(row[key]) for key in ('precision', 'recall', 'f1')]
            values.extend('—' if row[key] is None else str(row[key]) for key in ('tp', 'fp', 'fn'))
            values.extend(score(row[key]) for key in ('aff_precision', 'aff_recall', 'aff_f1'))
            values.extend('—' if row[key] is None else str(row[key])
                for key in ('positive_stations', 'negative_station_fp_days'))
            lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
    lines.extend(['', '## 解释边界', '',
        'Aff-F1是该类阳性站逐站F1均值，阴性站误报另列。“—”是不适用。',
        '暂态由复核直接决定，无数值暂态回填；单点、阶跃、形变仍由数值组件输出。',
        '图文及去证据共享视觉扫描候选，仅隔离复核阶段；vector_nearest是完全无视觉内部诊断。',
        'bmv_no_evidence仍有数值候选，只移除局部模型/向量证据，不称去除全部数值。',
        '机制仿真开发结果不等于现场泛化或领域SOTA；全部失败计分，原始请求不重试。'])
    (folder / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with gzip.open(folder / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        station_rows = json.load(stream)
    paired = {}
    for control in ('bmv_text', 'bmv_no_evidence'):
        stations = []
        for row in station_rows['bmv_image']:
            paths = [Path(f'artifacts/exp41/{phase}/review/predictions') / method
                / row['id'] / 'result.json' for method in ('bmv_image', control)]
            if all(read_json(path)['status'] == 'success' for path in paths):
                stations.append(row['id'])
        filtered = {method: [row for row in values if row['id'] in stations]
            for method, values in station_rows.items()}
        paired[control] = {'successful_pairs': len(stations), 'stations': stations,
            'comparison': compare(filtered, 'bmv_image', control) if stations else None}
    write_json(folder / 'paired-success-audit.json', paired)

    frozen = []
    requests = []
    for stage in ('baseline', 'scan', 'review'):
        registration = read_json(Path(f'artifacts/exp41/{phase}/{stage}/registration.json'))
        for path, digest in registration['frozen_files'].items():
            frozen.append({'path': path, 'sha256': digest, 'matches': sha256(Path(path)) == digest})
        for job in registration['jobs']:
            path = Path(job['work']) / 'request.json'
            requests.append({'path': str(path), 'matches': sha256(path) == job['request_sha256']})
    write_json(folder / 'verification.json', {'frozen_checks': frozen, 'request_checks': requests,
        'passed': all(item['matches'] for item in [*frozen, *requests]),
        'source_summary_sha256': sha256(folder / 'summary.json'), 'new_model_requests': 0})
    print(json.dumps({'phase': phase, 'methods': len(grouped),
        'frozen_checks': len(frozen), 'request_checks': len(requests),
        'paired': {key: value['successful_pairs'] for key, value in paired.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'confirmation'))
    assess(parser.parse_args().phase)
