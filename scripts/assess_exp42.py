"""Readable per-type BEV results, successful-pair audits and frozen-input checks."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import experiment_exp42 as experiment
from report_exp36 import compare
from report_range_variants import TYPES, score

from gnss_research.files import read_json, sha256, write_json


def assess(phase):
    folder = Path(f'results/exp42-{phase}-v1')
    summary = read_json(folder / 'summary.json')
    rows = read_json(folder / 'per-type.json')
    grouped = {method: {row['kind']: row for row in rows if row['method'] == method}
        for method in summary['methods']}
    with (folder / 'metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# EXP42 BEV-GNSS：{phase}逐类结果', '',
        f"{summary['stations']}站；不计算跨类型平均。pilot是18个已见开发站诊断，confirmation是108个新站。", '',
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
    lines.extend(['', 'Aff-F1为该类阳性站逐站F1均值，阴性误报另列。',
        'bmv_*在本目录指BEV变体；暂态由复核直接决定，无数值暂态回填。',
        '单点、阶跃、持续形变仍由数值组件输出，不能将其成绩归为视觉收益。',
        '图文共享视觉扫描候选，只隔离复核模态；去证据仍有数值候选坐标与放大窗口。',
        'vector_nearest是内部无视觉诊断，论文不必作为独立对比；无类型检测器标不适用。',
        '失败全部计分，不因格式或传输失败重试；不预称领域SOTA。'])
    (folder / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with gzip.open(folder / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        station_rows = json.load(stream)
    comparisons = {}
    for control in ('bmv_text', 'bmv_no_evidence', 'vector_nearest'):
        comparisons[control] = {'all_stations': compare(station_rows, 'bmv_image', control)}
        if control == 'vector_nearest':
            continue
        successful = []
        for record in experiment.records_for(phase):
            station = record['station_id']
            if all(read_json(experiment.ROOT / phase / 'review/predictions' / method
                    / station / 'result.json')['status'] == 'success' for method in ('bmv_image', control)):
                successful.append(station)
        filtered = {method: [row for row in values if row['id'] in successful]
            for method, values in station_rows.items()}
        comparisons[control]['successful_pairs'] = len(successful)
        comparisons[control]['paired'] = compare(filtered, 'bmv_image', control) if successful else None
    write_json(folder / 'contribution-audit.json', comparisons)
    paths = [experiment.baseline_root(phase), experiment.scan_root(phase),
        experiment.ROOT / phase / 'review']
    verification = []
    for path in paths:
        registration = read_json(path / 'registration.json')
        for filename, digest in registration['frozen_files'].items():
            verification.append({'path': filename, 'matches': sha256(Path(filename)) == digest})
        for job in registration['jobs']:
            request = Path(job['work']) / 'request.json'
            verification.append({'path': str(request), 'matches': sha256(request) == job['request_sha256']})
    write_json(folder / 'verification.json', {'checks': verification,
        'passed': all(item['matches'] for item in verification), 'new_model_requests': 0})
    print(json.dumps({'checks': len(verification), 'passed': all(item['matches'] for item in verification),
        'source_summary_sha256': sha256(folder / 'summary.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'confirmation'))
    assess(parser.parse_args().phase)
