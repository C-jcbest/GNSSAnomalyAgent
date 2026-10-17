"""Separate modality evidence from paired API availability, without replacing all-station scores."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from report_exp36 import aggregate_method, compare

from gnss_research.files import read_json, sha256, write_json


def analyze(source, raw):
    source_rows = source / 'per-station.json.gz'
    rows = json.loads(gzip.decompress(source_rows.read_bytes()))
    methods = ('vector_nearest', 'owv_flash', 'owv_flash_text',
        'owv_flash_pure', 'owv_flash_text_pure')
    groups = {'both_success': [], 'image_only_success': [], 'text_only_success': [], 'both_failed': []}
    for row in rows['owv_flash']:
        station = row['id']
        image = read_json(raw / 'validation-visual-v1/predictions/owv_flash' / station / 'result.json')
        text = read_json(raw / 'validation-visual-v1/predictions/owv_flash_text' / station / 'result.json')
        image_ok = image['status'] == 'success'
        text_ok = text['status'] == 'success'
        if image_ok and text_ok:
            group = 'both_success'
        elif image_ok:
            group = 'image_only_success'
        elif text_ok:
            group = 'text_only_success'
        else:
            group = 'both_failed'
        groups[group].append(station)
    matched = set(groups['both_success'])
    selected = {method: [row for row in rows[method] if row['id'] in matched] for method in methods}
    if not selected['owv_flash']:
        raise ValueError('no jointly successful stations for sensitivity analysis')
    scores = {method: aggregate_method(values) for method, values in selected.items()}
    paired = {'pure_image_vs_text': compare(selected, 'owv_flash_pure', 'owv_flash_text_pure'),
        'fusion_image_vs_text': compare(selected, 'owv_flash', 'owv_flash_text'),
        'fusion_image_vs_nearest': compare(selected, 'owv_flash', 'vector_nearest')}
    output = source / 'modality-sensitivity-v1'
    output.mkdir(parents=True, exist_ok=False)
    summary = {'source_sha256': sha256(source_rows), 'groups': groups, 'methods': scores,
        'paired': paired, 'scope': 'joint-success sensitivity only; all-station failure-inclusive scores remain primary'}
    write_json(output / 'summary.json', summary)
    lines = ['# 图文均成功站敏感性分析', '',
        '这是附加子集分析，不替代保留全部失败的主结果，也不改变原准入。', '',
        f"两者成功{len(matched)}站；仅图成功{len(groups['image_only_success'])}站；"
        f"仅文成功{len(groups['text_only_success'])}站；两者失败{len(groups['both_failed'])}站。", '',
        '| 方法 | 暂态Aff-F1 | 辅助日F1 | TP | FP | FN |',
        '|---|---:|---:|---:|---:|---:|']
    for method, value in scores.items():
        aff = value['range']['transient']['affiliation']['station_macro_f1']
        daily = value['original']['types']['transient']['daily_micro']
        lines.append(f"| {method} | {aff:.4f} | {daily['f1']:.4f} | "
            f"{daily['tp']} | {daily['fp']} | {daily['fn']} |")
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print({'groups': {key: len(value) for key, value in groups.items()},
        'paired': {key: {field: value[field] for field in ('aff_delta', 'tp_delta', 'fp_delta')}
            for key, value in paired.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.raw)
