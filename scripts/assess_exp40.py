"""Audit paired image/text successes and true coverage of the frozen EXP40 outputs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from experiment_exp35 import aggregate_types

from gnss_research.files import read_json, sha256, write_json


def compare(rows, main, control, ids):
    current = [row for row in rows[main] if row['id'] in ids]
    previous = [row for row in rows[control] if row['id'] in ids]
    a, b = aggregate_types(current), aggregate_types(previous)
    changes = []
    reference = {row['id']: row for row in previous}
    for row in current:
        daily = row['original']['types']['transient']['daily']
        old = reference[row['id']]['original']['types']['transient']['daily']
        changes.append({'station': row['id'], 'scenario': row['scenario'], 'cohort_seed': row['cohort_seed'],
            'tp_delta': daily['tp'] - old['tp'], 'fp_delta': daily['fp'] - old['fp'],
            'fn_delta': daily['fn'] - old['fn']})
    return {'stations': len(ids), 'main': a, 'control': b,
        'aff_delta': a['range']['transient']['affiliation']['station_macro_f1']
            - b['range']['transient']['affiliation']['station_macro_f1'],
        **{key: sum(r[key] for r in changes) for key in ('tp_delta', 'fp_delta', 'fn_delta')},
        'tp_improved_stations': sum(r['tp_delta'] > 0 for r in changes),
        'tp_harmed_stations': sum(r['tp_delta'] < 0 for r in changes), 'changes': changes}


def assess(folder, artifacts):
    rows = json.loads(gzip.decompress((folder / 'per-station.json.gz').read_bytes()))
    ids = {row['id'] for row in rows['tlv_image']}
    successful, statuses = set(), {}
    for station in sorted(ids):
        pair = {mode: read_json(artifacts / 'review/predictions' / mode / station / 'result.json')['status']
            for mode in ('tlv_image', 'tlv_text')}
        statuses[station] = pair
        if all(value == 'success' for value in pair.values()):
            successful.add(station)
    output = {'scope': 'post-scoring contribution audit; no model calls or selection changes',
        'source_rows_sha256': sha256(folder / 'per-station.json.gz'), 'request_statuses': statuses,
        'all_stations': {control: compare(rows, 'tlv_image', control, ids)
            for control in ('vector_nearest', 'tlv_text', 'owv_union', 'owv_skip_overlap')},
        'paired_review_success_only': compare(rows, 'tlv_image', 'tlv_text', successful) if successful else None}
    write_json(folder / 'contribution-audit.json', output)
    print(json.dumps({'paired_successful_stations': len(successful),
        'all_station_deltas': {key: {k: v for k, v in value.items()
            if k in ('aff_delta', 'tp_delta', 'fp_delta', 'tp_improved_stations', 'tp_harmed_stations')}
            for key, value in output['all_stations'].items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('artifacts', type=Path)
    args = parser.parse_args()
    assess(args.results, args.artifacts)
