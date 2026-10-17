"""Whole-interval pulse/continuous-model competition after frozen visual proposals."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import experiment_exp41 as data
import numpy as np
from experiment_exp29 import plot_values
from report_exp25 import score_record
from report_exp36 import aggregate_method, compare

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmentation import regression

ROOT = Path('artifacts/exp44/pilot')
PROTOCOL = Path('docs/experiment-exp44.md')


def refine(values, candidate, jumps):
    left = max(365, candidate['start'] - 35)
    right = min(len(values), candidate['end'] + 36)
    indices = np.arange(left, right)
    observed = np.isfinite(values[left:right]).all(axis=1)
    time, samples = indices[observed], values[left:right][observed]
    onsets = time[np.abs(time - candidate['start']) <= 35]
    returns = time[np.abs(time - (candidate['end'] + 1)) <= 35]
    if not len(time):
        return {'accepted': False, 'reason': 'no observed context'}
    scaled_time = (time - time.mean()) / max(1, np.ptp(time))
    polynomial = [scaled_time ** 2, scaled_time ** 3]
    independent = [day for day in jumps if left < day < right]
    best_pulse, best_null, best_interval = np.inf, np.inf, None
    evaluated = 0
    for first in onsets:
        if (time < first).sum() < 5:
            continue
        for second in returns:
            if second <= first or (time >= second).sum() < 5 or ((time >= first) & (time < second)).sum() < 2:
                continue
            continuous = [*polynomial, np.maximum(time - first, 0), np.maximum(time - second, 0)]
            null_columns = [*continuous, *[(time >= day).astype(float) for day in independent]]
            null = regression(time, samples, null_columns)['criterion']
            nuisance = [day for day in independent if day not in (first, second)]
            pulse_columns = [*continuous, *[(time >= day).astype(float) for day in nuisance],
                ((time >= first) & (time < second)).astype(float)]
            pulse = regression(time, samples, pulse_columns)['criterion']
            best_null = min(best_null, null)
            if pulse < best_pulse:
                best_pulse = pulse
                best_interval = (int(first), int(time[time < second][-1]))
            evaluated += 1
    if best_interval is None:
        return {'accepted': False, 'reason': 'insufficient observed support'}
    return {'accepted': best_pulse < best_null, 'start': best_interval[0], 'end': best_interval[1],
        'pulse_criterion': float(best_pulse), 'null_criterion': float(best_null),
        'candidate': candidate, 'pairs_evaluated': evaluated,
        'context': [left, right - 1]}


def events_from_decisions(base, decisions):
    events = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] != 'transient']
    intervals = sorted((item['start'], item['end']) for item in decisions if item['accepted'])
    merged = []
    for first, last in intervals:
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    events.extend({'kind': 'transient', 'start': first, 'end': last, 'status': 'confirmed',
        'phases': [], 'evidence': 'Visual/numerical candidate; whole-interval pulse model preferred'}
        for first, last in merged)
    return events


def run():
    ROOT.mkdir(parents=True, exist_ok=False)
    records = data.records_for('pilot')
    write_json(ROOT / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP44-PILOT',
        'records': records, 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in [Path(__file__), PROTOCOL,
            Path('scripts/experiment_exp41.py')]}, 'new_requests': 0})
    methods = ('vmc_image', 'vmc_numeric_only', 'scan_only')
    rows = {method: [] for method in methods}
    upstream = []
    for index, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(data.DATASET / record['observation'])
        paths = [Path('artifacts/exp41/pilot/baseline/predictions') / method / station / 'result.json'
            for method in ('vector_nearest', 'segment_numeric')]
        scan_path = Path('artifacts/exp41/pilot/scan/predictions') / station / 'result.json'
        base, segmented, scan = [read_json(p) for p in [*paths, scan_path]]
        upstream.extend({'path': str(p), 'sha256': sha256(p)} for p in [*paths, scan_path])
        numeric = [e for e in base['prediction']['events'] if e['kind'] == 'transient']
        visual = data.data.confirmed_events(scan, 'owv_image')
        unique = {(e['start'], e['end']): e for e in [*numeric, *visual]}
        values = plot_values(observation)
        fits = {key: refine(values, candidate, [j['index'] for j in segmented['jumps']])
            for key, candidate in unique.items()}
        write_json(ROOT / 'fits' / (station + '.json'), list(fits.values()))
        predictions = {'vmc_image': data.data.result_from_events(observation,
            events_from_decisions(base, list(fits.values()))),
            'vmc_numeric_only': data.data.result_from_events(observation, events_from_decisions(base,
                [fits[(e['start'], e['end'])] for e in numeric])),
            'scan_only': data.visual_owned(observation, base, scan)}
        truth = read_json(data.DATASET / record['truth'])
        for method, prediction in predictions.items():
            rows[method].append({'id': station, 'scenario': truth['scenario'],
                'cohort_seed': record['cohort_seed'], 'events': prediction['prediction'],
                **score_record(observation, truth, prediction)})
        print('Interval model competition', index, '/', len(records), station, flush=True)
    write_json(ROOT / 'upstream-hashes.json', upstream)
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    main, numeric, scan = (summaries[m] for m in methods)

    def aff(scores):
        return scores['range']['transient']['affiliation']['station_macro_f1']

    def tp(scores):
        return scores['original']['types']['transient']['daily_micro']['tp']

    checks = {'aff_vs_numeric': aff(main) - aff(numeric) >= .01,
        'tp_vs_numeric': tp(main) > tp(numeric), 'aff_preserved_vs_scan': aff(main) - aff(scan) >= -.02,
        'negative_fp_halved': main['range']['transient']['negative_station_fp_days']
            <= .5 * scan['range']['transient']['negative_station_fp_days']}
    target = Path('results/exp44-pilot-v1')
    target.mkdir(parents=True, exist_ok=False)
    summary = {'experiment_id': 'GNSS-VLM-EXP44-PILOT', 'stations': len(records),
        'methods': summaries, 'new_requests': 0, 'reused_scan_requests': 18}
    write_json(target / 'summary.json', summary)
    write_json(target / 'per-type.json', data.data.typed_rows(summary))
    write_json(target / 'admission.json', {'admitted': all(checks.values()), 'checks': checks})
    write_json(target / 'contribution.json', compare(rows, 'vmc_image', 'vmc_numeric_only'))
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'checks': checks, 'transient': {m: summaries[m]['range']['transient']
        for m in methods}}, indent=2), flush=True)


if __name__ == '__main__':
    run()
