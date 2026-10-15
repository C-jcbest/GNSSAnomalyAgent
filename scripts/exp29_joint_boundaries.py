"""Joint ordered endpoint selection conditioned on visually proposed ranges."""
from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from exp29_data_baseline import DATASET, ROOT
from experiment_exp29 import compose, transient_event
from report_exp25 import aggregate_rows, bootstrap, score_record

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmentation import regression

PROTOCOL = Path('docs/experiment-exp29-s1.md')
OUTPUT = ROOT / 'validation-joint-v1'


def joint_ranges(observation, numeric, decision):
    evidence = numerical_evidence(observation)
    eligible = evidence.observed.copy()
    eligible[:max(0, evidence.background_end - 89)] = False
    for event in numeric['prediction']['events']:
        if event['kind'] == 'spike':
            eligible[event['start']:event['end'] + 1] = False
    all_time = np.flatnonzero(eligible)
    all_values = (evidence.values / evidence.noise)[eligible]
    accepted, receipts = [], []
    for interval in decision.get('intervals', []):
        candidate_edges = []
        for target in (interval['start'], interval['end'] + 1):
            days = [day for day in range(max(evidence.background_end + 1, target - 14),
                    min(len(eligible) - 1, target + 14) + 1)
                    if eligible[day] and eligible[day - 1]]
            candidate_edges.append(days)
        context = ((all_time >= interval['start'] - 44)
                   & (all_time <= interval['end'] + 45))
        time, values = all_time[context], all_values[context]
        best, best_criterion, candidate_count = None, float('inf'), 0
        for first, second in itertools.product(*candidate_edges):
            if first >= second or (time < first).sum() < 5 or (time >= second).sum() < 5:
                continue
            candidate_count += 1
            jumps = [(time >= day).astype(float) for day in (first, second)]
            hinges = [np.maximum(time - day, 0) for day in (first, second)]
            nuisance_days = sorted({jump['index'] for jump in numeric.get('jumps', [])
                if len(time) and time[0] < jump['index'] < time[-1]
                and jump['index'] not in (first, second)})
            nuisance = [(time >= day).astype(float) for day in nuisance_days]
            columns = [*hinges, *nuisance]
            pulse = regression(time, values, [*columns, jumps[0] - jumps[1]])
            if pulse['criterion'] < best_criterion:
                best_criterion = pulse['criterion']
                continuous = regression(time, values, columns)
                independent = regression(time, values, [*columns, *jumps])
                best = {'first': first, 'second': second,
                    'criteria': {'pulse': pulse['criterion'], 'continuous': continuous['criterion'],
                                 'independent': independent['criterion']},
                    'nuisance_step_days': nuisance_days}
        receipt = {'visual_interval': interval, 'candidate_count': candidate_count,
                   'best': best, 'accepted': False}
        if best is not None:
            scores = best['criteria']
            receipt['accepted'] = scores['pulse'] < min(scores['continuous'], scores['independent'])
            if receipt['accepted']:
                accepted.append(transient_event(best['first'], best['second'] - 1,
                                                decision['evidence']))
        receipts.append(receipt)
    return accepted, receipts


def station_tools(record):
    station = record['station_id']
    observation = read_json(DATASET / record['observation'])
    visual = ROOT / 'validation-visual-v1'
    baseline = ROOT / 'validation-baseline-v1'
    numeric = read_json(baseline / 'predictions/segment_numeric' / station / 'result.json')
    strong = read_json(visual / 'final/vector_nearest' / station / 'result.json')
    decisions = {}
    for family in ('image', 'text'):
        receipt = read_json(visual / 'predictions' / ('vf_vl_' + family) / station / 'result.json')
        decisions['joint_vl_' + family] = receipt.get('range_decision',
            {'intervals': [], 'evidence': 'No validated first-stage proposals'})
    jump_days = sorted(jump['index'] for jump in numeric['jumps'])
    decisions['joint_numeric_all_pairs'] = {'intervals': [
        {'start': first, 'end': second - 1} for position, first in enumerate(jump_days)
        for second in jump_days[position + 1:]], 'evidence': 'Unrestricted numerical endpoint pairs'}
    for mode, decision in decisions.items():
        accepted, receipts = joint_ranges(observation, numeric, decision)
        work = OUTPUT / mode / station
        write_json(work / 'tool-evidence.json', {'receipts': receipts, 'accepted': accepted})
        write_json(work / 'result.json', compose(observation, strong, accepted))
    return station


def execute():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    records = [r for r in read_json(DATASET / 'manifest.json')['records'] if r['split'] == 'validation']
    reused = {str(p): sha256(p) for p in (ROOT / 'validation-visual-v1/predictions').rglob('result.json')}
    write_json(OUTPUT / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP29-S1',
        'records': records, 'frozen_files': {str(p): sha256(p) for p in (Path(__file__), PROTOCOL)},
        'source_hashes': source_hashes(), 'reused_response_sha256': reused,
        'new_requests': 0, 'manifest_sha256': sha256(DATASET / 'manifest.json')})
    with ProcessPoolExecutor(max_workers=2) as executor:
        for number, station in enumerate(executor.map(station_tools, records), 1):
            if number % 30 == 0:
                print('Joint tools', number, '/', len(records), station, flush=True)
    write_json(OUTPUT / 'execution-complete.json', {'stations': len(records), 'new_requests': 0})


def summarize():
    if not (OUTPUT / 'execution-complete.json').exists():
        raise ValueError('complete all joint tools first')
    registration = read_json(OUTPUT / 'registration.json')
    for path, digest in registration['frozen_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('joint sources changed')
    original = ROOT / 'validation-visual-v1/final'
    methods = {'joint_vl_image': OUTPUT, 'joint_vl_text': OUTPUT, 'joint_numeric_all_pairs': OUTPUT,
        'vector_nearest': original, 'vf_vl_image_complement': original,
        'vf_vl_image_raw_complement': original, 'vf_vl_text_raw_complement': original}
    rows = {mode: [] for mode in methods}
    for record in registration['records']:
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        for mode, folder in methods.items():
            result = read_json(folder / mode / record['station_id'] / 'result.json')
            rows[mode].append({'id': record['station_id'], 'scenario': truth['scenario'],
                'cohort_seed': record['cohort_seed'], **score_record(observation, truth, result)})
    summary = {mode: aggregate_rows(values) for mode, values in rows.items()}
    main = summary['joint_vl_image']['range']['transient']
    admitted = True
    checks = {}
    for control in ('vector_nearest', 'joint_vl_text'):
        other = summary[control]['range']['transient']
        aff_delta = main['affiliation']['station_macro_f1'] - other['affiliation']['station_macro_f1']
        daily_delta = main['daily_micro']['f1'] - other['daily_micro']['f1']
        checks[control] = {'aff_delta': aff_delta, 'daily_delta': daily_delta}
        admitted &= aff_delta >= .02 and daily_delta >= -.01
    by_seed = {}
    for seed in sorted({r['cohort_seed'] for r in rows['joint_vl_image']}):
        by_seed[str(seed)] = {mode: aggregate_rows([r for r in values if r['cohort_seed'] == seed])
                             for mode, values in rows.items()}
    positive_seeds = sum(values['joint_vl_image']['range']['transient']['affiliation']['station_macro_f1']
        > values['vector_nearest']['range']['transient']['affiliation']['station_macro_f1']
        for values in by_seed.values())
    added_tp = main['daily_micro']['tp'] - summary['vector_nearest']['range']['transient']['daily_micro']['tp']
    admitted &= positive_seeds >= 2 and added_tp > 0
    output = Path('results/exp29-joint-development-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', summary)
    write_json(output / 'seeds.json', by_seed)
    write_json(output / 'admission.json', {'admitted': bool(admitted), 'checks': checks,
        'positive_seeds': positive_seeds, 'added_tp_days': added_tp, 'new_requests': 0})
    write_json(output / 'paired.json', {mode: bootstrap(rows['joint_vl_image'], values)
        for mode, values in rows.items() if mode != 'joint_vl_image'})
    for mode, report in summary.items():
        metric = report['range']['transient']
        print(mode, 'Aff', round(metric['affiliation']['station_macro_f1'], 4),
            'daily', round(metric['daily_micro']['f1'], 4), 'FP', metric['daily_micro']['fp'],
            'four', round(report['original']['types_daily_macro_f1'], 4), flush=True)
    print('Admitted', admitted, 'added TP', added_tp, 'positive seeds', positive_seeds, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('execute', 'summarize'))
    args = parser.parse_args()
    {'execute': execute, 'summarize': summarize}[args.stage]()
