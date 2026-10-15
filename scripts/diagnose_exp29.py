"""Trace where visual evidence changes detections or is removed by later stages."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from exp14_fingerprint import free_graph
from exp29_data_baseline import DATASET, ROOT

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values


def counts(events, target, observed):
    mask = event_masks(Prediction.model_validate({'events': events}), observed)['transient']
    return {'tp_days': int((mask & target).sum()), 'fp_days': int((mask & ~target).sum())}


def diagnose(split):
    visual = ROOT / f'{split}-visual-v1'
    baseline = ROOT / f'{split}-baseline-v1'
    if not (visual / 'remaining-materialized.json').exists():
        raise ValueError('complete full materialization before diagnosis')
    records = [r for r in read_json(DATASET / 'manifest.json')['records'] if r['split'] == split]
    modes = ('vf_vl_image', 'vf_vl_text')
    totals = {mode: Counter() for mode in modes}
    stations = []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = event_masks(Prediction.model_validate({'events': truth['events']}), observed)['transient']
        numeric = read_json(baseline / 'predictions/segment_numeric' / station / 'result.json')
        graph = free_graph(numeric)
        free_edges = {j['index'] for j in graph['jumps']}
        all_edges = {j['index'] for j in numeric['jumps']}
        strong = read_json(visual / 'final/vector_nearest' / station / 'result.json')
        strong_mask = event_masks(Prediction.model_validate(strong['prediction']), observed)['transient']
        item = {'station': station, 'scenario': truth['scenario'], 'cohort_seed': record['cohort_seed'],
                'positive_transient': bool(target.any()), 'modes': {}}
        for mode in modes:
            receipt = read_json(visual / 'predictions' / mode / station / 'result.json')
            tool = read_json(visual / 'predictions' / mode / station / 'tool-evidence.json')
            row = {'status': receipt['status'], 'visual_intervals': len(tool['receipts']),
                'snapped_intervals': len(tool['proposed']), 'verified_intervals': len(tool['verified']),
                'rejection_reasons': [], 'outside_free_graph': 0, 'outside_all_pelt_edges': 0}
            for event in tool['proposed']:
                edges = {event['start'], event['end'] + 1}
                row['outside_free_graph'] += int(not edges <= free_edges)
                row['outside_all_pelt_edges'] += int(not edges <= all_edges)
            for evidence in tool['receipts']:
                if len(evidence['refined_edges']) != 2:
                    reason = 'no_two_supported_jump_edges'
                elif evidence['refined_edges'][0] >= evidence['refined_edges'][1]:
                    reason = 'unordered_or_same_edge'
                elif 'criteria' not in evidence:
                    reason = 'insufficient_pair_context'
                elif evidence['accepted']:
                    reason = 'verified'
                else:
                    reason = min(evidence['criteria'], key=evidence['criteria'].get) + '_wins'
                row['rejection_reasons'].append(reason)
            for suffix in ('pure', 'snap', 'verified', 'complement', 'raw_complement'):
                result = read_json(visual / 'final' / (mode + '_' + suffix) / station / 'result.json')
                events = result['prediction']['events']
                row[suffix] = counts(events, target, observed)
                if suffix in ('complement', 'raw_complement'):
                    final_mask = event_masks(Prediction.model_validate(result['prediction']), observed)['transient']
                    additions = final_mask & ~strong_mask
                    row[suffix].update(added_tp_days=int((additions & target).sum()),
                                       added_fp_days=int((additions & ~target).sum()))
            item['modes'][mode] = row
            total = totals[mode]
            total['stations'] += 1
            total['positive_stations'] += int(target.any())
            total['failed_requests'] += int(receipt['status'] != 'success')
            total['stations_with_visual_intervals'] += int(bool(tool['receipts']))
            total['positive_stations_without_visual_intervals'] += int(target.any() and not tool['receipts'])
            for key in ('visual_intervals', 'snapped_intervals', 'verified_intervals',
                        'outside_free_graph', 'outside_all_pelt_edges'):
                total[key] += row[key]
            for reason in row['rejection_reasons']:
                total['tool_' + reason] += 1
            for suffix in ('complement', 'raw_complement'):
                for key in ('added_tp_days', 'added_fp_days'):
                    total[suffix + '_' + key] += row[suffix][key]
                total[suffix + '_stations_added_tp'] += int(row[suffix]['added_tp_days'] > 0)
                total[suffix + '_stations_added_fp'] += int(row[suffix]['added_fp_days'] > 0)
        stations.append(item)
    output = Path(f'results/exp29-{split}-full-v1')
    write_json(output / 'visual-role-diagnosis.json', {'totals': totals, 'stations': stations})
    print(totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    diagnose(parser.parse_args().split)
