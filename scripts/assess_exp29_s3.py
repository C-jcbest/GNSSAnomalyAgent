"""Audit independent visual contributions and plot honest positive/negative examples."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp29_data_baseline import DATASET, ROOT
from experiment_exp29 import plot_values
from report_exp29_s3 import MAIN

from gnss_research.files import read_json, write_json


def main():
    report = Path('results/exp29-s3-test-full-v1')
    with gzip.open(report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    output = report / 'cases'
    output.mkdir(exist_ok=False)
    comparisons = {}
    for control in ('vector_nearest', 'vf_vl_text_raw_complement'):
        changes = []
        for a, b in zip(rows[MAIN], rows[control]):
            x, y = a['range']['transient'], b['range']['transient']
            u, v = x['affiliation']['f1'], y['affiliation']['f1']
            aff_delta = u - v if u is not None and v is not None else None
            tp_delta = x['daily']['tp'] - y['daily']['tp']
            fp_delta = x['daily']['fp'] - y['daily']['fp']
            if tp_delta or fp_delta or (aff_delta is not None and abs(aff_delta) > 1e-12):
                changes.append({'station': a['id'], 'cohort_seed': a['cohort_seed'], 'scenario': a['scenario'],
                    'tp_delta': tp_delta, 'fp_delta': fp_delta, 'aff_delta': aff_delta})
        comparisons[control] = {'changes': changes,
            'tp_improved_stations': sum(c['tp_delta'] > 0 for c in changes),
            'fp_increased_stations': sum(c['fp_delta'] > 0 for c in changes),
            'aff_only_improved_stations': sum(c['aff_delta'] is not None and c['aff_delta'] > 0
                                             and c['tp_delta'] <= 0 for c in changes)}
    candidates = comparisons['vector_nearest']['changes']
    selected = {}
    for title, score in [('added_tp', lambda c: c['tp_delta']),
                         ('added_fp', lambda c: c['fp_delta']),
                         ('aff_only', lambda c: c['aff_delta'] if c['aff_delta'] is not None
                                                   and c['tp_delta'] == 0 else 0)]:
        if candidates:
            case = max(candidates, key=score)
            if score(case) > 0:
                selected.setdefault(case['station'], {'selection': [], 'comparison': case})['selection'].append(title)
    for station, case in selected.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        predictions = {'Truth': {'events': truth['events']}}
        for title, mode in [('Visual raw', 'vf_vl_image_pure'),
                            ('Visual complement', MAIN), ('Numerical nearest', 'vector_nearest')]:
            predictions[title] = read_json(ROOT / 'test-visual-v1/final' / mode / station / 'result.json')['prediction']
        values = plot_values(observation)
        first = observation['background_end'] + 1
        time = np.arange(first, len(values))
        figure, panels = plt.subplots(4, 1, figsize=(14, 9), dpi=140, sharex=True)
        for axis, panel in enumerate(panels[:3]):
            panel.plot(time, values[first:, axis], '.-', color='#346c85', linewidth=.8, markersize=2.5)
            for event in truth['events']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'] + 1, color='#d5a537', alpha=.2)
            panel.set_ylabel('NEU'[axis] + ' / station MAD')
            panel.grid(alpha=.2)
        colors = ('#c29525', '#9b6eab', '#387f91', '#727272')
        for index, (_, prediction) in enumerate(predictions.items()):
            spans = [(e['start'], e['end'] - e['start'] + 1)
                     for e in prediction['events'] if e['kind'] == 'transient']
            panels[3].broken_barh(spans, (index - .3, .6), facecolors=colors[index])
        panels[3].set_yticks(range(4), list(predictions))
        panels[3].set_xlim(first, len(values) - 1)
        panels[3].set_ylim(-.7, 3.7)
        panels[3].grid(axis='x', alpha=.2)
        panels[3].set_xlabel('Original daily index; missing samples remain absent')
        figure.suptitle(station + ': ' + ', '.join(case['selection']) + '; gold = transient truth')
        figure.tight_layout()
        figure.savefig(output / (station + '.png'))
        plt.close(figure)
        case['truth_events'] = truth['events']
        case['predictions'] = predictions
    write_json(output / 'cases.json', selected)
    write_json(report / 'contribution-audit.json', comparisons)
    print({k: {name: value for name, value in v.items() if name != 'changes'}
           for k, v in comparisons.items()}, flush=True)


if __name__ == '__main__':
    main()
