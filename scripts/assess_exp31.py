"""Select true recovery, worst false alarm and affiliation-only examples without cherry-picking."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp31_data_baseline import DATASET
from experiment_exp29 import plot_values

from gnss_research.files import read_json, write_json


def assess(report, method='jcv_image', control='vector_nearest'):
    with gzip.open(report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    if method == 'jcv_image' and control == 'vector_nearest':
        changes = read_json(report / 'paired-changes.json')[control]['changes']
    else:
        changes = []
        for main, other in zip(rows[method], rows[control], strict=True):
            a, b = main['range']['transient'], other['range']['transient']
            aff_a, aff_b = a['affiliation']['f1'], b['affiliation']['f1']
            changes.append({'station': main['id'], 'scenario': main['scenario'],
                'tp_delta': a['daily']['tp'] - b['daily']['tp'],
                'fp_delta': a['daily']['fp'] - b['daily']['fp'],
                'aff_delta': aff_a - aff_b if aff_a is not None and aff_b is not None else None})
    selected = {}
    for name, score in (
        ('added_tp', lambda change: change['tp_delta']),
        ('lost_tp', lambda change: -change['tp_delta']),
        ('added_fp', lambda change: change['fp_delta']),
        ('aff_only', lambda change: change['aff_delta']
            if change['aff_delta'] is not None and change['tp_delta'] <= 0 else 0),
    ):
        if not changes:
            continue
        case = max(changes, key=score)
        if score(case) > 0:
            selected.setdefault(case['station'], {'selection': [], 'change': case})['selection'].append(name)
    methods = ('jcv_image', 'jcv_image_pure', 'jcv_global_only', 'jcv_local_only',
               'jcv_text_final', 'jcv_text_pipeline', 'air_image', 'air_global', 'vrc_image', 'vector_nearest')
    methods = list(dict.fromkeys([method, control, *methods]))
    default = method == 'jcv_image' and control == 'vector_nearest'
    output = report / ('cases' if default else f'cases-{method}-vs-{control}')
    output.mkdir(exist_ok=False)
    for station, case in selected.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        values = plot_values(observation)
        first = observation['background_end'] + 1
        time = np.arange(first, len(values))
        figure, panels = plt.subplots(4, 1, figsize=(16, 12), dpi=130, sharex=True)
        for axis, panel in enumerate(panels[:3]):
            panel.plot(time, values[first:, axis], '.-', linewidth=.7, markersize=2, color='#356d85')
            for event in truth['events']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'] + 1, color='#cf9c29', alpha=.2)
            panel.set_ylabel('NEU'[axis] + ' / station MAD')
            panel.grid(alpha=.2)
        predictions = {'truth': {'events': truth['events']}}
        for comparison_method in methods:
            row = next(row for row in rows[comparison_method] if row['id'] == station)
            predictions[comparison_method] = row['events']
        for index, prediction in enumerate(predictions.values()):
            spans = [(event['start'], event['end'] - event['start'] + 1)
                     for event in prediction['events'] if event['kind'] == 'transient']
            panels[3].broken_barh(spans, (index - .3, .6),
                facecolors='#cf9c29' if index == 0 else '#397e91')
        panels[3].set_yticks(range(len(predictions)), list(predictions))
        panels[3].set_ylim(-.7, len(predictions) - .3)
        panels[3].set_xlim(first, len(values) - 1)
        panels[3].grid(axis='x', alpha=.2)
        panels[3].set_xlabel('Original index; missing observations remain absent')
        figure.suptitle(station + f' / {method} vs {control}: ' + ', '.join(case['selection']) + '; gold = transient truth')
        figure.tight_layout()
        figure.savefig(output / (station + '.png'))
        plt.close(figure)
        case.update(truth=truth['events'], predictions=predictions)
    write_json(output / 'cases.json', selected)
    print({station: case['selection'] for station, case in selected.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--method', choices=('jcv_image', 'jcv_image_pure'), default='jcv_image')
    parser.add_argument('--control', choices=('vector_nearest', 'jcv_text_final_pure'), default='vector_nearest')
    args = parser.parse_args()
    assess(args.report, args.method, args.control)
