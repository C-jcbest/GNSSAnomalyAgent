"""Plot the largest true recovery, added false detection and distance-only gain."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp30_data_baseline import DATASET
from experiment_exp30 import plot_values

from gnss_research.files import read_json, write_json


def assess(report, method='air_image'):
    if method == 'air_image':
        changes = read_json(report / 'paired-changes.json')['vector_nearest']['changes']
    else:
        audit = read_json(report / 'contribution-audit.json')
        changes = [item for item in audit['details'] if item['method'] == method]
    selected = {}
    selections = [('added_tp', lambda c: c['tp_delta']), ('added_fp', lambda c: c['fp_delta']),
        ('aff_only', lambda c: c['aff_delta'] if c['aff_delta'] is not None and c['tp_delta'] == 0 else 0)]
    for name, score in selections:
        if changes:
            case = max(changes, key=score)
            if score(case) > 0:
                selected.setdefault(case['station'], {'selection': [], 'change': case})['selection'].append(name)
    with gzip.open(report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    output = report / ('cases' if method == 'air_image' else 'cases-' + method)
    output.mkdir(exist_ok=False)
    methods = ('air_image', 'air_image_pure', 'air_text_final', 'air_text_pipeline', 'air_global', 'vrc_image', 'vector_nearest')
    for station, case in selected.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        values = plot_values(observation)
        first = observation['background_end'] + 1
        time = np.arange(first, len(values))
        figure, panels = plt.subplots(4, 1, figsize=(15, 11), dpi=130, sharex=True)
        for axis, panel in enumerate(panels[:3]):
            panel.plot(time, values[first:, axis], '.-', linewidth=.7, markersize=2, color='#356d85')
            for event in truth['events']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'] + 1, color='#cf9c29', alpha=.2)
            panel.set_ylabel('NEU'[axis] + ' / station MAD')
            panel.grid(alpha=.2)
        predictions = {'truth': {'events': truth['events']}}
        for comparison_method in methods:
            row = next(r for r in rows[comparison_method] if r['id'] == station)
            predictions[comparison_method] = row['events']
        for index, (_, prediction) in enumerate(predictions.items()):
            spans = [(e['start'], e['end'] - e['start'] + 1)
                for e in prediction['events'] if e['kind'] == 'transient']
            panels[3].broken_barh(spans, (index - .3, .6), facecolors='#cf9c29' if index == 0 else '#397e91')
        panels[3].set_yticks(range(len(predictions)), list(predictions))
        panels[3].set_ylim(-.7, len(predictions) - .3)
        panels[3].set_xlim(first, len(values) - 1)
        panels[3].grid(axis='x', alpha=.2)
        panels[3].set_xlabel('Original index; missing observations remain absent')
        figure.suptitle(station + ' / ' + method + ': ' + ', '.join(case['selection']) + '; gold = transient truth')
        figure.tight_layout()
        figure.savefig(output / (station + '.png'))
        plt.close(figure)
        case['truth'] = truth['events']
        case['predictions'] = predictions
    write_json(output / 'cases.json', selected)
    print({station: case['selection'] for station, case in selected.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--method', choices=('air_image', 'air_global'), default='air_image')
    args = parser.parse_args()
    assess(args.report, args.method)
