"""Audit independent image contributions and draw masked positive/negative cases."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiment_exp29 import plot_values
from experiment_exp36 import DATASET

from gnss_research.baselines import contiguous_runs
from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import Prediction, event_masks, observed_values


def audit(source):
    source_path = source / 'per-station.json.gz'
    original_sha = sha256(source_path)
    with gzip.open(source_path, 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    paired = read_json(source / 'paired-changes.json')
    changes = paired['vector_nearest']['changes']
    selected = {}
    for reason, key in (('added_tp', lambda row: row['tp_delta']),
                        ('lost_tp', lambda row: -row['tp_delta']),
                        ('added_fp', lambda row: row['fp_delta']),
                        ('aff_without_added_tp', lambda row: row['aff_delta']
                            if row['aff_delta'] is not None and row['tp_delta'] <= 0 else 0)):
        row = max(changes, key=key)
        if key(row) > 0:
            selected.setdefault(row['station'], {'selection': [], 'changes': row})['selection'].append(reason)
    output = source / 'contribution-audit'
    output.mkdir(parents=True, exist_ok=False)
    methods = ('vector_nearest', 'owv_flash', 'owv_flash_text')
    indexed = {method: {row['id']: row for row in rows[method]} for method in methods}
    for station, selection in selected.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        observed[:observation['background_end'] + 1] = False
        first = observation['background_end'] + 1
        length = len(observed)
        values = plot_values(observation)
        figure, axes = plt.subplots(4, 1, figsize=(17, 12), sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 1, 1.25]}, dpi=110)
        for axis in range(3):
            axes[axis].plot(np.arange(first, length), values[first:, axis], '.-',
                markersize=3, linewidth=.7, color='#5b6770')
            axes[axis].set_ylabel('NEU'[axis] + ' / station MAD')
            axes[axis].grid(alpha=.2)
        predictions = {'truth': {'events': truth['events']},
            **{method: indexed[method][station]['events'] for method in methods}}
        for position, (method, prediction) in enumerate(predictions.items()):
            masks = event_masks(Prediction.model_validate(prediction), observed)
            for left, right in contiguous_runs(masks['transient']):
                axes[3].broken_barh([(left - .45, right - left + .9)], (position - .3, .6),
                    facecolors=('#b94136', '#777777', '#147da3', '#d49a2d')[position])
        axes[3].set_yticks(range(len(predictions)), labels=list(predictions))
        axes[3].set_ylim(-.6, len(predictions) - .4)
        axes[3].set_xlim(first, length - 1)
        axes[3].set_xlabel('Original daily index; bars exclude missing observations')
        axes[3].set_title('TEMPORARY OFFSET observed-day masks; other event types omitted from bars')
        axes[3].grid(axis='x', alpha=.2)
        figure.suptitle(station + ' | ' + ', '.join(selection['selection']))
        figure.tight_layout()
        figure.savefig(output / (station + '.png'))
        plt.close(figure)
        write_json(output / (station + '.json'), {'station': station, 'selection': selection,
            'predictions': predictions, 'observed_monitoring_mask': observed.tolist(),
            'observation_sha256': sha256(DATASET / 'observations' / (station + '.json'))})
    summary = {'source_sha256': original_sha, 'cases': selected,
        'scope': read_json(source / 'summary.json')['scope'],
        'added_tp_stations': [row['station'] for row in changes if row['tp_delta'] > 0],
        'added_tp_days': sum(max(0, row['tp_delta']) for row in changes),
        'lost_tp_stations': [row['station'] for row in changes if row['tp_delta'] < 0],
        'added_fp_stations': [row['station'] for row in changes if row['fp_delta'] > 0],
        'aff_improved_without_added_tp_stations': [row['station'] for row in changes
            if row['aff_delta'] is not None and row['aff_delta'] > 0 and row['tp_delta'] <= 0]}
    write_json(output / 'audit.json', summary)
    if sha256(source_path) != original_sha:
        raise ValueError('source predictions changed during read-only audit')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    audit(parser.parse_args().source)
