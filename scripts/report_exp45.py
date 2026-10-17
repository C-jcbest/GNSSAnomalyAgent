"""Evaluate frozen EXP45 requests with the project's actual missing-row representation."""
from __future__ import annotations

import argparse
from pathlib import Path

import experiment_exp45 as experiment
import numpy as np

from gnss_research.files import sha256, write_json
from gnss_research.schema import observed_values


def visual_events(observation, receipt):
    if receipt['status'] != 'success':
        return []
    # A missing day can be JSON null for the entire row, not just three null values.
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    events = []
    for item in receipt['decision']['intervals']:
        if item['status'] != 'confirmed':
            continue
        available = np.flatnonzero(observed[item['start']:item['end'] + 1]) + item['start']
        if not len(available):
            continue
        events.append({'kind': 'deformation', 'start': int(available[0]), 'end': int(available[-1]),
            'status': 'confirmed', 'phases': [], 'evidence': item['observation']})
    return events


def report(phase):
    output = experiment.ROOT / phase / 'review'
    write_json(output / 'evaluation-correction.json', {
        'reason': 'frozen evaluator used ndarray on observations containing whole-row null gaps',
        'correction': 'use existing observed_values; all detection and selection rules unchanged',
        'script_sha256': sha256(Path(__file__)),
        'frozen_script_sha256': sha256(Path(experiment.__file__)),
        'new_requests': 0, 'retry_count': 0, 'scores_seen_before_correction': False})
    partial = experiment.ROOT / phase / 'final'
    if partial.exists():
        partial.rename(partial.with_name('final-incomplete-mask-error-v0'))
    # Replace only the defective conversion, retaining the registered report and all requests.
    experiment.visual_events = visual_events
    experiment.report(phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    report(args.phase)
