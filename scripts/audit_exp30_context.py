"""Development-only context visibility diagnosis; no oracle predictions or API calls."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from exp30_data_baseline import DATASET, ROOT, records_for
from experiment_exp30 import local_context

from gnss_research.files import read_json, write_json
from gnss_research.schema import observed_values


def connected_contexts(contexts):
    merged = []
    for left, right in sorted(contexts):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def audit():
    output = Path('results/exp30-context-audit-v1')
    output.mkdir(exist_ok=False)
    summaries, details = {}, []
    for family in ('image', 'text'):
        counts = Counter()
        for record in records_for('validation', 'pilot'):
            station = record['station_id']
            observation = read_json(DATASET / record['observation'])
            truth = read_json(DATASET / record['truth'])
            observed = np.isfinite(observed_values(observation)).all(axis=1)
            receipt = read_json(ROOT / 'implementation-s1/validation-visual-v1/predictions'
                / ('scan_' + family) / station / 'result.json')
            regions = receipt.get('decision', {}).get('regions', [])
            contexts = [local_context(region, observation) for region in regions]
            merged = connected_contexts(contexts)
            for event in truth['events']:
                if event['kind'] != 'transient':
                    continue
                before = np.flatnonzero(observed & (np.arange(len(observed)) < event['start']))
                after = np.flatnonzero(observed & (np.arange(len(observed)) > event['end']))
                if not len(before) or not len(after):
                    counts['no_observed_outside_context_in_full_record'] += 1
                    continue
                left_required, right_required = int(before[-1]), int(after[0])
                full = any(left <= left_required and right_required < right for left, right in contexts)
                joined = any(left <= left_required and right_required < right for left, right in merged)
                counts['true_intervals'] += 1
                counts['single_crop_contains_both_observed_sides'] += int(full)
                counts['merged_overlapping_crops_contain_both_sides'] += int(joined)
                counts['only_merged_context_contains_both_sides'] += int(joined and not full)
                counts['neither_contains_both_sides'] += int(not joined)
                details.append({'family': family, 'station': station,
                    'truth_interval': [event['start'], event['end']],
                    'required_observed_outside_indices': [left_required, right_required],
                    'contexts': contexts, 'merged_contexts': merged,
                    'single_crop_complete': full, 'merged_context_complete': joined})
        summaries[family] = dict(counts)
    result = {'scope': 'Posthoc development diagnosis only; merging is a visibility counterfactual, not a tested detector.',
        'new_model_requests': 0, 'summary': summaries, 'details': details}
    write_json(output / 'context-visibility.json', result)
    print(summaries, flush=True)


if __name__ == '__main__':
    audit()
