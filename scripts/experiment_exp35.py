"""Diagnose fixed type routing on saved development predictions, with no API calls."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
from exp20_range_metrics import aggregate_affiliation
from report_exp25 import score_record
from report_per_type import export, per_type_rows

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import counts_metrics
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks, observed_values

RANGE_TYPES = ('transient', 'deformation')
RANGE_SOURCES = {
    'gnv_image': 'jcv_image',
    'gnv_text_final': 'jcv_text_final',
    'gnv_text_pipeline': 'jcv_text_pipeline',
}


def route_prediction(numerical, ranges, length):
    """Keep numerical point events and range-source intervals, including overlaps."""
    numerical = Prediction.model_validate(numerical).validate_length(length)
    ranges = Prediction.model_validate(ranges).validate_length(length)
    events = [event.model_copy(deep=True) for event in numerical.events
        if event.kind in ('spike', 'step')]
    events.extend(event.model_copy(deep=True) for event in ranges.events
        if event.kind in RANGE_TYPES)
    events.sort(key=lambda event: (event.start, event.end, event.kind))
    return Prediction(events=events)


def aggregate_types(rows):
    """Aggregate within each anomaly type; deliberately omit cross-type summaries."""
    typed = {}
    for kind in EVENT_TYPES:
        counts = [row['original']['types'][kind]['daily'] for row in rows]
        typed[kind] = {'daily_micro': counts_metrics(
            *(sum(row[key] for row in counts) for key in ('tp', 'fp', 'fn')))}
    ranges = {}
    for kind in RANGE_TYPES:
        values = [row['range'][kind] for row in rows]
        negatives = [row for row in values if row['negative_station']]
        ranges[kind] = {
            'affiliation': aggregate_affiliation(values, 'affiliation'),
            'negative_stations': len(negatives),
            'negative_station_fp_days': sum(row['false_positive_days'] for row in negatives),
            'negative_station_fp_runs': sum(row['false_positive_runs'] for row in negatives),
        }
    return {'original': {'types': typed}, 'range': ranges}


def retain_type_scores(scores):
    return {'original': {'types': copy.deepcopy(scores['original'].get('types'))},
        'range': {kind: copy.deepcopy(scores['range'][kind]) for kind in RANGE_TYPES}
            if scores.get('range') else None}


def run(source, dataset, output):
    source_summary = source / 'summary.json'
    source_rows = source / 'per-station.json.gz'
    manifest_path = dataset / 'manifest.json'
    source_summary_sha = sha256(source_summary)
    source_rows_sha = sha256(source_rows)
    summary = read_json(source_summary)
    saved = json.loads(gzip.decompress(source_rows.read_bytes()))
    records = {row['station_id']: row for row in read_json(manifest_path)['records']}
    numerical = {row['id']: row for row in saved['vector_nearest']}
    ids = list(numerical)
    if len(ids) != summary['stations']:
        raise ValueError('source station count differs from frozen summary')
    for rows in saved.values():
        if len(rows) != len(ids) or {row['id'] for row in rows} != set(ids):
            raise ValueError('methods do not contain the same stations')
    if any(records[station]['split'] != 'validation' for station in ids):
        raise ValueError('EXP35 diagnosis accepts development stations only')
    range_rows = {method: {row['id']: row for row in saved[parent]}
        for method, parent in RANGE_SOURCES.items()}
    routed = {method: [] for method in RANGE_SOURCES}
    inputs = []
    for station in ids:
        record = records[station]
        observation_path = dataset / record['observation']
        truth_path = dataset / record['truth']
        if sha256(observation_path) != record['observation_sha256']:
            raise ValueError('observation differs from saved manifest')
        if sha256(truth_path) != record['truth_sha256']:
            raise ValueError('truth differs from saved manifest')
        observation = read_json(observation_path)
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        predictions = {method: route_prediction(numerical[station]['events'],
            values[station]['events'], len(observed)) for method, values in range_rows.items()}
        # Truth is loaded only after fixed routing; it cannot choose event sources.
        truth = read_json(truth_path)
        for method, prediction in predictions.items():
            masks = event_masks(prediction, observed)
            generic = np.any(np.column_stack(list(masks.values())), axis=1)
            result = {'status': 'success', 'prediction': prediction.model_dump(),
                'generic_mask': generic.tolist(), 'generic_score': None}
            routed[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'],
                **score_record(observation, truth, result)})
        inputs.append({'id': station, 'observation_sha256': record['observation_sha256'],
            'truth_sha256': record['truth_sha256']})
    methods = {method: retain_type_scores(scores) for method, scores in summary['methods'].items()}
    methods.update({method: aggregate_types(rows) for method, rows in routed.items()})
    report = {'experiment_id': 'GNSS-VLM-EXP35', 'stations': len(ids),
        'scope': 'post-hoc development diagnosis; not independent confirmation',
        'source_experiment_id': summary['experiment_id'], 'methods': methods,
        'new_model_requests': 0}
    strata = {}
    full_rows = {method: rows for method, rows in saved.items()
        if not method.endswith('_pure') and rows[0]['original'].get('types') is not None}
    full_rows.update(routed)
    for key in ('cohort_seed', 'scenario'):
        strata[key] = {method: {str(value): aggregate_types([row for row in rows if row[key] == value])
            for value in sorted({row[key] for row in rows})} for method, rows in full_rows.items()}
    primary = per_type_rows(report)
    comparison = {}
    for method in RANGE_SOURCES:
        candidate = {row['kind']: row for row in primary if row['method'] == method}
        comparison[method] = {}
        for control in ('vector_nearest', *RANGE_SOURCES.values()):
            reference = {row['kind']: row for row in primary if row['method'] == control}
            comparison[method][control] = {kind: {
                'primary_metric': candidate[kind]['primary_metric'],
                'primary_f1_delta': candidate[kind]['primary_f1'] - reference[kind]['primary_f1'],
                **{key + '_delta': candidate[kind][key] - reference[kind][key]
                    for key in ('tp', 'fp', 'fn')}} for kind in EVENT_TYPES}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', report)
    write_json(output / 'strata.json', strata)
    write_json(output / 'comparisons.json', comparison)
    write_json(output / 'provenance.json', {
        'source_directory': str(source.resolve()), 'source_summary_sha256': source_summary_sha,
        'source_per_station_sha256': source_rows_sha, 'manifest_sha256': sha256(manifest_path),
        'script_sha256': sha256(Path(__file__)),
        'protocol_sha256': sha256(Path('docs/experiment-exp35.md')),
        'point_source': 'vector_nearest', 'range_sources': RANGE_SOURCES,
        'input_records': inputs, 'new_model_requests': 0,
        'inputs_unchanged': (sha256(source_summary) == source_summary_sha
            and sha256(source_rows) == source_rows_sha),
        'original_request_failures': summary.get('costs'),
        'source_failures_policy': 'preserve source final outputs; no retries or station exclusions'})
    with (output / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(routed, ensure_ascii=False, allow_nan=False).encode('utf-8'), mtime=0))
    export(output / 'summary.json', output / 'per-type')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.dataset, args.out)
