"""A separately registered text-screen control for the deanchored diagnostic."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import experiment_exp33 as experiment
from experiment_exp29 import compose, transient_event
from report_exp25 import aggregate_cost, aggregate_rows, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact
from report_exp32 import compare

from gnss_research.files import read_json, sha256, write_json

ROOT = experiment.ROOT / 'extension-s1'
PROTOCOL = Path('docs/experiment-exp33-s1.md')


def configure():
    experiment.ROOT = ROOT
    experiment.PROTOCOL = PROTOCOL
    experiment.MODES = {'dav_text_pipeline': 'jcv_text_pipeline'}


def report():
    registration = experiment.check_registration()
    if not (ROOT / 'complete.json').exists():
        raise ValueError('finish every request first')
    with gzip.open('results/exp33-development-diagnostic-v1/per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    rows['dav_text_pipeline'], rows['dav_text_pipeline_pure'] = [], []
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(experiment.DATASET / record['observation'])
        truth = read_json(experiment.DATASET / record['truth'])
        base = read_json(experiment.PARENT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        work = ROOT / 'predictions/dav_text_pipeline' / station
        receipt = read_json(work / 'result.json')
        additions = [transient_event(item['start'], item['end'], item['observation'])
            for item in receipt.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']
        for suffix, pure in (('', False), ('_pure', True)):
            result = compose(observation, base, additions, pure=pure)
            result['evaluation_dependency_workdirs'] = [str(work)]
            write_json(ROOT / 'final' / ('dav_text_pipeline' + suffix) / station / 'result.json', result)
            rows['dav_text_pipeline' + suffix].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_rows(values) for method, values in rows.items()}
    table = {method: compact(method, summary) for method, summary in summaries.items()}
    paired = compare(rows, 'dav_text_pure', 'dav_text_pipeline_pure')
    main, text = table['dav_text_pure'], table['dav_text_pipeline_pure']
    aff = main['transient_aff_f1'] - text['transient_aff_f1']
    daily = main['transient_daily_f1'] - text['transient_daily_f1']
    passed = ((aff >= .02 and daily >= -.02) or (daily >= .02 and aff >= -.02)) and paired['tp_delta'] > 0
    costs = aggregate_cost(request_receipts([job['work'] for job in registration['jobs']]))
    output = Path('results/exp33-s1-development-diagnostic-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP33-S1', 'stations': 18,
        'methods': summaries, 'new_costs': costs, 'scope': 'Posthoc used-development supplementary control'})
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'admission.json', {'visual_screening_lead': passed, 'aff_delta': aff,
        'daily_delta': daily, 'tp_delta': paired['tp_delta'], 'main_admission_unchanged': True})
    write_table(output, 'comparison.csv', list(table.values()))
    ranges = []
    for method, summary in summaries.items():
        for kind, metrics in (summary['range'] or {}).items():
            if isinstance(metrics, dict) and 'affiliation' in metrics:
                ranges.append({'method': method, 'kind': kind, **metrics['affiliation'],
                    **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', ranges)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'visual_screening_lead': passed, 'main': main, 'text': text, 'costs': costs}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'report'))
    args = parser.parse_args()
    configure()
    if args.stage == 'prepare':
        experiment.prepare()
        write_json(ROOT / 'wrapper-registration.json', {'wrapper': str(Path(__file__)),
            'wrapper_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(PROTOCOL)})
    else:
        frozen = read_json(ROOT / 'wrapper-registration.json')
        if sha256(Path(__file__)) != frozen['wrapper_sha256'] or sha256(PROTOCOL) != frozen['protocol_sha256']:
            raise ValueError('supplementary wrapper changed')
        if args.stage == 'infer':
            experiment.execute()
        else:
            report()
