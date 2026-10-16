"""Paired development diagnostic: remove screening narratives and nothing else."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import report_exp32 as reporting
from experiment_exp25 import settings
from experiment_exp29 import compose, transient_event
from experiment_exp30 import LocalDecision
from experiment_exp32 import DATASET
from experiment_exp32 import ROOT as PARENT
from report_exp25 import aggregate_cost, aggregate_rows, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request

ROOT = PARENT.parent / 'exp33'
PROTOCOL = Path('docs/experiment-exp33.md')
PARENT_REPORT = Path('results/exp32-validation-pilot-v1')
MODES = {'dav_image': 'jcv_image', 'dav_text': 'jcv_text_final'}


def remove_narratives(request):
    result = copy.deepcopy(request)
    block = result['messages'][1]['content'][0]
    metadata = json.loads(block['text'])
    del metadata['screening_summary']
    for region in metadata['screened_regions']:
        del region['observation']
    block['text'] = json.dumps(metadata)
    return result


def prepare():
    parent = read_json(PARENT / 'validation-visual-v1/pilot-joint-registration.json')
    records = [record for record in parent['records'] if int(record['station_id'][1:]) % 120 < 42]
    stations = {record['station_id'] for record in records}
    jobs = []
    for original in parent['jobs']:
        if original['station'] not in stations or original['mode'] not in MODES.values():
            continue
        old_work = Path(original['work'])
        if sha256(old_work / 'request.json') != original['request_sha256']:
            raise ValueError('parent request changed')
        mode = next(mode for mode, parent_mode in MODES.items() if parent_mode == original['mode'])
        work = ROOT / 'predictions' / mode / original['station']
        request = remove_narratives(read_json(old_work / 'request.json'))
        write_json(work / 'request.json', request)
        jobs.append({**original, 'work': str(work), 'mode': mode,
            'request_sha256': sha256(work / 'request.json'), 'parent_work': str(old_work),
            'parent_request_sha256': original['request_sha256'],
            'parent_result_sha256': sha256(old_work / 'result.json')})
    write_json(ROOT / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP33', 'records': records,
        'jobs': jobs, 'source_hashes': source_hashes(),
        'frozen_files': {str(path): sha256(path) for path in (Path(__file__), PROTOCOL,
            Path('scripts/experiment_exp29.py'), Path('scripts/experiment_exp30.py'),
            Path('scripts/experiment_exp32.py'))},
        'parent_report_sha256': sha256(PARENT_REPORT / 'per-station.json.gz'),
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'scope': 'Posthoc used-development diagnostic'})
    print('Prepared', len(records), 'stations', len(jobs), 'new requests', flush=True)


def check_registration():
    registration = read_json(ROOT / 'registration.json')
    if registration['source_hashes'] != source_hashes():
        raise ValueError('production source changed')
    if any(sha256(Path(path)) != digest for path, digest in registration['frozen_files'].items()):
        raise ValueError('registered experiment source changed')
    if sha256(DATASET / 'manifest.json') != registration['manifest_sha256']:
        raise ValueError('dataset changed')
    if sha256(PARENT_REPORT / 'per-station.json.gz') != registration['parent_report_sha256']:
        raise ValueError('parent comparison rows changed')
    for job in registration['jobs']:
        parent = Path(job['parent_work'])
        if sha256(parent / 'request.json') != job['parent_request_sha256'] or sha256(parent / 'result.json') != job['parent_result_sha256']:
            raise ValueError('parent request or result changed')
    return registration


def execute():
    registration = check_registration()
    write_json(ROOT / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings('qwen3-vl-plus'))
        if receipt['status'] == 'success':
            try:
                decision = LocalDecision.model_validate_json(content).model_dump()
                for item in decision['intervals']:
                    if not any(left <= item['start'] <= item['end'] <= right for left, right in job['allowed_contexts']):
                        raise ValueError('interval outside supplied context')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            if index % 6 == 0 or result[-1] != 'success':
                print(index, '/', len(registration['jobs']), *result, flush=True)
    write_json(ROOT / 'complete.json', {'requests': len(registration['jobs'])})


def report():
    if not (ROOT / 'complete.json').exists():
        raise ValueError('complete all requests first')
    registration = check_registration()
    stations = {record['station_id'] for record in registration['records']}
    with gzip.open(PARENT_REPORT / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        parent_rows = json.load(stream)
    rows = {method: [row for row in values if row['id'] in stations] for method, values in parent_rows.items()}
    for mode in MODES:
        rows[mode], rows[mode + '_pure'] = [], []
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        base = read_json(PARENT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        for mode in MODES:
            work = ROOT / 'predictions' / mode / station
            receipt = read_json(work / 'result.json')
            additions = [transient_event(item['start'], item['end'], item['observation'])
                for item in receipt.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']
            for suffix, pure in (('', False), ('_pure', True)):
                result = compose(observation, base, additions, pure=pure)
                result['evaluation_dependency_workdirs'] = [str(work)]
                write_json(ROOT / 'final' / (mode + suffix) / station / 'result.json', result)
                rows[mode + suffix].append({'id': station, 'cohort_seed': record['cohort_seed'],
                    'scenario': truth['scenario'], 'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_rows(values) for method, values in rows.items()}
    table = {method: compact(method, summary) for method, summary in summaries.items()}
    paired = {'vs_original_image': reporting.compare(rows, 'dav_image_pure', 'jcv_image_pure'),
        'vs_deanchored_text': reporting.compare(rows, 'dav_image_pure', 'dav_text_pure'),
        'text_vs_original': reporting.compare(rows, 'dav_text_pure', 'jcv_text_final_pure'),
        'fusion_vs_nearest': reporting.compare(rows, 'dav_image', 'vector_nearest')}
    main, old, text = (table[method] for method in ('dav_image_pure', 'jcv_image_pure', 'dav_text_pure'))
    checks = {'daily_gain_at_least_003': main['transient_daily_f1'] - old['transient_daily_f1'] >= .03,
        'tp_gain': paired['vs_original_image']['tp_delta'] > 0,
        'fp_not_increased': paired['vs_original_image']['fp_delta'] <= 0}
    aff_delta = main['transient_aff_f1'] - text['transient_aff_f1']
    daily_delta = main['transient_daily_f1'] - text['transient_daily_f1']
    visual = (aff_delta >= .02 and daily_delta >= -.02) or (daily_delta >= .02 and aff_delta >= -.02)
    admission = {'diagnostic_expansion_admitted': all(checks.values()), 'checks': checks,
        'visual_modality_readout_passed': visual, 'scope': 'Posthoc used-development diagnostic only; no independent confirmation or test.'}
    costs = aggregate_cost(request_receipts([job['work'] for job in registration['jobs']]))
    output = Path('results/exp33-development-diagnostic-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP33', 'stations': len(stations),
        'methods': summaries, 'costs': costs, 'scope': admission['scope'], 'reused_control_requests': 36})
    write_json(output / 'admission.json', admission)
    write_json(output / 'paired-changes.json', paired)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    write_table(output, 'comparison.csv', list(table.values()))
    ranges = []
    for method, summary in summaries.items():
        for kind, metrics in (summary['range'] or {}).items():
            if isinstance(metrics, dict) and 'affiliation' in metrics:
                ranges.append({'method': method, 'kind': kind, **metrics['affiliation'],
                    **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', ranges)
    reporting.PRIMARY = 'dav_image_pure'
    reporting.CONTROLS = ('jcv_image_pure', 'dav_text_pure', 'jcv_text_final_pure')
    reporting.draw_cases(output, rows, paired['vs_original_image']['changes'])
    print(json.dumps({'admission': admission, 'costs': costs, 'main': main, 'old_image': old, 'matched_text': text}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'report'))
    args = parser.parse_args()
    {'prepare': prepare, 'infer': execute, 'report': report}[args.stage]()
