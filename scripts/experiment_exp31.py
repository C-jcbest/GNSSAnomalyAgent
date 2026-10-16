"""Joint global and merged-local review, with unchanged AIR and VRC controls."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiment_exp30 as air
from exp31_data_baseline import DATASET, PROTOCOL, ROOT, records_for
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import compose, image_block, plot_values, transient_event
from experiment_exp30 import LocalDecision, local_context, render_window, store_request
from experiment_exp30_s1 import complete_windows

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request

MODES = {
    'jcv_image': ('image', 'joint', True),
    'jcv_local_only': ('image', 'local', True),
    'jcv_global_only': ('image', 'global', True),
    'jcv_text_final': ('image', 'text', True),
    'jcv_text_pipeline': ('text', 'text', True),
    'jcv_image_no_numeric': ('image', 'joint', False),
}
PROMPT = """Review GNSS N/E/U displacement for complete TEMPORARY OBSERVATION OFFSETS.
First consider the broad trajectory, then inspect every supplied focus context. These are
views of the SAME original daily observations, not independent evidence. Global and local
views can look different because of scale; resolve the discrepancy using the actual curve.
Screened regions and numerical jumps are fallible clues, not answers or binding boundaries.
Review all focus contexts together and return a single set of intervals. You may change
the initial dates anywhere inside a supplied allowed_context. Do not join intervals across
unobserved gaps between focus contexts. Use ORIGINAL inclusive indices throughout.

A temporary offset has an abrupt onset, a sustained shifted interior, and an opposing return
on the same affected components. Compare these three parts and the observations on both
sides before confirming. Start is the FIRST shifted sample; end is the LAST shifted sample
BEFORE the opposing return, not the return sample. Do not use just the strongest interior
fluctuation as onset. If the full event or its return cannot be established, use uncertain.
Smooth displacement activity can continue under an offset. A separate permanent step can
occur before, inside or after it, so matching the two outside absolute levels is not required.
Follow the superposed shape on N/E/U. A permanent step, isolated spike, gradual bend or
plateau after smooth motion stops is not itself a temporary observation offset.
No universal amplitude or speed cutoff. Normal-history station MAD sets the plot scale;
missing samples are unknown, not zeros. Median lines are readability aids only.
Give concise observed shape evidence. Do not invent causes, unseen returns or a fixed count.
Empty is valid. Output temporary offsets only; other classes are handled separately.
"""


def configure_air():
    # This adapter reuses frozen request/rendering code without editing the old experiment.
    air.ROOT = ROOT
    air.records_for = records_for
    air.windows = complete_windows
    air.register = register
    air.LOCAL_MODES = {key: value for key, value in air.LOCAL_MODES.items()
                       if key in ('air_image', 'air_global')}
    air.DEPENDENCIES = [*air.DEPENDENCIES, Path('scripts/experiment_exp30_s1.py'),
        Path('scripts/experiment_exp31.py'), Path('scripts/exp31_data_baseline.py'), PROTOCOL]


def register(output, phase, stage, records, jobs):
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP31', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'frozen_files': {str(path): sha256(path) for path in air.DEPENDENCIES}})


def merged_contexts(regions, observation):
    contexts = []
    for left, right in sorted(local_context(region, observation) for region in regions):
        if contexts and left <= contexts[-1][1]:
            contexts[-1][1] = max(contexts[-1][1], right)
        else:
            contexts.append([left, right])
    return contexts


def prepare_joint(split, phase):
    output = ROOT / (split + '-visual-v1')
    if not (output / (phase + '-scan-complete.json')).exists():
        raise ValueError('finish the common screening first')
    air.check_registration(read_json(output / (phase + '-scan-registration.json')))
    jobs = []
    for record in records_for(split, phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        inputs = output / 'inputs' / station
        numeric = read_json(ROOT / (split + '-baseline-v1/predictions/segment_numeric') / station / 'result.json')
        for family in ('image', 'text'):
            scan_work = output / 'predictions' / ('scan_' + family) / station
            scan = read_json(scan_work / 'result.json')
            regions = scan.get('decision', {}).get('regions', [])
            contexts = merged_contexts(regions, observation)
            if not contexts:
                continue
            if family == 'image':
                for index, (left, right) in enumerate(contexts):
                    render_window(values, left, right, inputs / f'merged-{index}.png', local=True)
            jumps = [{key: jump[key] for key in ('index', 'jump_vector')}
                     for jump in numeric['jumps']
                     if any(left <= jump['index'] < right for left, right in contexts)]
            for mode, (first_family, view, numerical) in MODES.items():
                if first_family != family:
                    continue
                metadata = {'allowed_contexts': [[left, right - 1] for left, right in contexts],
                    'screened_regions': regions, 'screening_summary': scan['decision']['summary'],
                    'monitoring_start': observation['background_end'] + 1,
                    'scale': 'normal-history station MAD, centered; no physical cutoff'}
                if numerical:
                    metadata['optional_numerical_jumps'] = jumps
                content = [{'type': 'text', 'text': json.dumps(metadata)}]
                if view in ('joint', 'global'):
                    content.extend([{'type': 'text', 'text': 'GLOBAL REFERENCE: same station, full monitoring trajectory.'},
                        image_block(inputs / 'global.png')])
                if view in ('joint', 'local'):
                    for index, (left, right) in enumerate(contexts):
                        content.extend([{'type': 'text', 'text': f'LOCAL FOCUS {index}: original indices {left} through {right - 1}.'},
                            image_block(inputs / f'merged-{index}.png')])
                if view == 'text':
                    left = observation['background_end'] - 44
                    content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:]),
                        'focus_contexts_use_the_same_rows': True}, allow_nan=False)})
                jobs.append(store_request(output / 'predictions' / mode / station,
                    mode, station, 'local', content, PROMPT, LocalDecision,
                    allowed_contexts=metadata['allowed_contexts'], scan_work=str(scan_work)))
    air.register(output, phase, 'joint', records_for(split, phase), jobs)
    print('Prepared', len(jobs), 'joint-stage requests', flush=True)


def execute_joint(split, phase):
    output = ROOT / (split + '-visual-v1')
    registration = read_json(output / (phase + '-joint-registration.json'))
    air.check_registration(registration)
    write_json(output / (phase + '-joint-started.json'), {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered joint request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(air.MODEL))
        if receipt['status'] == 'success':
            try:
                decision = LocalDecision.model_validate_json(content).model_dump()
                for interval in decision['intervals']:
                    if not any(left <= interval['start'] <= interval['end'] <= right
                               for left, right in job['allowed_contexts']):
                        raise ValueError('interval outside a complete focus context')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            if index % 12 == 0 or result[-1] != 'success':
                print(index, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / (phase + '-joint-complete.json'), {'requests': len(registration['jobs'])})


def materialize(split, phase):
    output = ROOT / (split + '-visual-v1')
    if not (output / (phase + '-joint-complete.json')).exists():
        raise ValueError('finish joint review first')
    air.check_registration(read_json(output / (phase + '-joint-registration.json')))
    air.materialize(split, phase)
    for record in records_for(split, phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        strong = read_json(output / 'final/vector_nearest' / station / 'result.json')
        for mode, (family, _, _) in MODES.items():
            scan_work = output / 'predictions' / ('scan_' + family) / station
            scan = read_json(scan_work / 'result.json')
            additions, dependencies = [], [str(scan_work)]
            if scan.get('decision', {}).get('regions'):
                work = output / 'predictions' / mode / station
                receipt = read_json(work / 'result.json')
                dependencies.append(str(work))
                additions = [transient_event(interval['start'], interval['end'], interval['observation'])
                    for interval in receipt.get('decision', {}).get('intervals', [])
                    if interval['status'] == 'confirmed']
            for suffix, pure in (('', False), ('_pure', True)):
                result = compose(observation, strong, additions, pure=pure)
                result['evaluation_dependency_workdirs'] = dependencies
                write_json(output / 'final' / (mode + suffix) / station / 'result.json', result)
    write_json(output / (phase + '-joint-materialized.json'), {'stations': len(records_for(split, phase))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare_scan', 'infer_scan', 'prepare_local', 'infer_local',
        'prepare_joint', 'infer_joint', 'materialize'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    configure_air()
    if args.stage == 'prepare_scan':
        air.prepare_scan(args.split, args.phase)
    elif args.stage == 'infer_scan':
        air.execute(args.split, args.phase, 'scan')
    elif args.stage == 'prepare_local':
        air.prepare_local(args.split, args.phase)
    elif args.stage == 'infer_local':
        air.execute(args.split, args.phase, 'local')
    elif args.stage == 'prepare_joint':
        prepare_joint(args.split, args.phase)
    elif args.stage == 'infer_joint':
        execute_joint(args.split, args.phase)
    else:
        materialize(args.split, args.phase)
