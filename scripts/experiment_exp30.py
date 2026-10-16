"""Bounded visual scan, model-selected zoom and fresh interval localization."""
from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import apply, free_graph
from exp15_vector import nearest, vectors
from exp30_data_baseline import DATASET, PROTOCOL, ROOT, records_for
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import PROMPT as VRC_PROMPT
from experiment_exp29 import (
    RangeDecision,
    compose,
    image_block,
    payload,
    plot_values,
    render_scan,
    transient_event,
)
from pydantic import BaseModel, ConfigDict, Field

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request
from gnss_research.schema import Interval

MODEL = 'qwen3-vl-plus'
LOCAL_MODES = {'air_image': ('image', 'local', True),
    'air_global': ('image', 'global', True),
    'air_text_final': ('image', 'text', True),
    'air_text_pipeline': ('text', 'text', True),
    'air_image_no_numeric': ('image', 'local', False)}
DEPENDENCIES = [Path(__file__), Path('scripts/exp30_data_baseline.py'), PROTOCOL,
    Path('scripts/experiment_exp29.py'), Path('scripts/experiment_exp26.py'),
    Path('scripts/exp14_fingerprint.py'), Path('scripts/exp15_vector.py')]
SCAN_PROMPT = """Screen GNSS N/E/U displacement for REGIONS WORTH A CLOSER LOOK, not final labels.
The supplied windows overlap and show the same original daily samples at station-specific
normal-history MAD scales. Missing samples are absent. A median line is only a readability aid.
Describe the actual observed cross-axis shape briefly, then propose at most two regions.
Use ORIGINAL inclusive indices; never renumber a cropped window. No expected anomaly count.
Seek shifts, apparent platforms, possible opposing returns or ambiguous superposed changes.
A region can include uncertainty about which boundary returns: do not discard an abrupt
multi-sample change merely because a coarse view cannot resolve its counterpart. Give enough
surroundings to see the change and possible return. Do not report only a single edge as a
complete temporary event. Empty regions are appropriate for ordinary noise or smooth motion.
GNSS physical motion, permanent jumps, spikes and temporary offsets can overlap. Different
stations have different scales; there is no universal amplitude or rate cutoff. This is an
offline morphology task, not an alarm or a causal diagnosis. No numerical detections supplied.
The next step will inspect your chosen regions, so approximate bounds are acceptable.
"""
LOCAL_PROMPT = """Inspect ONE region selected by a fallible GNSS visual/text screening step.
Reconsider its interpretation from the supplied ORIGINAL N/E/U observations. You may propose
NEW dates anywhere in the allowed context; the initial region is not an interval to approve.
Return an observational summary and temporary-offset intervals with confirmed or uncertain
status. No fixed anomaly count. Empty is valid. Use original inclusive daily indices:
start is the first shifted sample; end is the last shifted sample BEFORE an opposing return.
Before confirming, compare the onset, sustained interior and return on the affected axes.
Do not equate a change edge, a one-point spike, a smooth bend or an isolated permanent step
with a complete temporary interval. If the return is unobserved or ambiguous, use uncertain.
An independent permanent step or genuine smooth motion can coexist inside a temporary offset;
outside absolute levels need not coincide. Do not flatten real motion or assume adjacent edges
must pair. Numerical candidate jumps, when provided, are optional fallible evidence, not a
whitelist, final decision or binding boundary. You may reject them or use other dates.
No universal amplitude/speed threshold. Missing observations remain unknown, not zero.
State observed axis/shape evidence concisely, not a copied template, assumed truth or cause.
The same original samples may be supplied as a full view, local image, or explicit arrays.
Only report temporary offsets inside allowed_context. Other event classes are handled separately.
"""


class Region(Interval):
    observation: str


class ScanDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    summary: str
    regions: list[Region] = Field(max_length=2)


class LocalInterval(Interval):
    status: Literal['confirmed', 'uncertain']
    observation: str


class LocalDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    summary: str
    intervals: list[LocalInterval]


def windows(observation):
    first, length = observation['background_end'] + 1, len(observation['dates'])
    return [(left, min(left + 180, length)) for left in range(first, length, 120)]


def local_context(region, observation):
    return max(observation['background_end'] + 1, region['start'] - 35), min(len(observation['dates']), region['end'] + 36)


def render_window(values, left, right, target, local=False):
    figure, panels = plt.subplots(3, 1, figsize=(14, 9) if not local else (18, 10), dpi=100, sharex=True)
    time = np.arange(left, right)
    for axis, panel in enumerate(panels):
        panel.plot(time, values[left:right, axis], '.-', color='#555555', linewidth=.7, markersize=3)
        smooth = values[left:right, axis].copy()
        for position in range(1, len(smooth) - 1):
            if np.isfinite(smooth[position]):
                smooth[position] = np.nanmedian(values[left + position - 1:left + position + 2, axis])
        panel.plot(time, smooth, color='#267e9b', linewidth=1)
        panel.set_ylabel('NEU'[axis] + ' / station MAD; centered')
        panel.grid(alpha=.22)
    panels[-1].set_xlim(left, right - 1)
    step = 5 if local else 10
    panels[-1].set_xticks(np.arange((left // step + 1) * step, right, step))
    panels[-1].tick_params(axis='x', rotation=45)
    panels[-1].set_xlabel('Original daily index; missing samples remain absent')
    figure.suptitle(f'Original N/E/U observations, indices {left} through {right - 1}')
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def check_registration(registration):
    if registration['source_hashes'] != source_hashes():
        raise ValueError('production sources changed')
    for path, digest in registration['frozen_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('registered dependency changed: ' + path)
    if registration['manifest_sha256'] != sha256(DATASET / 'manifest.json'):
        raise ValueError('dataset manifest changed')


def register(output, phase, stage, records, jobs):
    target = output / f'{phase}-{stage}-registration.json'
    if target.exists():
        raise ValueError('stage already registered')
    write_json(target, {'experiment_id': 'GNSS-VLM-EXP30', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}})


def store_request(work, mode, station, kind, content, prompt, schema, **metadata):
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / 'request.json', payload(MODEL, prompt, schema, content))
    return {'work': str(work), 'mode': mode, 'station': station, 'kind': kind,
        'request_sha256': sha256(work / 'request.json'), **metadata}


def prepare_scan(split, phase):
    output = ROOT / (split + '-visual-v1')
    if not (ROOT / (split + '-baseline-v1') / (phase + '-complete.json')).exists():
        raise ValueError('complete numerical phase first')
    if phase == 'remaining':
        check_registration(read_json(output / 'pilot-scan-registration.json'))
    records, jobs = records_for(split, phase), []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True, exist_ok=False)
        render_scan(observation, values, inputs / 'global.png')
        ranges = windows(observation)
        for index, (left, right) in enumerate(ranges):
            render_window(values, left, right, inputs / f'window-{index}.png')
        for view in ('image', 'text'):
            info = {'monitoring_start': observation['background_end'] + 1,
                'last_index': len(values) - 1, 'window_indices_inclusive': [[a, b - 1] for a, b in ranges]}
            scan_content = [{'type': 'text', 'text': json.dumps(info)}]
            for index, (left, right) in enumerate(ranges):
                if view == 'image':
                    scan_content.append(image_block(inputs / f'window-{index}.png'))
                else:
                    scan_content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
            jobs.append(store_request(output / 'predictions' / ('scan_' + view) / station,
                'scan_' + view, station, 'scan', scan_content, SCAN_PROMPT, ScanDecision,
                allowed_context=[observation['background_end'] + 1, len(values) - 1]))
            old_content = [{'type': 'text', 'text': json.dumps({'monitoring_start': observation['background_end'] + 1,
                'last_index': len(values) - 1, 'historical_reference': [observation['background_end'] - 44, observation['background_end']]})}]
            if view == 'image':
                old_content.append(image_block(inputs / 'global.png'))
            else:
                left = observation['background_end'] - 44
                old_content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                    'original_N_E_U': finite_rows(values[left:])}, allow_nan=False)})
            jobs.append(store_request(output / 'predictions' / ('vrc_' + view) / station,
                'vrc_' + view, station, 'vrc', old_content, VRC_PROMPT, RangeDecision,
                allowed_context=[observation['background_end'] + 1, len(values) - 1]))
    register(output, phase, 'scan', records, jobs)
    print('Prepared', split, phase, 'scan', len(jobs), 'requests', flush=True)


def execute(split, phase, stage):
    output = ROOT / (split + '-visual-v1')
    registration = read_json(output / f'{phase}-{stage}-registration.json')
    check_registration(registration)
    marker = output / f'{phase}-{stage}-started.json'
    if marker.exists():
        raise ValueError('stage already started; no implicit retry')
    write_json(marker, {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(MODEL))
        if receipt['status'] == 'success':
            try:
                schema = {'scan': ScanDecision, 'local': LocalDecision, 'vrc': RangeDecision}[job['kind']]
                decision = schema.model_validate_json(content).model_dump()
                intervals = decision.get('regions', decision.get('intervals', []))
                left, right = job['allowed_context']
                if any(not left <= v['start'] <= v['end'] <= right for v in intervals):
                    raise ValueError('decision outside observed context')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            if index % 12 == 0 or result[-1] != 'success':
                print(index, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / f'{phase}-{stage}-complete.json', {'requests': len(registration['jobs'])})


def prepare_local(split, phase):
    output = ROOT / (split + '-visual-v1')
    if not (output / f'{phase}-scan-complete.json').exists():
        raise ValueError('complete scanning first')
    check_registration(read_json(output / f'{phase}-scan-registration.json'))
    records, jobs = records_for(split, phase), []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        numeric = read_json(ROOT / (split + '-baseline-v1/predictions/segment_numeric') / station / 'result.json')
        for family in ('image', 'text'):
            receipt = read_json(output / 'predictions' / ('scan_' + family) / station / 'result.json')
            regions = receipt.get('decision', {}).get('regions', [])
            for index, region in enumerate(regions):
                left, right = local_context(region, observation)
                target = output / 'inputs' / station / f'local-{family}-{index}.png'
                if family == 'image':
                    render_window(values, left, right, target, local=True)
                jumps = [{key: jump[key] for key in ('index', 'jump_vector')}
                    for jump in numeric['jumps'] if left <= jump['index'] < right]
                for mode, (first_family, view, numerical) in LOCAL_MODES.items():
                    if first_family != family:
                        continue
                    metadata = {'allowed_context': [left, right - 1], 'screened_region': region,
                        'scanning_summary': receipt['decision']['summary'], 'scale': 'station MAD, not physical mm'}
                    if numerical:
                        metadata['optional_numerical_jumps'] = jumps
                    content = [{'type': 'text', 'text': json.dumps(metadata)}]
                    if view in ('local', 'global'):
                        content.append(image_block(target if view == 'local' else output / 'inputs' / station / 'global.png'))
                    else:
                        content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                            'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
                    work = output / 'predictions' / mode / station / f'region-{index}'
                    jobs.append(store_request(work, mode, station, 'local', content, LOCAL_PROMPT,
                        LocalDecision, allowed_context=[left, right - 1], region_index=index))
    register(output, phase, 'local', records, jobs)
    print('Prepared', split, phase, 'local', len(jobs), 'requests', flush=True)


def materialize(split, phase):
    output = ROOT / (split + '-visual-v1')
    registration = read_json(output / f'{phase}-local-registration.json')
    check_registration(registration)
    if not (output / f'{phase}-local-complete.json').exists():
        raise ValueError('complete local decisions first')
    for record in records_for(split, phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        numeric = read_json(ROOT / (split + '-baseline-v1/predictions/segment_numeric') / station / 'result.json')
        graph = free_graph(numeric)
        strong = apply(observation, numeric, graph, nearest(graph, vectors(observation, graph)))
        write_json(output / 'final/vector_nearest' / station / 'result.json', strong)
        for view in ('image', 'text'):
            work = output / 'predictions' / ('vrc_' + view) / station
            receipt = read_json(work / 'result.json')
            additions = [transient_event(v['start'], v['end'], receipt['decision']['evidence'])
                for v in receipt.get('decision', {}).get('intervals', [])]
            result = compose(observation, strong, additions)
            result['evaluation_dependency_workdirs'] = [str(work)]
            write_json(output / 'final' / ('vrc_' + view) / station / 'result.json', result)
        for mode, (family, _, _) in LOCAL_MODES.items():
            scan_work = output / 'predictions' / ('scan_' + family) / station
            scan = read_json(scan_work / 'result.json')
            additions, dependencies = [], [str(scan_work)]
            for index, _ in enumerate(scan.get('decision', {}).get('regions', [])):
                work = output / 'predictions' / mode / station / f'region-{index}'
                receipt = read_json(work / 'result.json')
                dependencies.append(str(work))
                additions.extend(transient_event(v['start'], v['end'], v['observation'])
                    for v in receipt.get('decision', {}).get('intervals', []) if v['status'] == 'confirmed')
            for suffix, pure in (('', False), ('_pure', True)):
                result = compose(observation, strong, additions, pure=pure)
                result['evaluation_dependency_workdirs'] = copy.copy(dependencies)
                write_json(output / 'final' / (mode + suffix) / station / 'result.json', result)
    write_json(output / f'{phase}-materialized.json', {'stations': len(registration['records'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare_scan', 'infer_scan', 'prepare_local', 'infer_local', 'materialize'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    if args.stage.startswith('infer_'):
        execute(args.split, args.phase, args.stage.removeprefix('infer_'))
    else:
        {'prepare_scan': prepare_scan, 'prepare_local': prepare_local, 'materialize': materialize}[args.stage](args.split, args.phase)
