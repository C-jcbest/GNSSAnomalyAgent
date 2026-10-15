"""Visual-first range localization, independent numerical verification and paired controls."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import PairDecision, apply, free_graph
from exp15_vector import nearest, vectors
from exp29_data_baseline import DATASET, ROOT
from experiment_exp25 import settings
from experiment_exp26 import PROMPT as CF_PROMPT
from experiment_exp26 import draw_panels, finite_rows
from pydantic import BaseModel, ConfigDict

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request
from gnss_research.schema import Interval, Prediction, event_masks, observed_values
from gnss_research.segmentation import jump_evidence, regression

PROTOCOL = Path('docs/experiment-exp29.md')
DEPENDENCIES = [Path(__file__), Path('scripts/exp29_data_baseline.py'), PROTOCOL,
    Path('scripts/exp14_fingerprint.py'), Path('scripts/exp15_vector.py'),
    Path('scripts/experiment_exp25.py'), Path('scripts/experiment_exp26.py')]
MODES = {'vf_vl_image': ('qwen3-vl-plus', 'image'),
         'vf_vl_text': ('qwen3-vl-plus', 'text'),
         'vf_flash_image': ('qwen3.8-flash', 'image')}
PROMPT = """Locate TEMPORARY observation-offset intervals in GNSS displacement N/E/U.
Return {"intervals":[{"start":integer,"end":integer}],"evidence":"short observations"}.
Indices are ORIGINAL daily indices. Start is the first shifted sample, end is the LAST
shifted sample BEFORE the return; inclusive. Empty intervals are valid. No expected count.

First inspect the whole displacement trajectory and separate smooth physical movement,
isolated spikes, permanent steps, and temporary offsets. Report ONLY temporary offsets:
an abrupt shift sustained over multiple observed samples, followed by an opposing return
across the same components. A permanent unrelated step may occur BEFORE, INSIDE or AFTER
a temporary interval, and physical motion may continue underneath. In that case the two
outside absolute levels need not agree. Follow the superposed trajectory across all axes.
Do not automatically pair adjacent jumps, the earliest jumps, or two isolated points.
A smooth bend, acceleration, deceleration, or plateau after physical motion stops is NOT
a temporary observation offset. A spike plus a separate permanent step is NOT a platform.

There are no supplied numerical detections or candidate boundaries. Propose dates from
the observed curve itself. A subsequent numerical tool can refine approximate dates;
do not invent boundaries inside missing data or extend a platform into smooth movement.
No universal amplitude/speed threshold. Different stations and axes have different scales.
Image: two overlapping time windows of the SAME original daily samples, three axes.
Gray points/line are originals, blue is a 3-day median readability aid, never extra data.
Axes are divided by normal-history station MAD and centered; shape remains original.
Text: contains exactly the same plotted original samples; the median is computable from
them. Historical reference is shaded. Focus on indices >= monitoring_start. Evidence
must mention observed shape, not assumed truth. Do not report point anomalies or activity.
"""


class RangeDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    intervals: list[Interval]
    evidence: str


def plot_values(observation):
    evidence = numerical_evidence(observation)
    values = evidence.values / evidence.noise
    center = np.nanmedian(values[evidence.background_end - 29:evidence.background_end + 1], axis=0)
    return values - center


def render_scan(observation, values, target):
    first = observation['background_end'] + 1
    middle = (first + len(values)) // 2
    windows = ((first - 45, middle + 25), (middle - 25, len(values)))
    smooth = values.copy()
    for day in range(first - 44, len(values) - 1):
        if np.isfinite(values[day]).all():
            smooth[day] = np.nanmedian(values[day - 1:day + 2], axis=0)
    figure, panels = plt.subplots(3, 2, figsize=(15, 9), dpi=120, sharey='row')
    for axis in range(3):
        for column, (left, right) in enumerate(windows):
            panel = panels[axis, column]
            time = np.arange(left, right)
            panel.plot(time, values[left:right, axis], '.-', color='#aaaaaa', linewidth=.6, markersize=2.3)
            panel.plot(time, smooth[left:right, axis], color='#246e8c', linewidth=1)
            if left < first:
                panel.axvspan(left, first - 1, color='#d9e1e5', alpha=.45)
            panel.set_xlim(left, right - 1)
            panel.set_xticks(np.arange((left // 14 + 1) * 14, right, 14))
            panel.tick_params(axis='x', rotation=45)
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD; centered")
            panel.grid(alpha=.25)
            if axis == 0:
                panel.set_title(f'Original daily observations, indices {left}–{right - 1}')
            if axis == 2:
                panel.set_xlabel('Original index; missing samples remain absent')
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def payload(model, prompt, schema, content):
    return {'model': model, 'temperature': 0, 'enable_thinking': False,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'gnss_decision', 'strict': True, 'schema': schema.model_json_schema()}},
        'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]}


def image_block(path):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'
        + base64.b64encode(path.read_bytes()).decode('ascii')}}


def phase_records(split, phase):
    records = read_json(DATASET / 'manifest.json')['records']
    records = [r for r in records if r['split'] == split]
    pilot = {f'{"v" if split == "validation" else "e"}{group * 120 + i:04d}'
             for group in range(3) for i in range(20)}
    return [r for r in records if (r['station_id'] in pilot) == (phase == 'pilot')]


def verify_frozen(registration):
    if registration['source_hashes'] != source_hashes():
        raise ValueError('production sources changed')
    for path, digest in registration['frozen_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('experiment changed after registration: ' + path)
    if registration['manifest_sha256'] != sha256(DATASET / 'manifest.json'):
        raise ValueError('dataset changed')


def prepare(split, phase):
    output = ROOT / f'{split}-visual-v1'
    baseline = ROOT / f'{split}-baseline-v1'
    if split == 'test':
        gate = read_json(Path('results/exp29-validation-full-v1/admission.json'))
        if not gate['admitted']:
            raise ValueError('development admission failed; test remains sealed')
    if not (baseline / 'execution-complete.json').exists():
        raise ValueError('complete baseline first')
    if phase == 'pilot':
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / 'pilot-completed.json').exists():
            raise ValueError('complete pilot before expansion')
        verify_frozen(read_json(output / 'pilot-registration.json'))
    registration_path = output / f'{phase}-registration.json'
    if registration_path.exists():
        raise ValueError('phase already registered')
    records = phase_records(split, phase)
    modes = MODES if phase == 'pilot' else {k: v for k, v in MODES.items() if 'flash' not in k}
    jobs = []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True, exist_ok=False)
        values = plot_values(observation)
        render_scan(observation, values, inputs / 'scan.png')
        first = observation['background_end'] + 1
        metadata = {'monitoring_start': first, 'last_index': len(values) - 1,
                    'historical_reference': [first - 45, first - 1]}
        for mode, (model, view) in modes.items():
            work = output / 'predictions' / mode / station
            work.mkdir(parents=True)
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            if view == 'image':
                content.append(image_block(inputs / 'scan.png'))
            else:
                content.append({'type': 'text', 'text': json.dumps({'first_index': first - 45,
                    'original_N_E_U': finite_rows(values[first - 45:])}, allow_nan=False)})
            write_json(work / 'request.json', payload(model, PROMPT, RangeDecision, content))
            jobs.append({'mode': mode, 'station': station, 'kind': 'range', 'model': model,
                         'request_sha256': sha256(work / 'request.json')})
        numeric = read_json(baseline / 'predictions/segment_numeric' / station / 'result.json')
        graph = free_graph(numeric)
        work = output / 'predictions/nf_vl' / station
        work.mkdir(parents=True)
        write_json(work / 'candidates.json', {'graph': graph, 'numerical_prediction': numeric})
        if graph['pairs']:
            raw_path = inputs / 'nf-original.png'
            draw_panels(observation, graph, raw_path)
            content = [{'type': 'text', 'text': json.dumps(graph)}, image_block(raw_path)]
            mapping = {v['id']: v['delta'] for v in vectors(observation, graph)}
            for pair in graph['pairs']:
                path = inputs / (pair['id'] + '.png')
                draw_panels(observation, graph, path, pair, np.asarray(mapping[pair['first_id']]))
                content.append(image_block(path))
            write_json(work / 'request.json', payload('qwen3-vl-plus', CF_PROMPT, PairDecision, content))
            jobs.append({'mode': 'nf_vl', 'station': station, 'kind': 'pair',
                         'model': 'qwen3-vl-plus', 'request_sha256': sha256(work / 'request.json')})
        else:
            write_json(work / 'result.json', {'status': 'success', 'request_count': 0,
                'pair_decision': {'pair_ids': [], 'evidence': 'No free numerical pairs'}})
    registration = {'experiment_id': 'GNSS-VLM-EXP29', 'split': split, 'phase': phase,
        'records': records, 'jobs': jobs, 'dataset': str(DATASET), 'baseline': str(baseline),
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}}
    write_json(registration_path, registration)
    frozen = output / 'frozen-source'
    frozen.mkdir(exist_ok=True)
    for path in DEPENDENCIES:
        if not (frozen / path.name).exists():
            shutil.copyfile(path, frozen / path.name)
    print('Prepared', phase, len(records), 'stations', len(jobs), 'requests', flush=True)


def infer(split, phase):
    output = ROOT / f'{split}-visual-v1'
    registration = read_json(output / f'{phase}-registration.json')
    verify_frozen(registration)
    marker = output / f'{phase}-started.json'
    if marker.exists():
        raise ValueError('phase already started; no implicit retry')
    write_json(marker, {'requests': len(registration['jobs']), 'workers': 3, 'retries': 0})

    def execute(job):
        work = output / 'predictions' / job['mode'] / job['station']
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(job['model']))
        if receipt['status'] == 'success':
            try:
                schema = RangeDecision if job['kind'] == 'range' else PairDecision
                decision = schema.model_validate_json(content)
                if job['kind'] == 'range':
                    observation = read_json(DATASET / 'observations' / (job['station'] + '.json'))
                    for interval in decision.intervals:
                        if not observation['background_end'] < interval.start <= interval.end < len(observation['dates']):
                            raise ValueError('invalid monitoring interval')
                    receipt['range_decision'] = decision.model_dump()
                else:
                    receipt['pair_decision'] = decision.model_dump()
                    candidates = read_json(work / 'candidates.json')
                    observation = read_json(DATASET / 'observations' / (job['station'] + '.json'))
                    apply(observation, candidates['numerical_prediction'], candidates['graph'], receipt)
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for number, result in enumerate(executor.map(execute, registration['jobs']), 1):
            print(number, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / f'{phase}-completed.json', {'requests': len(registration['jobs']), 'retries': 0})


def transient_event(start, end, evidence):
    return {'kind': 'transient', 'start': start, 'end': end, 'status': 'confirmed',
            'phases': [], 'evidence': evidence}


def refine_ranges(observation, numeric, decision):
    evidence = numerical_evidence(observation)
    eligible = evidence.observed.copy()
    for event in numeric['prediction']['events']:
        if event['kind'] == 'spike':
            eligible[event['start']:event['end'] + 1] = False
    time = np.flatnonzero(eligible)
    values = (evidence.values / evidence.noise)[eligible]
    proposed, accepted, receipts = [], [], []
    for interval in decision.get('intervals', []):
        edges = []
        for target in (interval['start'], interval['end'] + 1):
            candidates = []
            for day in range(max(evidence.background_end + 1, target - 14),
                             min(len(eligible) - 1, target + 14) + 1):
                if not eligible[day] or not eligible[day - 1]:
                    continue
                fit = jump_evidence(time, values, day, day - 21, day + 21)
                if fit['selected'] in ('step', 'step_hinge'):
                    candidates.append(fit)
            if candidates:
                best = min(candidates, key=lambda f: min(f['criteria'][k] for k in ('step', 'step_hinge')))
                edges.append(best['index'])
        receipt = {'visual_interval': interval, 'refined_edges': edges, 'accepted': False}
        if len(edges) == 2 and edges[0] < edges[1]:
            first, second = edges
            event = transient_event(first, second - 1, decision['evidence'])
            proposed.append(event)
            keep = (time >= first - 30) & (time <= second + 30)
            local_time, local_values = time[keep], values[keep]
            if (local_time < first).sum() >= 5 and (local_time >= second).sum() >= 5:
                jumps = [(local_time >= day).astype(float) for day in edges]
                hinges = [np.maximum(local_time - day, 0) for day in edges]
                nuisance_days = sorted({item['index'] for item in numeric.get('jumps', [])
                    if first - 30 < item['index'] < second + 30 and item['index'] not in edges})
                nuisance = [(local_time >= day).astype(float) for day in nuisance_days]
                context = [*hinges, *nuisance]
                continuous = regression(local_time, local_values, context)
                pulse = regression(local_time, local_values, [*context, jumps[0] - jumps[1]])
                independent = regression(local_time, local_values, [*context, *jumps])
                receipt['nuisance_step_days'] = nuisance_days
                receipt['criteria'] = {'continuous': continuous['criterion'],
                    'pulse': pulse['criterion'], 'independent': independent['criterion']}
                receipt['accepted'] = pulse['criterion'] < min(continuous['criterion'], independent['criterion'])
                if receipt['accepted']:
                    accepted.append(event)
        receipts.append(receipt)
    return proposed, accepted, receipts


def compose(observation, base, additions, replace=False, pure=False):
    events = [] if pure else copy.deepcopy(base['prediction']['events'])
    if replace:
        events = [e for e in events if e['kind'] != 'transient']
    for addition in additions:
        if any(e['kind'] == 'transient' and e['start'] <= addition['end']
               and addition['start'] <= e['end'] for e in events):
            continue
        endpoints = {addition['start'], addition['end'] + 1}
        events = [e for e in events if e['kind'] != 'step' or e['start'] not in endpoints]
        events.append(copy.deepcopy(addition))
    prediction = Prediction.model_validate({'events': events}).validate_length(len(observation['dates']))
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {'status': 'success', 'prediction': prediction.model_dump(), 'request_count': 0,
        'generic_score': None, 'generic_mask': np.any(np.column_stack(list(masks.values())), axis=1).tolist()}


def materialize(split, phase):
    output = ROOT / f'{split}-visual-v1'
    registration = read_json(output / f'{phase}-registration.json')
    verify_frozen(registration)
    if not (output / f'{phase}-completed.json').exists():
        raise ValueError('inference incomplete')
    baseline = Path(registration['baseline'])
    modes = MODES if phase == 'pilot' else {k: v for k, v in MODES.items() if 'flash' not in k}
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        numeric = read_json(baseline / 'predictions/segment_numeric' / station / 'result.json')
        graph = free_graph(numeric)
        strong = apply(observation, numeric, graph, nearest(graph, vectors(observation, graph)))
        write_json(output / 'final/vector_nearest' / station / 'result.json', strong)
        jump_days = sorted(jump['index'] for jump in numeric['jumps'])
        numerical_ranges = {'intervals': [{'start': first, 'end': second - 1}
            for position, first in enumerate(jump_days) for second in jump_days[position + 1:]],
            'evidence': 'All numerical jump pairs; same refinement and verification without a model'}
        _, numerical_verified, numerical_tools = refine_ranges(observation, numeric, numerical_ranges)
        write_json(output / 'inputs' / station / 'numeric-all-pairs-tools.json', numerical_tools)
        write_json(output / 'final/numeric_all_pairs' / station / 'result.json',
                   compose(observation, strong, numerical_verified))
        work = output / 'predictions/nf_vl' / station
        response = read_json(work / 'result.json')
        nf = apply(observation, numeric, graph, response)
        nf['evaluation_dependency_workdirs'] = [str(work)]
        write_json(output / 'final/nf_vl' / station / 'result.json', nf)
        for mode in modes:
            work = output / 'predictions' / mode / station
            response = read_json(work / 'result.json')
            decision = response.get('range_decision', {'intervals': [], 'evidence': 'Failed request; no proposals'})
            raw = [transient_event(iv['start'], iv['end'], decision['evidence']) for iv in decision['intervals']]
            refined, verified, tools = refine_ranges(observation, numeric, decision)
            write_json(work / 'tool-evidence.json', {'proposed': refined, 'verified': verified, 'receipts': tools})
            variants = {'pure': compose(observation, strong, raw, pure=True),
                'snap': compose(observation, strong, refined, replace=True),
                'verified': compose(observation, strong, verified, replace=True),
                'complement': compose(observation, strong, verified),
                'raw_complement': compose(observation, strong, raw)}
            for suffix, result in variants.items():
                result['visual_dependency_status'] = response['status']
                result['evaluation_dependency_workdirs'] = [str(work)]
                if suffix == 'pure' and response['status'] != 'success':
                    result['status'] = 'failed'
                write_json(output / 'final' / (mode + '_' + suffix) / station / 'result.json', result)
    write_json(output / f'{phase}-materialized.json', {'stations': len(registration['records'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'materialize'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'materialize': materialize}[args.stage](args.split, args.phase)
