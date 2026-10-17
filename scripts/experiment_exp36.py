"""Contrastive training examples, visual proposals and frozen boundary verification."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiment_exp30 as air
import experiment_exp31 as joint
import experiment_exp32 as workflow
import matplotlib.pyplot as plt
import numpy as np
from exp28_data_baseline import DATASET as TRAIN_DATASET
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import (
    compose,
    image_block,
    payload,
    plot_values,
    refine_ranges,
    render_scan,
    transient_event,
)
from experiment_exp30_s1 import complete_windows
from experiment_exp33 import remove_narratives
from experiment_exp35 import route_prediction

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request
from gnss_research.schema import Prediction, event_masks, observed_values

ROOT = Path('artifacts/exp36')
DATASET = workflow.DATASET
PROTOCOL = Path('docs/experiment-exp36.md')
MODES = ('cev_image', 'cev_no_examples', 'cev_text')
PROMPT = """Identify complete TEMPORARY OBSERVATION OFFSETS in TARGET GNSS displacement.
Only TARGET is to be predicted. If TRAINING EXAMPLES are supplied, learn the shape distinction
from them; never copy their indices, event count, amplitudes or durations to the target.
There is no universal amplitude or speed cutoff across stations. Every plot is independently
scaled by its own normal-history noise MAD. Text arrays contain the same plotted raw samples.

Compare three parts before confirming: abrupt onset on affected N/E/U components, a sustained
offset interior, and an opposing abrupt return on those same components. Smooth displacement
accumulation followed by arrest is NOT a temporary observation offset: it lacks two opposite
discontinuous edges. A permanent jump has no matching return. Spikes are isolated points.
Real smooth motion and separate permanent steps can coexist with a temporary offset, so the
two outside absolute displacement levels need NOT match. Use the original curves to decide.
Missing points are unknown, not zeros. The blue 3-day median is just a readability aid.

Review ALL target windows jointly. Overlap repeats samples, never independent evidence.
Use ORIGINAL inclusive indices: start = first shifted sample, end = last shifted sample
BEFORE the return. Numerical jumps are fallible optional clues, not a whitelist or answers.
You may propose dates absent from those clues. Do not infer an unobserved return. If onset,
interior or return cannot be established, output uncertain rather than a complete offset.
Empty is valid and no anomaly count is expected. Output temporary offsets ONLY. Other
anomaly types are handled independently; do not diagnose causes or predict activity phases.
Give concise specific evidence comparing onset and return on named axes, including why a
smooth bend or a permanent step is insufficient when relevant. Avoid generic templates.
"""
DEPENDENCIES = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp35.py'),
    Path('scripts/experiment_exp33.py'), *workflow.DEPENDENCIES]


def records_for(phase):
    records = [record for record in read_json(DATASET / 'manifest.json')['records']
        if record['split'] == 'validation' and int(record['station_id'][1:]) % 120 >= 96]
    if phase == 'full':
        return records
    return [record for record in records
        if (int(record['station_id'][1:]) % 120 < 102) == (phase == 'pilot')]


def guard_phase(phase):
    if phase == 'remaining':
        admission = read_json(Path('results/exp36-pilot-v1/admission.json'))
        if not admission['expansion_admitted']:
            raise ValueError('pilot failed; do not expand or use test')


def register(output, phase, stage, records, jobs):
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP36', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'frozen_files': {str(path): sha256(path) for path in DEPENDENCIES}})


def store_joint(work, mode, station, kind, content, prompt, schema, **metadata):
    request = remove_narratives(payload(air.MODEL, prompt, schema, content))
    write_json(work / 'request.json', request)
    return {'work': str(work), 'mode': mode, 'station': station, 'kind': kind,
        'request_sha256': sha256(work / 'request.json'), **metadata}


def configure():
    workflow.ROOT = air.ROOT = joint.ROOT = ROOT
    workflow.records_for = records_for
    workflow.guard_phase = guard_phase
    workflow.register = air.register = register
    joint.records_for = lambda split, phase: records_for(phase)
    joint.store_request = store_joint
    joint.MODES = {'jcv_image': ('image', 'joint', True)}


def render_examples(examples, target):
    figure, panels = plt.subplots(3, len(examples), figsize=(22, 9), dpi=110, squeeze=False)
    for column, example in enumerate(examples):
        values = np.asarray(example['values'], dtype=float)
        time = np.arange(example['first_index'], example['first_index'] + len(values))
        for axis in range(3):
            panel = panels[axis, column]
            panel.plot(time, values[:, axis], '.-', linewidth=.7, markersize=2.5, color='#555555')
            for event in example['labels']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'], alpha=.18, color='#cf623e')
            panel.set_ylabel('NEU'[axis] + ' / own station MAD')
            panel.grid(alpha=.2)
            if axis == 0:
                panel.set_title(example['station'] + ': ' + example['scenario'])
            if axis == 2:
                panel.set_xlabel('Training original daily index')
    figure.suptitle('TRAINING EXAMPLES ONLY. Shading = labeled temporary offset; other motion can coexist.')
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def prepare_examples():
    records = {record['station_id']: record
        for record in read_json(TRAIN_DATASET / 'manifest.json')['records']}
    examples = []
    for station in ('t0002', 't0003', 't0004', 't0005'):
        record = records[station]
        if record['split'] != 'train':
            raise ValueError('examples must be from original training set')
        observation = read_json(TRAIN_DATASET / record['observation'])
        truth = read_json(TRAIN_DATASET / record['truth'])
        labels = truth['events']
        focus = next((event for event in labels if event['kind'] == 'transient'), labels[0])
        left = max(observation['background_end'] + 1, focus['start'] - 25)
        right = min(len(observation['dates']), focus['end'] + 26)
        # A permanent onset is a point label, but the following platform is needed in its example.
        if focus['kind'] == 'step':
            right = min(len(observation['dates']), focus['start'] + 80)
        examples.append({'station': station, 'scenario': truth['scenario'],
            'first_index': left, 'values': finite_rows(plot_values(observation)[left:right]),
            'labels': labels, 'observation_sha256': sha256(TRAIN_DATASET / record['observation']),
            'truth_sha256': sha256(TRAIN_DATASET / record['truth'])})
    folder = ROOT / 'training-examples'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'examples.json', examples)
    render_examples(examples, folder / 'examples.png')


def prepare_direct(phase):
    guard_phase(phase)
    examples = read_json(ROOT / 'training-examples/examples.json')
    output = ROOT / 'validation-visual-v1'
    jobs = []
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True, exist_ok=False)
        render_scan(observation, values, inputs / 'global.png')
        windows = complete_windows(observation)
        for index, (left, right) in enumerate(windows):
            air.render_window(values, left, right, inputs / f'window-{index}.png')
        numeric = read_json(ROOT / 'validation-baseline-v1/predictions/segment_numeric' / station / 'result.json')
        metadata = {'monitoring_start': observation['background_end'] + 1,
            'last_index': len(values) - 1, 'optional_numerical_jumps': [
                {key: jump[key] for key in ('index', 'jump_vector')} for jump in numeric['jumps']]}
        for mode in MODES:
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            if mode != 'cev_no_examples':
                descriptions = [{key: example[key] for key in
                    ('station', 'scenario', 'first_index', 'labels')} for example in examples]
                content.append({'type': 'text', 'text': 'TRAINING EXAMPLE LABELS: ' + json.dumps(descriptions)})
                if mode == 'cev_text':
                    content.append({'type': 'text', 'text': 'TRAINING EXAMPLE ARRAYS: ' + json.dumps(examples)})
                else:
                    content.append(image_block(ROOT / 'training-examples/examples.png'))
            for index, (left, right) in enumerate(windows):
                content.append({'type': 'text', 'text': f'TARGET window {index}, original indices {left}..{right - 1}'})
                if mode == 'cev_text':
                    content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
                else:
                    content.append(image_block(inputs / f'window-{index}.png'))
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(air.MODEL, PROMPT, air.LocalDecision, content))
            jobs.append({'work': str(work), 'mode': mode, 'station': station, 'kind': 'local',
                'request_sha256': sha256(work / 'request.json'),
                'allowed_contexts': [[metadata['monitoring_start'], metadata['last_index']]]})
    register(output, phase, 'direct', records_for(phase), jobs)
    write_json(output / f'{phase}-examples-link.json', {
        'example_json_sha256': sha256(ROOT / 'training-examples/examples.json'),
        'example_image_sha256': sha256(ROOT / 'training-examples/examples.png')})
    print('Prepared', len(jobs), 'direct requests', flush=True)


def prepare_scan(phase):
    output = ROOT / 'validation-visual-v1'
    jobs = []
    for record in records_for(phase):
        observation = read_json(DATASET / record['observation'])
        station = record['station_id']
        windows = complete_windows(observation)
        metadata = {'monitoring_start': observation['background_end'] + 1,
            'last_index': len(observation['dates']) - 1,
            'window_indices_inclusive': [[left, right - 1] for left, right in windows]}
        content = [{'type': 'text', 'text': json.dumps(metadata)}]
        for index in range(len(windows)):
            content.append(image_block(output / 'inputs' / station / f'window-{index}.png'))
        work = output / 'predictions/scan_image' / station
        write_json(work / 'request.json', payload(air.MODEL, air.SCAN_PROMPT, air.ScanDecision, content))
        jobs.append({'work': str(work), 'mode': 'scan_image', 'station': station, 'kind': 'scan',
            'request_sha256': sha256(work / 'request.json'),
            'allowed_context': [metadata['monitoring_start'], metadata['last_index']]})
        # Frozen joint preparer expects both screening families; unused text family is empty.
        write_json(output / 'predictions/scan_text' / station / 'result.json', {
            'status': 'success', 'request_count': 0, 'decision': {'summary': 'Not executed', 'regions': []}})
    register(output, phase, 'scan', records_for(phase), jobs)


def infer(phase, stage):
    output = ROOT / 'validation-visual-v1'
    registration = read_json(output / f'{phase}-{stage}-registration.json')
    air.check_registration(registration)
    write_json(output / f'{phase}-{stage}-started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed after registration')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(air.MODEL))
        if receipt['status'] == 'success':
            try:
                schema = air.ScanDecision if job['kind'] == 'scan' else air.LocalDecision
                decision = schema.model_validate_json(content).model_dump()
                intervals = decision['regions'] if job['kind'] == 'scan' else decision['intervals']
                contexts = [job['allowed_context']] if job['kind'] == 'scan' else job['allowed_contexts']
                if any(not any(left <= item['start'] <= item['end'] <= right for left, right in contexts)
                       for item in intervals):
                    raise ValueError('prediction exceeds supplied context')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            print(stage, index, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / f'{phase}-{stage}-complete.json', {'requests': len(registration['jobs'])})


def result_from_events(observation, events):
    prediction = Prediction.model_validate({'events': events}).validate_length(len(observation['dates']))
    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    return {'status': 'success', 'prediction': prediction.model_dump(), 'request_count': 0,
        'generic_score': None, 'generic_mask': np.any(np.column_stack(list(masks.values())), axis=1).tolist()}


def fuse(observation, numerical, additions):
    # Preserve numerical point ownership; range additions never remove independent point events.
    ranges = compose(observation, numerical, additions)
    routed = route_prediction(numerical['prediction'], ranges['prediction'], len(observation['dates']))
    return result_from_events(observation, routed.model_dump()['events'])


def materialize(phase):
    output = ROOT / 'validation-visual-v1'
    for stage in ('direct', 'joint'):
        if not (output / f'{phase}-{stage}-complete.json').exists():
            raise ValueError('finish all requests first')
        air.check_registration(read_json(output / f'{phase}-{stage}-registration.json'))
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        base = read_json(ROOT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        numeric = read_json(ROOT / 'validation-baseline-v1/predictions/segment_numeric' / station / 'result.json')
        for mode in MODES:
            work = output / 'predictions' / mode / station
            receipt = read_json(work / 'result.json')
            decision = receipt.get('decision', {'intervals': []})
            intervals = [item for item in decision['intervals'] if item['status'] == 'confirmed']
            proposal = {'intervals': intervals, 'evidence': decision.get('summary', 'Failed request; no additions')}
            raw = [transient_event(item['start'], item['end'], item['observation']) for item in intervals]
            refined, verified, evidence = refine_ranges(observation, numeric, proposal)
            write_json(work / 'numerical-verification.json', evidence)
            for suffix, events in (('raw', raw), ('refined', refined), ('verified', verified)):
                write_json(output / 'final' / f'{mode}_{suffix}' / station / 'result.json', fuse(observation, base, events))
                write_json(output / 'final' / f'{mode}_{suffix}_pure' / station / 'result.json', result_from_events(observation, events))
        days = sorted(jump['index'] for jump in numeric['jumps'])
        proposal = {'intervals': [{'start': left, 'end': right - 1}
            for position, left in enumerate(days) for right in days[position + 1:]],
            'evidence': 'All numerical jump pairs; identical frozen verification, no visual model'}
        _, verified, evidence = refine_ranges(observation, numeric, proposal)
        write_json(output / 'inputs' / station / 'numeric-all-pairs-verification.json', evidence)
        write_json(output / 'final/numeric_all_pairs' / station / 'result.json', fuse(observation, base, verified))
        old_work = output / 'predictions/jcv_image' / station
        additions = []
        if old_work.exists():
            old = read_json(old_work / 'result.json')
            additions = [transient_event(item['start'], item['end'], item['observation'])
                for item in old.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']
        write_json(output / 'final/gnv_image' / station / 'result.json', fuse(observation, base, additions))
    write_json(output / f'{phase}-materialized.json', {'stations': len(records_for(phase))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'examples', 'prepare_direct', 'infer_direct',
        'prepare_scan', 'infer_scan', 'prepare_joint', 'infer_joint', 'materialize'))
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    configure()
    guard_phase(args.phase)
    if args.stage == 'baseline':
        workflow.baseline(args.phase)
    elif args.stage == 'examples':
        prepare_examples()
    elif args.stage == 'prepare_direct':
        prepare_direct(args.phase)
    elif args.stage == 'prepare_scan':
        prepare_scan(args.phase)
    elif args.stage == 'prepare_joint':
        joint.prepare_joint('validation', args.phase)
    elif args.stage.startswith('infer_'):
        infer(args.phase, args.stage.removeprefix('infer_'))
    else:
        materialize(args.phase)


if __name__ == '__main__':
    main()
