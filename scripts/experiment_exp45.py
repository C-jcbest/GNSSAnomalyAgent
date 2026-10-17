"""Preserve numerical deformation detection while testing visual boundary alternatives."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import exp28_data_baseline as numerical
import experiment_exp38 as plotting
import experiment_exp39 as local
import numpy as np
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values
from experiment_exp30 import LocalDecision
from experiment_exp36 import result_from_events
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request

ROOT = Path('artifacts/exp45')
DATASET = Path('artifacts/exp41/data')
PROTOCOL = Path('docs/experiment-exp45.md')
MODES = ('nab_image', 'nab_text')
MODEL = 'qwen3.8-flash'
DEPENDENCIES = [Path(__file__), PROTOCOL, *[Path('scripts') / filename for filename in (
    'exp28_data_baseline.py', 'experiment_exp38.py', 'experiment_exp39.py',
    'experiment_exp25.py', 'experiment_exp26.py', 'experiment_exp29.py',
    'experiment_exp30.py', 'experiment_exp36.py', 'report_exp25.py', 'report_exp36.py')]]
PROMPT = """Review SUSTAINED DEFORMATION ACTIVITY in original daily GNSS N/E/U displacement.
Return summary and intervals with original inclusive start/end, confirmed/uncertain status,
and a short axis-specific observation. Output deformation activity ONLY, no spikes, permanent
steps, temporary observation offsets, physical causes, future motion, or internal phases.

The full original curves and normal reference establish the morphology. Numerical activity
ranges are FALLIBLE proposals. Local panels show their onset and the day after their end,
each in a +/-21-day neighborhood, not known change points. You may retain, adjust, reject,
or mark a proposal uncertain. Missing samples are unknown, not zero. Gray is original samples;
blue 3-day median is only a reading aid. All dates are original indices. Local panels are
centered separately; their absolute heights are not directly comparable. Text input contains
the same plotted original samples and median lines. Use monitoring indices 365..729.

Confirm continuous, persistent displacement movement on the ORIGINAL trajectory before
setting boundaries. Start is the first observed sample of continued movement departing
from the prior trajectory, end is the last observed active sample before arrest when visible.
A cumulative displacement plateau AFTER arrest is not still active. A single abrupt permanent
step and its shifted plateau are not sustained movement. A temporary jump-and-return may
coexist with physical motion; do not let such offset edges define the whole activity range.
Noise, ordinary seasonal bending and short fluctuations do not establish a new movement.
One clear component may support activity while other components are weak. Do not use an
anticipated derivative response to extend the original displacement activity into a flat
prefix. Do not infer movement boundaries inside gaps. For right-censored activity only the
observable active portion can be described; do not invent arrest after the record.

If both morphology and the observed active range are supported, give confirmed bounds and
concise evidence comparing continued motion with the pre-onset trajectory and post-arrest
plateau. Otherwise use uncertain or empty intervals. No fixed count, universal millimeter
or speed cutoff. A later numerical tool will compare your candidate with the original
numerical range; your proposal is not automatically the final boundary.
"""


def records_for(phase):
    records = read_json(DATASET / 'manifest.json')['records']
    return [record for record in records if record['split'] == 'validation'
        and (record['within_cohort'] < 6) == (phase == 'pilot')]


def baseline_root(phase):
    if phase == 'pilot':
        return Path('artifacts/exp41/pilot/baseline')
    return ROOT / phase / 'baseline'


def guard(phase):
    if phase == 'confirmation' and not read_json(Path('results/exp45-pilot-v1/admission.json'))['admitted']:
        raise ValueError('pilot did not meet the registered preservation target')


def register(output, phase, stage, jobs, routes=None):
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP45',
        'phase': phase, 'stage': stage, 'records': records_for(phase), 'jobs': jobs,
        'routes': routes, 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'source_hashes': source_hashes(), 'frozen_files': {str(path): sha256(path) for path in DEPENDENCIES}})


def verify_registration(registration):
    if registration['source_hashes'] != source_hashes():
        raise ValueError('production sources changed')
    if registration['manifest_sha256'] != sha256(DATASET / 'manifest.json'):
        raise ValueError('dataset manifest changed')
    for name, digest in registration['frozen_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('registered source changed: ' + name)
    for record in registration['records']:
        if sha256(DATASET / record['observation']) != record['observation_sha256']:
            raise ValueError('registered observation changed')


def initialize_worker(model_path):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model_path)


def baseline(phase):
    guard(phase)
    if phase == 'pilot':
        raise ValueError('pilot uses already recorded numerical predictions')
    training = read_json(numerical.TRAINING / 'training.json')
    if training['source_hashes'] != source_hashes() or training['model_sha256'] != sha256(numerical.TRAINING / 'models.joblib'):
        raise ValueError('frozen numerical training changed')
    output = baseline_root(phase)
    output.mkdir(parents=True, exist_ok=False)
    register(output, phase, 'baseline', [])
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    jobs = [(record, output) for record in records_for(phase)]
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
            initargs=(numerical.TRAINING / 'models.joblib',)) as executor:
        for index, station in enumerate(executor.map(numerical.predict_station, jobs), 1):
            if index % 6 == 0:
                print('Numerical baseline', phase, index, '/', len(jobs), station, flush=True)
    write_json(output / 'complete.json', {'stations': len(jobs), 'new_training': False})


def activity_events(prediction):
    return [event for event in prediction['prediction']['events']
        if event['kind'] == 'deformation' and event['status'] == 'confirmed']


def prepare(phase):
    guard(phase)
    if not (baseline_root(phase) / 'complete.json').exists():
        raise ValueError('finish numerical predictions first')
    output = ROOT / phase / 'review'
    output.mkdir(parents=True, exist_ok=False)
    jobs, routes, inputs = [], [], []
    for record in records_for(phase):
        station = record['station_id']
        path = baseline_root(phase) / 'predictions/segment_numeric' / station / 'result.json'
        numeric = read_json(path)
        candidates = [{'id': f'A{index + 1}', 'kind': 'deformation',
            'start': event['start'], 'end': event['end']}
            for index, event in enumerate(activity_events(numeric))]
        routes.append({'station': station, 'visual_requested': bool(candidates),
            'reason': 'numerical_activity_candidate' if candidates else 'no_supported_activity',
            'candidates': candidates})
        inputs.append({'path': str(path), 'sha256': sha256(path)})
        if not candidates:
            continue
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        views = local.local_views(values, candidates)
        folder = output / 'inputs' / station
        folder.mkdir(parents=True)
        overview = folder / 'original.png'
        edges = folder / 'activity-edges.png'
        plotting.render_sheet({'original': values}, overview, 'nrv_raw_only')
        local.render_local(views, edges)
        for mode in MODES:
            content = [{'type': 'text', 'text': json.dumps({'monitoring_indices': [365, 729],
                'normal_reference': [180, 364], 'fallible_numerical_activity': candidates})}]
            if mode == 'nab_image':
                content.extend(image_block(image) for image in (overview, edges))
            else:
                text_views = [{'id': 'normal_reference', 'left': 180, 'values': values[180:365]},
                    {'id': 'monitoring', 'left': 365, 'values': values[365:730]}, *views]
                for view in text_views:
                    content.append({'type': 'text', 'text': json.dumps({'view': view['id'],
                        'first_index': view['left'], 'samples_NEU': finite_rows(view['values']),
                        'median3_NEU': finite_rows(plotting.median_line(view['values']))})})
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(MODEL, PROMPT, LocalDecision, content))
            jobs.append({'work': str(work), 'station': station, 'mode': mode,
                'request_sha256': sha256(work / 'request.json')})
        print('Prepared activity review', phase, station, flush=True)
    register(output, phase, 'review', jobs, routes)
    write_json(output / 'input-hashes.json', inputs)
    print('Planned', len(jobs), 'requests across', len(routes), 'evaluated stations', flush=True)


def verify_inputs(output, registration):
    verify_registration(registration)
    for item in read_json(output / 'input-hashes.json'):
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('registered numerical prediction changed')


def infer(phase):
    output = ROOT / phase / 'review'
    if (output / 'started.json').exists():
        raise ValueError('registered requests already started; no implicit replay')
    registration = read_json(output / 'registration.json')
    verify_inputs(output, registration)
    write_json(output / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(MODEL))
        if receipt['status'] == 'success':
            try:
                decision = LocalDecision.model_validate_json(content).model_dump()
                if any(not 365 <= event['start'] <= event['end'] <= 729 for event in decision['intervals']):
                    raise ValueError('visual interval outside monitoring target')
                receipt['decision'] = decision
            except (ValueError, KeyError, TypeError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['station'], job['mode'], receipt['status']

    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, status in enumerate(executor.map(call, registration['jobs']), 1):
            print('Activity review', phase, index, '/', len(registration['jobs']), *status, flush=True)
    write_json(output / 'complete.json', {'requests': len(registration['jobs'])})


def visual_events(observation, receipt):
    if receipt['status'] != 'success':
        return []
    observed = np.isfinite(np.asarray(observation['displacement_mm'], dtype=float)).all(axis=1)
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


def activity_context(observation, numeric):
    evidence = numerical_evidence(observation)
    eligible = evidence.observed.copy()
    eligible[:max(0, evidence.background_end - 89)] = False
    for event in numeric['prediction']['events']:
        if event['kind'] == 'spike':
            eligible[event['start']:event['end'] + 1] = False
    time = np.flatnonzero(eligible)
    values = evidence.residual[eligible] / evidence.noise
    design = np.column_stack([np.ones(len(time)), time - time.mean(),
        *[(time >= jump['index']).astype(float) for jump in numeric['jumps']]])
    basis = np.linalg.qr(design, mode='reduced')[0]
    residual = values - basis @ (basis.T @ values)
    return time, basis, residual


def candidate_gain(context, start, end):
    time, basis, residual = context
    fraction = np.clip((time - start + 1) / (end - start + 1), 0, 1)
    templates = np.column_stack([fraction, fraction * fraction * (3 - 2 * fraction)])
    projected = templates - basis @ (basis.T @ templates)
    norm = np.square(projected).sum(axis=0)
    covariance = projected.T @ residual
    gain = np.square(covariance).sum(axis=1) / np.maximum(norm, np.finfo(float).eps)
    return float(gain.max())


def select_boundaries(observation, numeric, alternatives):
    events = copy.deepcopy(numeric['prediction']['events'])
    activities = activity_events(numeric)
    if not activities:
        return result_from_events(observation, events), []
    context = activity_context(observation, numeric)
    decisions = []
    for event in events:
        if event['kind'] != 'deformation' or event['status'] != 'confirmed':
            continue
        original = (event['start'], event['end'])
        choices = {original: candidate_gain(context, *original)}
        for candidate in alternatives:
            pair = (candidate['start'], candidate['end'])
            if candidate['end'] - candidate['start'] < 14:
                continue
            if candidate['start'] <= event['end'] and event['start'] <= candidate['end']:
                choices[pair] = candidate_gain(context, *pair)
        selected = max(choices, key=choices.get)
        if selected != original:
            event.update(start=selected[0], end=selected[1], phases=[],
                evidence='Visual activity boundary alternative selected by the unchanged station displacement-fit criterion')
        decisions.append({'numeric': original, 'selected': selected, 'changed': selected != original,
            'numeric_gain': choices[original], 'selected_gain': choices[selected],
            'candidate_gains': [{'start': pair[0], 'end': pair[1], 'gain': gain}
                for pair, gain in choices.items()]})
    return result_from_events(observation, events), decisions


def report(phase):
    output = ROOT / phase / 'review'
    registration = read_json(output / 'registration.json')
    verify_inputs(output, registration)
    if not (output / 'complete.json').exists():
        raise ValueError('finish all registered requests first')
    methods = [*numerical.METHODS, *MODES, 'visual_bounds_image', 'visual_bounds_text']
    rows = {method: [] for method in methods}
    audit, protected = [], True
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        predictions = {method: read_json(baseline_root(phase) / 'predictions' / method / station / 'result.json')
            for method in numerical.METHODS}
        numeric = predictions['segment_numeric']
        routed = bool(activity_events(numeric))
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json') if routed else {'status': 'not_routed'}
            alternatives = visual_events(observation, receipt)
            predictions[mode], choices = select_boundaries(observation, numeric, alternatives)
            direct = 'visual_bounds_image' if mode == 'nab_image' else 'visual_bounds_text'
            others = [copy.deepcopy(e) for e in numeric['prediction']['events'] if e['kind'] != 'deformation']
            predictions[direct] = result_from_events(observation, [*others, *alternatives])
            for kind in ('spike', 'step', 'transient'):
                protected = protected and [e for e in predictions[mode]['prediction']['events']
                    if e['kind'] == kind] == [e for e in numeric['prediction']['events'] if e['kind'] == kind]
            audit.append({'station': station, 'mode': mode, 'routed': routed,
                'visual_status': receipt['status'], 'visual_alternatives': alternatives, 'boundary_choices': choices})
            write_json(ROOT / phase / 'final' / mode / station / 'result.json', predictions[mode])
        truth = read_json(DATASET / record['truth'])
        for method, prediction in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': prediction['prediction'],
                **score_record(observation, truth, prediction)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): plotting.aggregate_types([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    receipts = request_receipts([job['work'] for job in registration['jobs']])
    summary = {'experiment_id': 'GNSS-VLM-EXP45-' + phase.upper(), 'stations': len(registration['records']),
        'methods': summaries, 'costs': aggregate_cost(receipts),
        'routed_stations': sum(route['visual_requested'] for route in registration['routes']),
        'deformation_owner': 'numerical presence, optional visual boundaries selected by unchanged fit',
        'other_types_owner': 'unchanged segment_numeric'}
    target = Path(f'results/exp45-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'strata.json', strata)
    write_json(target / 'boundary-audit.json', audit)
    write_json(target / 'request-receipts.json', receipts)
    base = summaries['segment_numeric']

    def aff(scores):
        return scores['range']['deformation']['affiliation']['station_macro_f1']

    def daily(scores):
        return scores['original']['types']['deformation']['daily_micro']['f1']

    main = summaries['nab_image']
    seed_deltas = {seed: aff(strata['cohort_seed']['nab_image'][seed]) - aff(strata['cohort_seed']['segment_numeric'][seed])
        for seed in strata['cohort_seed']['nab_image']}
    checks = {'aff_close_to_numeric': aff(main) - aff(base) >= -.01,
        'daily_f1_close_to_numeric': daily(main) - daily(base) >= -.02,
        'other_event_types_preserved': protected,
        'each_seed_aff_close': phase == 'pilot' or all(delta >= -.02 for delta in seed_deltas.values())}
    write_json(target / 'admission.json', {'admitted': all(checks.values()), 'checks': checks,
        'aff_delta': aff(main) - aff(base), 'daily_f1_delta': daily(main) - daily(base),
        'seed_aff_deltas': seed_deltas, 'scope': 'descriptive preservation, not a formal noninferiority test'})
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'costs': summary['costs'], 'routed_stations': summary['routed_stations'],
        'checks': checks, 'aff_delta': aff(main) - aff(base), 'daily_delta': daily(main) - daily(base),
        'changed_image_boundaries': sum(choice['changed'] for item in audit if item['mode'] == 'nab_image'
            for choice in item['boundary_choices'])}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare', 'infer', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    globals()[args.stage](args.phase)
