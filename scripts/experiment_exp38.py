"""Normal-reference raw/residual vision, with separately scored range categories."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import exp28_data_baseline as numerical
import experiment_exp37 as old_visual
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import apply, free_graph
from exp15_vector import nearest, vectors
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload
from experiment_exp30 import LocalDecision
from experiment_exp35 import aggregate_types
from experiment_exp36 import result_from_events
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, ConfigDict
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method

from gnss_research.data import SCENARIOS, simulate
from gnss_research.experiment import verify_dataset
from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request
from gnss_research.schema import EVENT_TYPES, Interval

ROOT = Path('artifacts/exp38')
DATASET = ROOT / 'data'
PROTOCOL = Path('docs/experiment-exp38.md')
SEEDS = (2026101738, 2026101739, 2026101740)
MODEL = 'qwen3.8-flash'
MODES = ('nrv_image', 'nrv_text', 'nrv_no_reference', 'nrv_raw_only', 'owv_image')
RANGES = ('transient', 'deformation')
DEPENDENCIES = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp37.py'),
    Path('scripts/experiment_exp36.py'), Path('scripts/exp28_data_baseline.py'),
    Path('scripts/exp14_fingerprint.py'), Path('scripts/exp15_vector.py'),
    Path('scripts/experiment_exp29.py'), Path('scripts/experiment_exp35.py'),
    Path('scripts/report_exp25.py'), Path('scripts/report_exp36.py')]
PROMPT = """Find two kinds of RANGE events in daily GNSS N/E/U ORIGINAL displacement.
One station, rows N/E/U. Gray points/line are actual samples; blue 3-day medians are only
readability aids. The normal reference and monitoring period have ORIGINAL inclusive indices.
Panels for the same component and representation share their vertical scale. Original values
are centered and scaled by this station's historical MAD; residual views additionally remove
a linear trend and annual harmonic fitted ONLY on the normal historical prefix. No universal
amplitude or speed cutoff. A residual excursion alone can be an extrapolation error.
Some ablations omit reference or residual views; do not invent the absent view.
Text, when supplied, gives the identical displayed samples and medians. No numerical candidate,
anomaly count, expected type, or target label is supplied. Empty output is valid.

First compare ordinary historical fluctuations with the target, then inspect ORIGINAL curves
for persistent morphological changes. Do not force a range event onto every station.
TRANSIENT: abrupt displacement offset, persistent shifted segment, then opposing return in
the SAME component(s). One clear component may suffice. A permanent step and smooth motion
may coexist, so absolute before/after levels need not coincide; use local boundary directions.
Ordinary noise or a seasonal bend is not an offset. A step with a permanently shifted plateau
and no observed opposing return is NOT a transient; one outlier is NOT a shifted interval.
DEFORMATION: sustained continuous displacement activity away from the prior local trajectory,
established on ORIGINAL displacement, with onset of continued movement and eventual arrest
when observable. Confirm the activity interval BEFORE any rate/phase interpretation. A single
jump followed by a plateau is NOT continued activity; the plateau after arrest is NOT active.
Residual views help distinguish background drift/seasonality but cannot replace raw evidence.
Both range types can coexist; inspect all components, including small or noisy ones.

Use the FIRST actually observed changed/active sample and LAST actually observed shifted/
active sample, before observed return/arrest. Missing samples are unknown, never zero. Do not
infer onset inside gaps or invent a return beyond the final observation. If either boundary
or the event morphology is ambiguous, use uncertain; only confirmed events are scored as
detections. Do not output spikes, permanent step events, physical causes, or internal phases.
Return original inclusive indices plus a concise axis-specific observation supporting the
activity or BOTH offset boundaries. Multiple or zero intervals are allowed.
"""


class RangeInterval(Interval):
    kind: Literal['transient', 'deformation']
    status: Literal['confirmed', 'uncertain']
    observation: str


class RangeDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    summary: str
    intervals: list[RangeInterval]


def records_for(phase):
    records = read_json(DATASET / 'manifest.json')['records']
    if phase == 'test':
        return [r for r in records if r['split'] == 'test']
    development = [r for r in records if r['split'] == 'validation']
    return [r for r in development if (r['within_cohort'] < 6) == (phase == 'pilot')]


def generate():
    DATASET.mkdir(parents=True, exist_ok=False)
    previous = [Path('data/exp28-expanded-1528-1530'),
        Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp29/data'),
        Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp30/data')]
    used = {read_json(dataset / r['truth'])['seed'] for dataset in previous
        for r in read_json(dataset / 'manifest.json')['records']}
    records = []
    for group, seed in enumerate(SEEDS):
        streams = np.random.SeedSequence(seed).spawn(264)
        for split, count, offset in (('validation', 24, 0), ('test', 240, 24)):
            for index in range(count):
                case_seed = int(streams[offset + index].generate_state(1)[0])
                if case_seed in used:
                    raise ValueError('mechanism seed overlaps previous or current data')
                used.add(case_seed)
                prefix = 'v' if split == 'validation' else 'e'
                station = f'{prefix}{group * count + index:04d}'
                observation, truth = simulate(case_seed, station, SCENARIOS[index % 6], 730, 365)
                op = Path('observations') / (station + '.json')
                tp = Path('truth') / (station + '.json')
                write_json(DATASET / op, observation)
                write_json(DATASET / tp, truth)
                records.append({'station_id': station, 'split': split, 'cohort_seed': seed,
                    'within_cohort': index, 'observation': str(op), 'truth': str(tp),
                    'observation_sha256': sha256(DATASET / op), 'truth_sha256': sha256(DATASET / tp)})
    write_json(DATASET / 'manifest.json', {'schema': 'gnss-research-dataset-v1',
        'seed': SEEDS[0], 'cohort_seeds': list(SEEDS), 'days': 730, 'background_days': 365,
        'split_counts': {'validation': 72, 'test': 720}, 'records': records,
        'origin': 'unchanged mechanism simulation, not field measurements',
        'sources': source_hashes(), 'script_sha256': sha256(Path(__file__)),
        'seed_exclusion_manifests': {str(p): sha256(p / 'manifest.json') for p in previous}})
    verify_dataset(DATASET)
    print('Generated 72 development and 720 independent test stations', flush=True)


def guard(phase):
    if phase != 'pilot':
        previous = 'pilot' if phase == 'confirmation' else 'confirmation'
        if not read_json(Path(f'results/exp38-{previous}-v1/admission.json'))['admitted']:
            raise ValueError(f'{previous} did not meet frozen gate; do not expand')


def register(output, phase, stage, jobs):
    write_json(output / f'{stage}-registration.json', {'experiment_id': 'GNSS-VLM-EXP38',
        'phase': phase, 'records': records_for(phase), 'jobs': jobs,
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}})


def verify_registration(registration):
    if registration['source_hashes'] != source_hashes():
        raise ValueError('production source changed after registration')
    if registration['manifest_sha256'] != sha256(DATASET / 'manifest.json'):
        raise ValueError('registered dataset changed')
    for path, digest in registration['frozen_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('registered dependency changed: ' + path)
    for record in registration['records']:
        if sha256(DATASET / record['observation']) != record['observation_sha256']:
            raise ValueError('registered observation changed')


def initialize_worker(model):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model)


def predict_station(job):
    station = numerical.predict_station(job)
    record, output = job
    observation = read_json(DATASET / record['observation'])
    numeric = read_json(output / 'predictions/segment_numeric' / station / 'result.json')
    graph = free_graph(numeric)
    result = apply(observation, numeric, graph, nearest(graph, vectors(observation, graph)))
    write_json(output / 'predictions/vector_nearest' / station / 'result.json', result)
    return station


def baseline(phase):
    guard(phase)
    training = read_json(numerical.TRAINING / 'training.json')
    if training['source_hashes'] != source_hashes() or training['model_sha256'] != sha256(numerical.TRAINING / 'models.joblib'):
        raise ValueError('frozen model or production source changed')
    verify_dataset(DATASET)
    output = ROOT / phase / 'baseline'
    output.mkdir(parents=True, exist_ok=False)
    register(output, phase, 'baseline', [])
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    records = records_for(phase)
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
            initargs=(numerical.TRAINING / 'models.joblib',)) as executor:
        for index, station in enumerate(executor.map(predict_station, [(r, output) for r in records]), 1):
            if index % 6 == 0:
                print('Baseline', phase, index, '/', len(records), station, flush=True)
    write_json(output / 'complete.json', {'stations': len(records), 'new_fit': False})


def representation(observation):
    evidence = numerical_evidence(observation)
    raw = evidence.values / evidence.noise
    raw -= np.nanmedian(raw[335:365], axis=0)
    residual = evidence.residual / evidence.noise
    return {'original': raw, 'historical_residual': residual}


def median_line(values):
    smooth = values.copy()
    for index in range(1, len(values) - 1):
        if np.isfinite(values[index]).all():
            smooth[index] = np.nanmedian(values[index - 1:index + 2], axis=0)
    return smooth


def render_sheet(arrays, target, mode):
    names = ['original'] if mode == 'nrv_raw_only' else list(arrays)
    figure, panels = plt.subplots(3 * len(names), 2, figsize=(24, 5.7 * len(names)),
        dpi=120, squeeze=False, sharey='row', gridspec_kw={'width_ratios': [1, 2]})
    for block, name in enumerate(names):
        for axis in range(3):
            for column, (left, right) in enumerate(((180, 365), (365, 730))):
                panel = panels[block * 3 + axis, column]
                if column == 0 and mode == 'nrv_no_reference':
                    panel.set_visible(False)
                    continue
                values = arrays[name][left:right]
                smooth = median_line(values)
                time = np.arange(left, right)
                panel.plot(time, values[:, axis], '.-', color='#888888', linewidth=.6, markersize=2)
                panel.plot(time, smooth[:, axis], color='#246e8c', linewidth=1)
                panel.set_xlim(left, right - 1)
                panel.xaxis.set_major_locator(MultipleLocator(20))
                panel.xaxis.set_minor_locator(MultipleLocator(10))
                panel.grid(alpha=.25)
                panel.grid(which='minor', alpha=.12)
                panel.tick_params(axis='x', rotation=60, labelsize=9)
                role = 'NORMAL REFERENCE' if column == 0 else 'MONITORING TARGET'
                panel.set_title(f"{'NEU'[axis]} | {name} | {role} | {left} to {right - 1}", fontsize=10)
                panel.set_ylabel('Station historical MAD units')
    figure.suptitle('ONE STATION | Original displacement first; residual is supporting context only')
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def prepare(phase):
    guard(phase)
    if not (ROOT / phase / 'baseline/complete.json').exists():
        raise ValueError('complete numerical comparisons first')
    output = ROOT / phase / 'visual'
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for number, record in enumerate(records_for(phase), 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        arrays = representation(observation)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True)
        for mode in MODES:
            metadata = {'monitoring_start': 365, 'last_index': 729,
                'normal_visible_indices': None if mode in ('owv_image', 'nrv_no_reference') else [180, 364],
                'index_convention': 'original inclusive daily indices; unknown gaps not labeled'}
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            prompt, schema = PROMPT, RangeDecision
            if mode == 'nrv_text':
                for left, right in ((180, 365), (365, 730)):
                    for name, values in arrays.items():
                        content.append({'type': 'text', 'text': json.dumps({'view': name,
                            'first_index': left, 'samples_N_E_U': finite_rows(values[left:right]),
                            'median3_N_E_U': finite_rows(median_line(values[left:right]))}, allow_nan=False)})
            else:
                image = inputs / (mode + '.png')
                if mode == 'owv_image':
                    old_visual.render_sheet(observation, image)
                    prompt, schema = old_visual.PROMPT, LocalDecision
                else:
                    render_sheet(arrays, image, mode)
                content.append(image_block(image))
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(MODEL, prompt, schema, content))
            jobs.append({'work': str(work), 'mode': mode, 'station': station,
                'request_sha256': sha256(work / 'request.json')})
        if number % 6 == 0:
            print('Prepared', phase, number, '/', len(records_for(phase)), flush=True)
    register(output, phase, 'visual', jobs)
    print('Registered', len(jobs), 'requests', flush=True)


def infer(phase):
    output = ROOT / phase / 'visual'
    registration = read_json(output / 'visual-registration.json')
    verify_registration(registration)
    if (output / 'started.json').exists():
        raise ValueError('registered execution already started; no implicit retries')
    write_json(output / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed after registration')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(MODEL))
        if receipt['status'] == 'success':
            try:
                schema = LocalDecision if job['mode'] == 'owv_image' else RangeDecision
                decision = schema.model_validate_json(content).model_dump()
                if any(not 365 <= item['start'] <= item['end'] <= 729 for item in decision['intervals']):
                    raise ValueError('prediction outside monitoring period')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            print('Infer', phase, index, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / 'complete.json', {'requests': len(registration['jobs'])})


def confirmed_events(receipt, mode):
    return [{'kind': item.get('kind', 'transient'), 'start': item['start'], 'end': item['end'],
        'status': 'confirmed', 'phases': [], 'evidence': item['observation']}
        for item in receipt.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']


def combine(observation, base, visual_events):
    events = copy.deepcopy(base['prediction']['events'])
    for event in visual_events:
        if any(old['kind'] == event['kind'] and old['start'] <= event['end']
                and event['start'] <= old['end'] for old in events):
            continue
        events.append(copy.deepcopy(event))
    return result_from_events(observation, events)


def typed_rows(summary):
    rows = []
    for method, scores in summary['methods'].items():
        typed = scores['original'].get('types')
        for kind in EVENT_TYPES:
            scope = ('transient',) if method == 'owv_image_vision' else RANGES
            evaluated = typed is not None and (not method.endswith('_vision') or kind in scope)
            daily = typed[kind]['daily_micro'] if evaluated else {}
            ranges = scores['range'][kind] if evaluated and kind in RANGES else {}
            aff = ranges.get('affiliation', {})
            rows.append({'method': method, 'kind': kind, 'evaluated': evaluated,
                **{key: daily.get(key) for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                'aff_precision': aff.get('precision'), 'aff_recall': aff.get('recall'),
                'aff_f1': aff.get('station_macro_f1'),
                'positive_stations': aff.get('positive_stations'),
                'negative_station_fp_days': ranges.get('negative_station_fp_days')})
    return rows


def admission(methods, strata, phase):
    main, text, base = (methods[name] for name in ('nrv_image', 'nrv_text', 'vector_nearest'))
    effects = {}
    for kind in RANGES:
        def value(scores):
            return scores['range'][kind]['affiliation']['station_macro_f1']
        effects[kind] = {'aff_vs_numeric': value(main) - value(base),
            'aff_vs_text': value(main) - value(text),
            'tp_delta': main['original']['types'][kind]['daily_micro']['tp'] - base['original']['types'][kind]['daily_micro']['tp'],
            'seed_aff_vs_numeric': {seed: value(strata['cohort_seed']['nrv_image'][seed])
                - value(strata['cohort_seed']['vector_nearest'][seed])
                for seed in strata['cohort_seed']['nrv_image']}}
    improving = [kind for kind, effect in effects.items() if effect['aff_vs_numeric'] >= .01
        and effect['aff_vs_text'] >= .01 and effect['tp_delta'] > 0
        and (phase == 'pilot' or sum(v > 0 for v in effect['seed_aff_vs_numeric'].values()) >= 2)]
    fp = main['range']['transient']['negative_station_fp_days']
    old_fp = methods['owv_image']['range']['transient']['negative_station_fp_days']
    checks = {'at_least_one_range_improves': bool(improving),
        'neither_range_aff_loses_more_than_0005': all(e['aff_vs_numeric'] >= -.005 for e in effects.values()),
        'transient_negative_fp_reduced_25_percent': fp <= .75 * old_fp,
        'numerical_point_ownership': all(main['original']['types'][kind] == base['original']['types'][kind]
            for kind in ('spike', 'step'))}
    return {'admitted': all(checks.values()), 'checks': checks, 'effects': effects,
        'improving_range_types': improving, 'scope': 'simulation evidence only; not domain SOTA'}


def report(phase):
    output = ROOT / phase / 'visual'
    if not (output / 'complete.json').exists():
        raise ValueError('complete all planned predictions before scoring')
    registration = read_json(output / 'visual-registration.json')
    verify_registration(registration)
    methods = [*numerical.METHODS, 'vector_nearest', *MODES, *[m + '_vision' for m in MODES]]
    rows = {method: [] for method in methods}
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        predictions = {method: read_json(ROOT / phase / 'baseline/predictions' / method / station / 'result.json')
            for method in (*numerical.METHODS, 'vector_nearest')}
        base = predictions['vector_nearest']
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json')
            events = confirmed_events(receipt, mode)
            predictions[mode] = combine(observation, base, events)
            predictions[mode + '_vision'] = result_from_events(observation, events)
            for name in (mode, mode + '_vision'):
                write_json(output / 'final' / name / station / 'result.json', predictions[name])
        truth = read_json(DATASET / record['truth'])
        for method, result in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): aggregate_types([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    receipts = request_receipts([job['work'] for job in registration['jobs']])
    summary = {'experiment_id': 'GNSS-VLM-EXP38-' + phase.upper(), 'stations': len(registration['records']),
        'phase': phase, 'methods': summaries, 'costs': aggregate_cost(receipts),
        'type_owners': {'spike': 'numerical', 'step': 'numerical',
            'transient': 'numerical plus nonoverlapping visual proposals',
            'deformation': 'numerical plus nonoverlapping visual proposals'}}
    target = Path(f'results/exp38-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'strata.json', strata)
    decision = admission(summaries, strata, phase)
    write_json(target / 'admission.json', decision)
    write_json(target / 'request-receipts.json', receipts)
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    with (target / 'per-type.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        table = typed_rows(summary)
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(target / 'per-type.json', table)
    print(json.dumps({'costs': summary['costs'], 'decision': decision}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'baseline', 'prepare', 'infer', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation', 'test'), default='pilot')
    args = parser.parse_args()
    if args.stage == 'generate':
        generate()
    else:
        {'baseline': baseline, 'prepare': prepare, 'infer': infer, 'report': report}[args.stage](args.phase)
