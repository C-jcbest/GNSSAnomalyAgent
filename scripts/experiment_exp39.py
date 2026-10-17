"""Local raw-continuity checking, explicitly conditional on OWV visual proposals."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiment_exp38 as previous
import matplotlib.pyplot as plt
import numpy as np
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values
from experiment_exp30 import LocalDecision
from matplotlib.ticker import MultipleLocator
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request

ROOT = Path('artifacts/exp39')
PROTOCOL = Path('docs/experiment-exp39.md')
MODES = {'lcv_flash': ('qwen3.8-flash', 'image'), 'lcv_flash_text': ('qwen3.8-flash', 'text'),
    'lcv_vl': ('qwen3-vl-plus', 'image'), 'lcv_vl_text': ('qwen3-vl-plus', 'text')}
DEPENDENCIES = [Path(__file__), PROTOCOL, *previous.DEPENDENCIES]
PROMPT = """Check fallible GNSS RANGE proposals against actual displacement observations.
Return exactly one JSON object with summary and intervals. Every interval must contain
kind (transient/deformation), original inclusive start and end, status (confirmed/uncertain),
and observation (brief supporting component-specific evidence). Empty intervals is valid.
The proposals are NOT truth, NOT expected anomalies. You may reject, adjust, or add events.
The full original N/E/U traces establish morphology. Historical samples 180..364 are normal
reference. Each local panel examines one candidate edge; edges can be wrong. Gray is actual
observed data; blue 3-day median is only a reading aid. Missing samples are unknown.
Local windows are vertically centered on their own first 7 observed samples: compare local change
directions, NEVER absolute levels across different local panels. All times are original
indices, not offsets in a cropped window. Text input contains exactly the same samples.

Use a deliberate check: locate the actual change within each neighborhood; inspect the
interior and exterior in the full original curve; verify if the OTHER edge supports the
SAME proposed morphology. If the graph is ordinary noise, a single spike, a permanently
shifted plateau, or ordinary historical drift/seasonality, reject the range proposal.
TRANSIENT requires abrupt onset, persistent shifted interior, and opposing observed return
in the same component(s). One clear component can suffice. Smooth movement and a separate
permanent step can coexist: compare local boundary directions, not absolute outside levels.
The final shifted sample is BEFORE the return; an onset and return need not match proposed
indices. Do not mistake a permanent step followed by noise for a shift-and-return pair.
DEFORMATION requires sustained CONTINUOUS movement visible in ORIGINAL displacement, from
departure into continued activity to observable arrest. A step itself and a static elevated
plateau are not sustained activity. Do not extend activity into the arrested plateau or
infer early onset from a derivative. No internal phases. Small motion must be supported by
persistent raw trajectory, not only short noisy deviations or a residual excursion.
Both kinds can coexist and overlap. A gap can obscure boundaries: label only the actually
observed changed interior if both morphology and opposing return/arrest are supported.
Use uncertain if not supported. Do not invent missing boundaries, a return after the record,
physical causes, universal millimeter/speed cutoffs, point spikes or permanent step events.
"""


def baseline_root(phase):
    return previous.ROOT / phase / 'baseline'


def stage_one_root(phase):
    return previous.ROOT / 'pilot/visual/predictions/owv_image' if phase == 'pilot' else ROOT / phase / 'scan/predictions'


def register(output, phase, stage, jobs):
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP39',
        'phase': phase, 'stage': stage, 'records': previous.records_for(phase), 'jobs': jobs,
        'manifest_sha256': sha256(previous.DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}})


def guard(phase):
    if phase != 'pilot':
        prior = 'pilot' if phase == 'confirmation' else 'confirmation'
        if not read_json(Path(f'results/exp39-{prior}-v1/admission.json'))['admitted']:
            raise ValueError('previous phase failed; do not expand')


def prepare_scan(phase):
    guard(phase)
    if phase == 'pilot':
        raise ValueError('pilot reuses already recorded OWV responses; no replay')
    output = ROOT / phase / 'scan'
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for record in previous.records_for(phase):
        station = record['station_id']
        observation = read_json(previous.DATASET / record['observation'])
        image = output / 'inputs' / (station + '.png')
        image.parent.mkdir(exist_ok=True)
        previous.old_visual.render_sheet(observation, image)
        work = output / 'predictions' / station
        content = [{'type': 'text', 'text': json.dumps({'monitoring_start': 365, 'last_index': 729})}, image_block(image)]
        write_json(work / 'request.json', payload(previous.MODEL, previous.old_visual.PROMPT, LocalDecision, content))
        jobs.append({'work': str(work), 'station': station, 'mode': 'owv', 'model': previous.MODEL,
            'request_sha256': sha256(work / 'request.json')})
    register(output, phase, 'scan', jobs)


def proposals(record, phase):
    station = record['station_id']
    base = read_json(baseline_root(phase) / 'predictions/vector_nearest' / station / 'result.json')
    receipt = read_json(stage_one_root(phase) / station / 'result.json')
    events = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] in previous.RANGES]
    for item in receipt.get('decision', {}).get('intervals', []):
        events.append({'kind': 'transient', 'start': item['start'], 'end': item['end']})
    seen, candidates = set(), []
    for event in events:
        key = (event['kind'], event['start'], event['end'])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({'id': f'R{len(candidates) + 1}', 'kind': event['kind'],
            'start': event['start'], 'end': event['end']})
    return base, receipt, candidates


def local_views(values, candidates):
    views = []
    for event in candidates:
        for edge, day in (('start', event['start']), ('after_end', event['end'] + 1)):
            left, right = max(365, day - 21), min(730, day + 22)
            local = values[left:right].copy()
            observed = local[np.isfinite(local).all(axis=1)]
            if len(observed):
                local -= np.median(observed[:7], axis=0)
            views.append({'id': event['id'] + '_' + edge, 'candidate_index': day,
                'left': left, 'right': right, 'values': local})
    return views


def render_local(views, path):
    if not views:
        return
    figure, panels = plt.subplots(3, len(views), figsize=(4.4 * len(views), 7.5),
        dpi=120, squeeze=False, sharey='row')
    for column, view in enumerate(views):
        time = np.arange(view['left'], view['right'])
        values = view['values']
        smooth = previous.median_line(values)
        for axis in range(3):
            panel = panels[axis, column]
            panel.plot(time, values[:, axis], '.-', color='#888888', linewidth=.8, markersize=3)
            panel.plot(time, smooth[:, axis], color='#246e8c', linewidth=1.1)
            panel.axvline(view['candidate_index'], linestyle='--', color='#b14b32', linewidth=.8)
            panel.set_title(f"{view['id']} | {'NEU'[axis]} | proposed {view['candidate_index']}", fontsize=9)
            panel.set_xlim(view['left'], view['right'] - 1)
            panel.xaxis.set_major_locator(MultipleLocator(10))
            panel.xaxis.set_minor_locator(MultipleLocator(5))
            panel.grid(alpha=.25)
            panel.tick_params(axis='x', rotation=60, labelsize=8)
            panel.set_ylabel('Local centered / station MAD')
    figure.suptitle('Fallible proposed edges: examine actual raw samples, not dashed lines alone')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare(phase):
    guard(phase)
    output = ROOT / phase / 'review'
    output.mkdir(parents=True, exist_ok=False)
    jobs, input_receipts = [], []
    for index, record in enumerate(previous.records_for(phase), 1):
        station = record['station_id']
        observation = read_json(previous.DATASET / record['observation'])
        base, receipt, candidates = proposals(record, phase)
        values = plot_values(observation)
        views = local_views(values, candidates)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True)
        overview = inputs / 'overview.png'
        previous.render_sheet({'original': values}, overview, 'nrv_raw_only')
        images = [overview]
        # Split large candidate sets into readable groups without dropping any candidate.
        for start in range(0, len(views), 6):
            path = inputs / f'edges-{start // 6}.png'
            render_local(views[start:start + 6], path)
            images.append(path)
        write_json(inputs / 'candidates.json', candidates)
        for mode, (model, modality) in MODES.items():
            content = [{'type': 'text', 'text': json.dumps({'normal_visible_indices': [180, 364],
                'monitoring_indices': [365, 729], 'fallible_range_proposals': candidates})}]
            if modality == 'image':
                content.extend(image_block(path) for path in images)
            else:
                for left, right in ((180, 365), (365, 730)):
                    content.append({'type': 'text', 'text': json.dumps({'view': 'original overview',
                        'first_index': left, 'samples_N_E_U': finite_rows(values[left:right]),
                        'median3_N_E_U': finite_rows(previous.median_line(values[left:right]))}, allow_nan=False)})
                for view in views:
                    content.append({'type': 'text', 'text': json.dumps({'view': view['id'],
                        'proposed_index': view['candidate_index'], 'first_index': view['left'],
                        'samples_N_E_U': finite_rows(view['values']),
                        'median3_N_E_U': finite_rows(previous.median_line(view['values']))}, allow_nan=False)})
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(model, PROMPT, previous.RangeDecision, content))
            jobs.append({'work': str(work), 'station': station, 'mode': mode, 'model': model,
                'request_sha256': sha256(work / 'request.json')})
        input_receipts.append({'station': station,
            'stage_one_result_sha256': sha256(stage_one_root(phase) / station / 'result.json'),
            'numerical_result_sha256': sha256(baseline_root(phase) / 'predictions/vector_nearest' / station / 'result.json')})
        print('Prepared local review', phase, index, '/', len(previous.records_for(phase)), flush=True)
    register(output, phase, 'review', jobs)
    write_json(output / 'input-receipts.json', input_receipts)


def infer(phase, stage):
    output = ROOT / phase / stage
    registration = read_json(output / 'registration.json')
    previous.verify_registration(registration)
    if (output / 'started.json').exists():
        raise ValueError('already started, no implicit retries')
    write_json(output / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(job['model']))
        if receipt['status'] == 'success':
            try:
                schema = LocalDecision if stage == 'scan' else previous.RangeDecision
                decision = schema.model_validate_json(content).model_dump()
                if any(not 365 <= item['start'] <= item['end'] <= 729 for item in decision['intervals']):
                    raise ValueError('prediction outside observed target')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, result in enumerate(executor.map(call, registration['jobs']), 1):
            print('Infer', phase, stage, index, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / 'complete.json', {'requests': len(registration['jobs'])})


def union_ranges(observation, base, additions):
    points = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] not in previous.RANGES]
    ranges = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] in previous.RANGES]
    ranges.extend(copy.deepcopy(additions))
    merged = []
    for kind in previous.RANGES:
        selected = sorted((e for e in ranges if e['kind'] == kind), key=lambda e: (e['start'], e['end']))
        for event in selected:
            if merged and merged[-1]['kind'] == kind and event['start'] <= merged[-1]['end']:
                merged[-1]['end'] = max(merged[-1]['end'], event['end'])
                merged[-1]['phases'] = []
                merged[-1]['evidence'] = 'Union of same-type numerical/visual observed ranges'
            else:
                merged.append(event)
    return previous.result_from_events(observation, [*points, *merged])


def report(phase):
    output = ROOT / phase / 'review'
    if not (output / 'complete.json').exists():
        raise ValueError('finish all planned requests before scoring')
    registration = read_json(output / 'registration.json')
    previous.verify_registration(registration)
    methods = [*previous.numerical.METHODS, 'vector_nearest', 'owv_skip_overlap', 'owv_union',
        *MODES, *[m + '_vision' for m in MODES]]
    rows = {name: [] for name in methods}
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(previous.DATASET / record['observation'])
        base, scan, candidates = proposals(record, phase)
        predictions = {method: read_json(baseline_root(phase) / 'predictions' / method / station / 'result.json')
            for method in (*previous.numerical.METHODS, 'vector_nearest')}
        scan_events = previous.confirmed_events(scan, 'owv_image')
        predictions['owv_skip_overlap'] = previous.combine(observation, base, scan_events)
        predictions['owv_union'] = union_ranges(observation, base, scan_events)
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json')
            events = previous.confirmed_events(receipt, mode)
            predictions[mode] = union_ranges(observation, base, events)
            predictions[mode + '_vision'] = previous.result_from_events(observation, events)
            for name in (mode, mode + '_vision'):
                write_json(output / 'final' / name / station / 'result.json', predictions[name])
        truth = read_json(previous.DATASET / record['truth'])
        for method, result in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'], 'scenario': truth['scenario'],
                'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): previous.aggregate_types([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    gates = {}
    minimum = .002 if phase == 'pilot' else .01
    for mode in ('lcv_flash', 'lcv_vl'):
        main, text, base = (summaries[name] for name in (mode, mode + '_text', 'vector_nearest'))
        def aff(scores, kind):
            return scores['range'][kind]['affiliation']['station_macro_f1']

        def tp(scores):
            return scores['original']['types']['transient']['daily_micro']['tp']
        effects = {'transient_aff_vs_numeric': aff(main, 'transient') - aff(base, 'transient'),
            'transient_aff_vs_text': aff(main, 'transient') - aff(text, 'transient'),
            'transient_tp_vs_numeric': tp(main) - tp(base),
            'deformation_aff_vs_numeric': aff(main, 'deformation') - aff(base, 'deformation')}
        seed_deltas = {seed: aff(strata['cohort_seed'][mode][seed], 'transient')
            - aff(strata['cohort_seed']['vector_nearest'][seed], 'transient')
            for seed in strata['cohort_seed'][mode]}
        checks = {'transient_aff_vs_numeric': effects['transient_aff_vs_numeric'] >= minimum,
            'transient_aff_vs_text': effects['transient_aff_vs_text'] >= minimum,
            'actual_tp_gain': effects['transient_tp_vs_numeric'] > 0,
            'negative_station_fp_halved': main['range']['transient']['negative_station_fp_days']
                <= .5 * summaries['owv_union']['range']['transient']['negative_station_fp_days'],
            'deformation_preserved': effects['deformation_aff_vs_numeric'] >= -.005,
            'points_preserved': all(main['original']['types'][kind] == base['original']['types'][kind]
                for kind in ('spike', 'step')),
            'two_seeds_positive': phase == 'pilot' or sum(v > 0 for v in seed_deltas.values()) >= 2}
        gates[mode] = {'admitted': all(checks.values()), 'checks': checks, 'effects': effects,
            'seed_aff_deltas': seed_deltas}
    primary = read_json(Path('results/exp39-pilot-v1/admission.json'))['selected'] if phase != 'pilot' else None
    eligible = [mode for mode, value in gates.items() if value['admitted'] and (primary is None or mode == primary)]
    eligible.sort(key=lambda mode: (-summaries[mode]['range']['transient']['affiliation']['station_macro_f1'], mode))
    decision = {'admitted': bool(eligible), 'selected': eligible[0] if eligible else None, 'models': gates,
        'scope': 'pilot is already seen development; confirmation/test must remain separate'}
    receipts = request_receipts([job['work'] for job in registration['jobs']])
    summary = {'experiment_id': 'GNSS-VLM-EXP39-' + phase.upper(), 'stations': len(registration['records']),
        'methods': summaries, 'costs': aggregate_cost(receipts), 'ablation_scope': 'review-stage modality only; visual scan shared'}
    target = Path(f'results/exp39-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'strata.json', strata)
    write_json(target / 'admission.json', decision)
    write_json(target / 'per-type.json', previous.typed_rows(summary))
    write_json(target / 'request-receipts.json', receipts)
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'costs': summary['costs'], 'decision': decision}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare-scan', 'infer-scan', 'prepare', 'infer', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation', 'test'), default='pilot')
    args = parser.parse_args()
    if args.stage == 'baseline':
        guard(args.phase)
        previous.guard = guard
        previous.baseline(args.phase)
    elif args.stage == 'prepare-scan':
        prepare_scan(args.phase)
    elif args.stage == 'infer-scan':
        infer(args.phase, 'scan')
    elif args.stage == 'prepare':
        prepare(args.phase)
    elif args.stage == 'infer':
        infer(args.phase, 'review')
    else:
        report(args.phase)
