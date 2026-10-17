"""Visual-owned transient identification with fallible local boundary-model evidence."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiment_exp38 as data
import experiment_exp39 as local
import numpy as np
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values
from experiment_exp30 import LocalDecision
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method, compare

from gnss_research.data import SCENARIOS, simulate
from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.provider import send_request
from gnss_research.segmentation import jump_evidence

ROOT = Path('artifacts/exp41')
DATASET = ROOT / 'data'
PROTOCOL = Path('docs/experiment-exp41.md')
MODES = ('bmv_image', 'bmv_text', 'bmv_no_evidence')
SEEDS = (2026101741, 2026101742, 2026101743)
DEPENDENCIES = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp39.py'),
    *data.DEPENDENCIES, Path('scripts/experiment_exp30.py'),
    Path('scripts/experiment_exp26.py'), Path('scripts/experiment_exp25.py')]
PROMPT = """Decide which GNSS displacement ranges really contain TEMPORARY OFFSETS.
You own the FINAL transient decisions. Nothing is automatically accepted from the proposals.
Return summary and intervals (original inclusive start/end, confirmed/uncertain, observation).
The first shifted sample is start; end is the LAST shifted sample BEFORE observed return.
Only temporary offsets, not deformation ranges, spikes, permanent steps, or internal phases.

First inspect original full N/E/U displacement and normal reference 180..364. Then compare
BOTH edges and the shifted interior in the original local observations. Approximate visual
and local numerical proposals are fallible, not expected events or ground truth. You may
reject all, adjust bounds, or add visibly supported intervals in monitoring indices 365..729.
The plots/text show the same original samples and 3-day median reading aid. Local views are
centered independently; compare CHANGES, not absolute heights across cropped panels. Gaps
are unknown, never zero; do not infer an unobserved return or a boundary hidden by a gap.

A transient requires a genuinely abrupt offset followed by an opposing abrupt return in the
same component(s), with multiple shifted samples between. One clear component can suffice.
Smooth acceleration/deceleration and a plateau after motion arrests are NOT a shift/return.
Continuous displacement can rise and then stop without an opposing jump. A permanent jump
with a later smooth slope change is not a transient. Inspect how raw points cross each edge:
is there a discontinuity, or do they simply continue along a smooth bend? Independent motion
or a permanent step may coexist inside a real offset: do not require identical outside levels,
adjacent jump pairing, or a flat interior. Weak components do not veto a clear other component.

When supplied, numerical boundary evidence compares four LOCAL fits to actual observed
samples: linear (continuous straight), hinge (continuous slope change), step (abrupt offset
plus trend), step_hinge (abrupt offset AND slope change). Smaller criterion fits better after
the disclosed complexity penalty. A hinge does NOT itself establish an offset. Models and
local median deltas can be distorted by noise, spikes, nearby changes and ongoing movement;
they are evidence, not a mandatory gate. Alternative nearby numerical edge dates are only
suggestions. Compare both their fit evidence and raw shape; numerical deltas that point in
opposite directions alone do not prove two discontinuities. If numerical evidence is absent,
use the raw observations without inventing fit results. There is no global mm/speed threshold.

For each confirmed interval give concise axis-specific observations of the abrupt onset,
shifted interior and abrupt return, considering continuous-motion alternatives. Reject a
candidate or mark uncertain if those observations cannot be established. No expected count.
"""


def records_for(phase):
    records = read_json(DATASET / 'manifest.json')['records']
    if phase == 'test':
        return [record for record in records if record['split'] == 'test']
    return [record for record in records if record['split'] == 'validation'
        and (record['within_cohort'] < 6) == (phase == 'pilot')]


def generate():
    DATASET.mkdir(parents=True, exist_ok=False)
    previous = [Path('data/exp28-expanded-1528-1530'),
        Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp29/data'),
        Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp30/data'),
        Path('artifacts/exp38/data')]
    used = {read_json(folder / record['truth'])['seed'] for folder in previous
        for record in read_json(folder / 'manifest.json')['records']}
    records = []
    for group, seed in enumerate(SEEDS):
        streams = np.random.SeedSequence(seed).spawn(282)
        for split, count, offset in (('validation', 42, 0), ('test', 240, 42)):
            for index in range(count):
                case_seed = int(streams[offset + index].generate_state(1)[0])
                if case_seed in used:
                    raise ValueError('mechanism seed already used')
                used.add(case_seed)
                station = f"{'v' if split == 'validation' else 'e'}{group * count + index:04d}"
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
        'split_counts': {'validation': 126, 'test': 720}, 'records': records,
        'origin': 'unchanged mechanism simulation, not field data', 'sources': source_hashes(),
        'script_sha256': sha256(Path(__file__)),
        'seed_exclusion_manifests': {str(p): sha256(p / 'manifest.json') for p in previous}})
    verify_dataset(DATASET)
    print('Generated 18 pilot, 108 confirmation, 720 sealed test stations', flush=True)


def guard(phase):
    if phase == 'confirmation':
        if not read_json(Path('results/exp41-pilot-v1/admission.json'))['admitted']:
            raise ValueError('pilot failed; do not expand')
    if phase == 'test':
        raise ValueError('test requires a separately registered execution protocol')


def register(output, phase, stage, jobs):
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP41',
        'phase': phase, 'stage': stage, 'records': records_for(phase), 'jobs': jobs,
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}})


def configure():
    data.ROOT = ROOT
    data.DATASET = DATASET
    data.records_for = records_for
    data.guard = guard
    data.register = register
    data.DEPENDENCIES = DEPENDENCIES
    data.initialize_worker = initialize_worker


def initialize_worker(model_path):
    # Windows spawns a fresh interpreter; parent module assignments do not propagate.
    data.DATASET = DATASET
    data.numerical.DATASET = DATASET
    data.numerical.initialize_worker(model_path)


def prepare_scan(phase):
    guard(phase)
    output = ROOT / phase / 'scan'
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        image = output / 'inputs' / (station + '.png')
        image.parent.mkdir(exist_ok=True)
        data.old_visual.render_sheet(observation, image)
        work = output / 'predictions' / station
        content = [{'type': 'text', 'text': json.dumps({'monitoring_start': 365,
            'last_index': 729})}, image_block(image)]
        write_json(work / 'request.json', payload(data.MODEL, data.old_visual.PROMPT,
            LocalDecision, content))
        jobs.append({'work': str(work), 'station': station, 'mode': 'owv',
            'request_sha256': sha256(work / 'request.json')})
    register(output, phase, 'scan', jobs)


def boundary_evidence(values, candidate, jump_indices):
    """Describe both putative boundaries without turning fit preference into a label."""
    observed = np.isfinite(values).all(axis=1)
    time = np.flatnonzero(observed)
    samples = values[observed]
    result = []
    for side, edge in (('onset', candidate['start']), ('return', candidate['end'] + 1)):
        alternatives = sorted({edge, *[day for day in jump_indices if abs(day - edge) <= 21]})
        fits = []
        for day in alternatives:
            if day >= len(values):
                continue
            fit = jump_evidence(time, samples, day, 365, len(values) - 1)
            before = values[max(365, day - 7):day]
            after = values[day:min(len(values), day + 7)]
            before = before[np.isfinite(before).all(axis=1)]
            after = after[np.isfinite(after).all(axis=1)]
            delta = None
            if len(before) and len(after):
                delta = (np.median(after, axis=0) - np.median(before, axis=0)).tolist()
            fits.append({**fit, 'local_median_change_NEU': delta,
                'before_observed': len(before), 'after_observed': len(after)})
        result.append({'candidate': candidate['id'], 'side': side,
            'proposed_index': edge, 'local_fits': fits})
    return result


def proposals(phase, station):
    basepath = ROOT / phase / 'baseline/predictions'
    base = read_json(basepath / 'vector_nearest' / station / 'result.json')
    segmented = read_json(basepath / 'segment_numeric' / station / 'result.json')
    scan = read_json(ROOT / phase / 'scan/predictions' / station / 'result.json')
    events = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] == 'transient']
    events.extend(data.confirmed_events(scan, 'owv_image'))
    candidates = []
    seen = set()
    for event in events:
        key = (event['start'], event['end'])
        if key not in seen:
            seen.add(key)
            candidates.append({'id': f'R{len(candidates) + 1}', 'kind': 'transient',
                'start': event['start'], 'end': event['end']})
    return base, segmented, scan, candidates


def prepare_review(phase):
    guard(phase)
    output = ROOT / phase / 'review'
    output.mkdir(parents=True, exist_ok=False)
    jobs, input_hashes = [], []
    for index, record in enumerate(records_for(phase), 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        base, segmented, scan, candidates = proposals(phase, station)
        values = plot_values(observation)
        views = local.local_views(values, candidates)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True)
        overview = inputs / 'overview.png'
        data.render_sheet({'original': values}, overview, 'nrv_raw_only')
        images = [overview]
        for start in range(0, len(views), 6):
            path = inputs / f'edges-{start // 6}.png'
            local.render_local(views[start:start + 6], path)
            images.append(path)
        evidence = [item for candidate in candidates for item in boundary_evidence(values,
            candidate, [jump['index'] for jump in segmented['jumps']])]
        write_json(inputs / 'evidence.json', {'candidates': candidates, 'boundary_evidence': evidence})
        for mode in MODES:
            metadata = {'monitoring_indices': [365, 729], 'normal_reference': [180, 364],
                'fallible_proposals': candidates}
            if mode != 'bmv_no_evidence':
                metadata['boundary_evidence'] = evidence
            content = [{'type': 'text', 'text': json.dumps(metadata, allow_nan=False)}]
            if mode == 'bmv_text':
                for left, right in ((180, 365), (365, 730)):
                    content.append({'type': 'text', 'text': json.dumps({'view': 'original overview',
                        'first_index': left, 'samples_NEU': finite_rows(values[left:right]),
                        'median3_NEU': finite_rows(data.median_line(values[left:right]))})})
                for view in views:
                    content.append({'type': 'text', 'text': json.dumps({'view': view['id'],
                        'proposed_index': view['candidate_index'], 'first_index': view['left'],
                        'samples_NEU': finite_rows(view['values']),
                        'median3_NEU': finite_rows(data.median_line(view['values']))})})
            else:
                content.extend(image_block(path) for path in images)
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(data.MODEL, PROMPT, LocalDecision, content))
            jobs.append({'work': str(work), 'station': station, 'mode': mode,
                'request_sha256': sha256(work / 'request.json')})
        input_hashes.extend({'path': str(path), 'sha256': sha256(path)} for path in [
            ROOT / phase / 'scan/predictions' / station / 'result.json',
            ROOT / phase / 'baseline/predictions/vector_nearest' / station / 'result.json',
            ROOT / phase / 'baseline/predictions/segment_numeric' / station / 'result.json'])
        print('Prepared review', phase, index, '/', len(records_for(phase)), flush=True)
    register(output, phase, 'review', jobs)
    write_json(output / 'input-hashes.json', input_hashes)


def verify_inputs(output, registration):
    data.verify_registration(registration)
    for record in registration['records']:
        if sha256(DATASET / record['truth']) != record['truth_sha256']:
            raise ValueError('registered truth changed')
    if (output / 'input-hashes.json').exists():
        for item in read_json(output / 'input-hashes.json'):
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('upstream prediction changed')


def infer(phase, stage):
    output = ROOT / phase / stage
    registration = read_json(output / 'registration.json')
    verify_inputs(output, registration)
    write_json(output / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('registered request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings(data.MODEL))
        if receipt['status'] == 'success':
            try:
                decision = LocalDecision.model_validate_json(content).model_dump()
                if any(not 365 <= item['start'] <= item['end'] <= 729 for item in decision['intervals']):
                    raise ValueError('prediction outside target')
                receipt['decision'] = decision
            except (ValueError, KeyError, TypeError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['station'], job['mode'], receipt['status']

    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, status in enumerate(executor.map(call, registration['jobs']), 1):
            print('Infer', phase, stage, index, '/', len(registration['jobs']), *status, flush=True)
    write_json(output / 'complete.json', {'requests': len(registration['jobs'])})


def visual_owned(observation, base, receipt):
    events = [copy.deepcopy(e) for e in base['prediction']['events'] if e['kind'] != 'transient']
    additions = data.confirmed_events(receipt, 'owv_image') if receipt['status'] == 'success' else []
    result = data.result_from_events(observation, [*events, *additions])
    result['visual_status'] = receipt['status']
    return result


def admission(methods, strata, protected, phase):
    main, text, plain = (methods[mode] for mode in MODES)

    def affiliation(scores):
        return scores['range']['transient']['affiliation']['station_macro_f1']

    def count(scores, key):
        return scores['original']['types']['transient']['daily_micro'][key]

    effect = {'aff_vs_text': affiliation(main) - affiliation(text),
        'aff_vs_no_evidence': affiliation(main) - affiliation(plain),
        'fp_vs_text': count(main, 'fp') - count(text, 'fp'),
        'fp_vs_no_evidence': count(main, 'fp') - count(plain, 'fp'),
        'tp_vs_no_evidence': count(main, 'tp') - count(plain, 'tp')}
    seed_deltas = {seed: affiliation(strata['cohort_seed']['bmv_image'][seed])
        - affiliation(strata['cohort_seed']['bmv_text'][seed])
        for seed in strata['cohort_seed']['bmv_image']}
    checks = {'evidence_aff_preserved': effect['aff_vs_no_evidence'] >= -.01,
        'evidence_gain': effect['aff_vs_no_evidence'] >= .01 or (
            count(plain, 'fp') > 0 and count(main, 'fp') <= .8 * count(plain, 'fp')),
        'visual_gain': effect['aff_vs_text'] >= .02 or (effect['aff_vs_text'] >= -.01
            and count(text, 'fp') > 0 and count(main, 'fp') <= .8 * count(text, 'fp')),
        'real_coverage_preserved': count(main, 'tp') >= .9 * count(plain, 'tp'),
        'transient_aff_at_least_080': affiliation(main) >= .8,
        'other_types_preserved': protected,
        'two_seeds_nonnegative': phase == 'pilot' or sum(v >= 0 for v in seed_deltas.values()) >= 2}
    return {'admitted': all(checks.values()), 'checks': checks,
        'effects': effect, 'seed_aff_deltas': seed_deltas}


def report(phase):
    output = ROOT / phase / 'review'
    registration = read_json(output / 'registration.json')
    verify_inputs(output, registration)
    if not (output / 'complete.json').exists():
        raise ValueError('finish planned requests before scoring')
    methods = [*data.numerical.METHODS, 'vector_nearest', 'bmv_scan_only', *MODES,
        *[mode + '_vision' for mode in MODES]]
    rows = {method: [] for method in methods}
    protected, paired = True, []
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        base, segmented, scan, candidates = proposals(phase, station)
        predictions = {method: read_json(ROOT / phase / 'baseline/predictions' / method
            / station / 'result.json') for method in (*data.numerical.METHODS, 'vector_nearest')}
        predictions['bmv_scan_only'] = visual_owned(observation, base, scan)
        receipts = {}
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json')
            receipts[mode] = receipt
            predictions[mode] = visual_owned(observation, base, receipt)
            additions = data.confirmed_events(receipt, 'owv_image') if receipt['status'] == 'success' else []
            predictions[mode + '_vision'] = data.result_from_events(observation, additions)
            for kind in ('spike', 'step', 'deformation'):
                protected = protected and [e for e in predictions[mode]['prediction']['events']
                    if e['kind'] == kind] == [e for e in base['prediction']['events'] if e['kind'] == kind]
            write_json(ROOT / phase / 'final' / mode / station / 'result.json', predictions[mode])
        if receipts['bmv_image']['status'] == receipts['bmv_text']['status'] == 'success':
            paired.append(station)
        truth = read_json(DATASET / record['truth'])
        for method, result in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'],
                **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): data.aggregate_types([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    scan_jobs = read_json(ROOT / phase / 'scan/registration.json')['jobs']
    receipts = request_receipts([job['work'] for job in [*scan_jobs, *registration['jobs']]])
    summary = {'experiment_id': 'GNSS-VLM-EXP41-' + phase.upper(),
        'stations': len(registration['records']), 'methods': summaries, 'costs': aggregate_cost(receipts),
        'type_owners': {'spike': 'numeric', 'step': 'numeric', 'transient': 'visual review only',
            'deformation': 'numeric, no new visual contribution'},
        'ablation_scope': 'review-stage modality only; scan shared; vector_nearest fully no vision'}
    target = Path(f'results/exp41-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'strata.json', strata)
    decision = admission(summaries, strata, protected, phase)
    write_json(target / 'admission.json', decision)
    table = data.typed_rows(summary)
    for row in table:
        if row['method'].endswith('_vision') and row['kind'] != 'transient':
            row.update(evaluated=False, **{key: None for key in row if key not in ('method', 'kind', 'evaluated')})
    write_json(target / 'per-type.json', table)
    write_json(target / 'request-receipts.json', receipts)
    write_json(target / 'paired-sensitivity.json', {'successful_pairs': len(paired),
        'stations': paired, 'comparison': compare({method: [row for row in values if row['id'] in paired]
            for method, values in rows.items()}, 'bmv_image', 'bmv_text') if paired else None})
    write_json(target / 'contribution.json', {'vs_no_evidence': compare(rows, 'bmv_image', 'bmv_no_evidence'),
        'vs_numeric_diagnostic': compare(rows, 'bmv_image', 'vector_nearest')})
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'costs': summary['costs'], 'admission': decision}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'baseline', 'prepare-scan', 'infer-scan',
        'prepare-review', 'infer-review', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation', 'test'), default='pilot')
    args = parser.parse_args()
    configure()
    if args.stage == 'generate':
        generate()
    elif args.stage == 'baseline':
        data.baseline(args.phase)
    elif args.stage == 'prepare-scan':
        prepare_scan(args.phase)
    elif args.stage == 'prepare-review':
        prepare_review(args.phase)
    elif args.stage.startswith('infer-'):
        infer(args.phase, args.stage.removeprefix('infer-'))
    else:
        report(args.phase)
