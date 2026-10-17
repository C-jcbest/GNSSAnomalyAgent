"""Freeze transient-only visual ownership after a disclosed development diagnosis."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import experiment_exp38 as data
import experiment_exp39 as review
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method

from gnss_research.files import read_json, sha256, write_json

ROOT = Path('artifacts/exp40')
PROTOCOL = Path('docs/experiment-exp40.md')
MODES = {'tlv_image': ('qwen3.8-flash', 'image'), 'tlv_text': ('qwen3.8-flash', 'text')}


def guard(phase):
    if phase != 'pilot':
        previous = 'pilot' if phase == 'confirmation' else 'confirmation'
        if not read_json(Path(f'results/exp40-{previous}-v1/admission.json'))['admitted']:
            raise ValueError('previous frozen stage did not meet gate')


def configure():
    review.ROOT = ROOT
    review.PROTOCOL = PROTOCOL
    review.MODES = MODES
    review.DEPENDENCIES = [*review.DEPENDENCIES, Path(__file__), PROTOCOL]
    review.guard = guard
    data.guard = guard


def review_output(phase):
    return Path('artifacts/exp39/pilot/review') if phase == 'pilot' else ROOT / phase / 'review'


def protect_other_types(observation, base, visual_events):
    return review.union_ranges(observation, base, [e for e in visual_events if e['kind'] == 'transient'])


def report(phase):
    output = review_output(phase)
    if not (output / 'complete.json').exists():
        raise ValueError('complete all planned requests first')
    registration = read_json(output / 'registration.json')
    data.verify_registration(registration)
    methods = [*data.numerical.METHODS, 'vector_nearest', 'owv_skip_overlap', 'owv_union',
        *MODES, *[m + '_vision' for m in MODES]]
    rows = {method: [] for method in methods}
    protected = True
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(data.DATASET / record['observation'])
        base, scan, candidates = review.proposals(record, phase)
        predictions = {method: read_json(review.baseline_root(phase) / 'predictions' / method / station / 'result.json')
            for method in (*data.numerical.METHODS, 'vector_nearest')}
        scan_events = data.confirmed_events(scan, 'owv_image')
        predictions['owv_skip_overlap'] = data.combine(observation, base, scan_events)
        predictions['owv_union'] = review.union_ranges(observation, base, scan_events)
        for mode in MODES:
            source = {'tlv_image': 'lcv_flash', 'tlv_text': 'lcv_flash_text'}[mode] if phase == 'pilot' else mode
            receipt = read_json(output / 'predictions' / source / station / 'result.json')
            events = [e for e in data.confirmed_events(receipt, mode) if e['kind'] == 'transient']
            predictions[mode] = protect_other_types(observation, base, events)
            predictions[mode + '_vision'] = data.result_from_events(observation, events)
            for kind in ('spike', 'step', 'deformation'):
                protected = protected and [e for e in predictions[mode]['prediction']['events'] if e['kind'] == kind] == [
                    e for e in base['prediction']['events'] if e['kind'] == kind]
            for name in (mode, mode + '_vision'):
                write_json(ROOT / phase / 'final' / name / station / 'result.json', predictions[name])
        truth = read_json(data.DATASET / record['truth'])
        for method, result in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'], 'scenario': truth['scenario'],
                'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): data.aggregate_types([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    main, text, base = (summaries[mode] for mode in ('tlv_image', 'tlv_text', 'vector_nearest'))

    def aff(scores):
        return scores['range']['transient']['affiliation']['station_macro_f1']

    seed_deltas = {seed: aff(strata['cohort_seed']['tlv_image'][seed])
        - aff(strata['cohort_seed']['vector_nearest'][seed]) for seed in strata['cohort_seed']['tlv_image']}
    minimum = .002 if phase == 'pilot' else .01
    effects = {'aff_vs_numeric': aff(main) - aff(base), 'aff_vs_text': aff(main) - aff(text),
        'tp_vs_numeric': main['original']['types']['transient']['daily_micro']['tp']
            - base['original']['types']['transient']['daily_micro']['tp']}
    checks = {'aff_vs_numeric': effects['aff_vs_numeric'] >= minimum,
        'aff_vs_text': effects['aff_vs_text'] >= minimum, 'actual_tp_gain': effects['tp_vs_numeric'] > 0,
        'negative_station_fp_halved': main['range']['transient']['negative_station_fp_days']
            <= .5 * summaries['owv_union']['range']['transient']['negative_station_fp_days'],
        'all_other_event_types_preserved': protected,
        'at_least_two_seeds_positive': phase == 'pilot' or sum(v > 0 for v in seed_deltas.values()) >= 2}
    jobs = [job for job in registration['jobs'] if phase != 'pilot' or job['mode'] in ('lcv_flash', 'lcv_flash_text')]
    receipts = request_receipts([job['work'] for job in jobs])
    scan_jobs = [] if phase == 'pilot' else read_json(ROOT / phase / 'scan/registration.json')['jobs']
    scan_receipts = request_receipts([job['work'] for job in scan_jobs])
    summary = {'experiment_id': 'GNSS-VLM-EXP40-' + phase.upper(), 'stations': len(registration['records']),
        'methods': summaries, 'review_costs': aggregate_cost(receipts),
        'new_requests': 0 if phase == 'pilot' else len(jobs) + len(scan_jobs),
        'scan_costs': aggregate_cost(scan_receipts) if scan_receipts else {'reused_stage_one_requests': 18},
        'ablation_scope': 'review stage only; visual scan proposals shared',
        'type_owners': {'spike': 'numerical', 'step': 'numerical', 'transient': 'numerical union visual review',
            'deformation': 'numerical, no new visual contribution'}}
    target = Path(f'results/exp40-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'strata.json', strata)
    write_json(target / 'admission.json', {'admitted': all(checks.values()), 'checks': checks,
        'effects': effects, 'seed_aff_deltas': seed_deltas, 'protected_event_lists_equal': protected})
    table = data.typed_rows(summary)
    for row in table:
        if row['method'].endswith('_vision') and row['kind'] != 'transient':
            row.update(evaluated=False, **{k: None for k in row if k not in ('method', 'kind', 'evaluated')})
    write_json(target / 'per-type.json', table)
    write_json(target / 'request-receipts.json', {'review': receipts, 'scan': scan_receipts,
        'source_registration': str(output / 'registration.json'), 'sha256': sha256(output / 'registration.json')})
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'new_requests': summary['new_requests'], 'checks': checks, 'effects': effects,
        'seed_aff_deltas': seed_deltas}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare-scan', 'infer-scan', 'prepare', 'infer', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation', 'test'), default='pilot')
    args = parser.parse_args()
    configure()
    if args.stage == 'baseline':
        data.baseline(args.phase)
    elif args.stage == 'prepare-scan':
        review.prepare_scan(args.phase)
    elif args.stage == 'infer-scan':
        review.infer(args.phase, 'scan')
    elif args.stage == 'prepare':
        review.prepare(args.phase)
    elif args.stage == 'infer':
        review.infer(args.phase, 'review')
    else:
        report(args.phase)
