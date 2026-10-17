"""Explore nearby boundaries before visual interpretation instead of vetoing coarse dates."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import experiment_exp41 as parent
import numpy as np
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmentation import jump_evidence

ROOT = Path('artifacts/exp43')
DATASET = Path('artifacts/exp41/data')
PROTOCOL = Path('docs/experiment-exp43.md')
DEPENDENCIES = [Path(__file__), PROTOCOL, *parent.DEPENDENCIES]
MODES = parent.MODES
PROMPT = """Identify temporary GNSS observation offsets from original N/E/U curves.
Return summary and intervals with original inclusive start/end, confirmed/uncertain status,
and a concise observation. You own final transient output; no numerical event is accepted
automatically. Monitoring indices are 365..729, normal reference is 180..364. Missing samples
are unknown, not zero. Gray is raw data and blue 3-day median is only a readability aid.

Initial ranges are approximate. Explore BOTH boundary neighborhoods before deciding that a
return is absent. A proposal a few days from a true edge can have weak fit evidence even when
a real edge is nearby. Additional panels show SEARCHED edges, not known events. Read actual
indices on each panel and distinguish onset from return. All three axes show one station.
Local panels are independently centered: compare local CHANGES, not their absolute heights.

A temporary offset has an abrupt onset, a shifted multi-sample interior, then an opposing
observed return on the SAME component(s). One clear component suffices. It need not form a
flat rectangle or restore the same absolute outside levels: ongoing motion or a permanent
step can coexist. In particular, an internal larger step is NOT necessarily the return.
Compare onset/return directions and magnitudes on all usable axes, allowing weak axes, noise,
and smooth drift. Do not pair adjacent or biggest jumps automatically. Check later edge
alternatives when an internal step is inconsistent with the onset vector. A short spike,
smooth acceleration/deceleration, permanent step alone, or plateau after motion arrests is
NOT a temporary offset. Original displacement establishes the shape; fit evidence is fallible.

When supplied, local models compare linear/hinge (continuous) against step/step_hinge
(discontinuous). A larger jump_gain favors a discontinuity, after complexity penalties.
The search always supplies a few candidates even for ordinary noise, so rank is not a label.
Local median deltas and vector cancellation errors help pair edges but can be distorted by
nearby unrelated changes or ongoing movement. Do not treat them as a hard gate. If those
numbers are omitted, judge the raw images/text without inventing numerical evidence.

First locate abrupt changes; then compare possible onset/return PAIRS in the full trajectory,
including other independent changes. Report the first observed shifted sample and last
observed shifted sample BEFORE the return. Reject unsupported pairs; uncertain is allowed.
You may choose other dates from the displayed raw data. No fixed amplitude/speed threshold,
expected event count, hidden return, physical cause, or motion phases. Only transient ranges.

EVIDENCE INTERPRETATION AND NULL-HYPOTHESIS CHECK:
vector_cancellation is an ERROR: 0 means equal opposing changes, 1 means no cancellation.
A large error is evidence AGAINST a matched offset, never a high-confidence score.
Negative jump_gain means a continuous model explains the local data better; the search
still returns its top ranks in pure noise. A searched edge is NOT evidence that a jump exists.
Neither positive jump_gain nor a low cancellation error alone establishes a transient.
Before confirming EACH interval, actively compare four explanations against ORIGINAL data:
(1) ordinary station noise/seasonality, (2) a single permanent step plus subsequent noise,
(3) continuous physical movement followed by arrest, (4) a genuine temporary shift and return.
Inspect affected axes immediately outside AND inside both boundaries, and throughout the
interior. Isolated high/low samples do not establish a sustained displacement state. A local
noise wiggle does not become a return just because a large permanent step precedes it.
A slowly moving curve does not jump back when it reaches its new arrested plateau.
The normal reference shows this station's ordinary variability: compare morphology with it.
For weak events, a repeated shifted interior and the raw boundary neighborhoods can support
an offset even if one local fit favors linear: explore nearby dates and inspect the full
interval, rather than making numerical fit selection a hard veto. Independent permanent
steps can coexist inside a real transient; compare the origin and return on the SAME axes.
If the available evidence fails to distinguish explanation (4) from the alternatives, omit
that interval or mark uncertain. Do NOT produce an interval merely to acknowledge a proposal.
Your summary must state the strongest alternative considered. For each confirmed interval,
state why its two raw boundaries and interior favor a transient over that alternative.
"""


def records_for(phase):
    records = read_json(DATASET / 'manifest.json')['records']
    return [r for r in records if r['split'] == 'validation'
        and (r['within_cohort'] < 6) == (phase == 'pilot')]


def guard(phase):
    if phase == 'test':
        raise ValueError('independent test remains sealed')
    if phase == 'confirmation' and not read_json(Path('results/exp43-pilot-v1/admission.json'))['admitted']:
        raise ValueError('pilot did not meet gate')


def register(output, phase, stage, jobs):
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP43',
        'phase': phase, 'stage': stage, 'records': records_for(phase), 'jobs': jobs,
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in DEPENDENCIES}})


def baseline_root(phase):
    return Path('artifacts/exp41/pilot/baseline') if phase == 'pilot' else ROOT / phase / 'baseline'


def scan_root(phase):
    return Path('artifacts/exp41/pilot/scan') if phase == 'pilot' else ROOT / phase / 'scan'


def proposals(phase, station):
    base = read_json(baseline_root(phase) / 'predictions/vector_nearest' / station / 'result.json')
    segmented = read_json(baseline_root(phase) / 'predictions/segment_numeric' / station / 'result.json')
    scan = read_json(scan_root(phase) / 'predictions' / station / 'result.json')
    events = [e for e in base['prediction']['events'] if e['kind'] == 'transient']
    events.extend(parent.data.confirmed_events(scan, 'owv_image'))
    seen, candidates = set(), []
    for event in events:
        key = (event['start'], event['end'])
        if key not in seen:
            seen.add(key)
            candidates.append({'id': f'R{len(candidates) + 1}', 'kind': 'transient',
                'start': event['start'], 'end': event['end']})
    return base, segmented, scan, candidates


def search_edges(values, proposed, cache):
    observed = np.isfinite(values).all(axis=1)
    time, samples = np.flatnonzero(observed), values[observed]
    alternatives = []
    for day in range(max(365, proposed - 35), min(len(values), proposed + 36)):
        if not observed[day]:
            continue
        if day not in cache:
            fit = jump_evidence(time, samples, day, 365, len(values) - 1)
            if fit['selected'] == 'unsupported':
                cache[day] = None
            else:
                criteria = fit['criteria']
                gain = min(criteria['linear'], criteria['hinge']) - min(criteria['step'], criteria['step_hinge'])
                before = values[max(365, day - 7):day]
                after = values[day:min(len(values), day + 7)]
                before = before[np.isfinite(before).all(axis=1)]
                after = after[np.isfinite(after).all(axis=1)]
                delta = (np.median(after, axis=0) - np.median(before, axis=0)).tolist()
                cache[day] = {**fit, 'jump_gain': float(gain), 'local_median_change_NEU': delta}
        if cache[day] is not None:
            alternatives.append(cache[day])
    selected = []
    for fit in sorted(alternatives, key=lambda f: (-f['jump_gain'], abs(f['index'] - proposed), f['index'])):
        if all(abs(fit['index'] - old['index']) >= 7 for old in selected):
            selected.append(fit)
        if len(selected) == 3:
            break
    return selected


def explore(values, candidates):
    cache, evidence, extra_edges = {}, [], set()
    for candidate in candidates:
        onset = search_edges(values, candidate['start'], cache)
        returning = search_edges(values, candidate['end'] + 1, cache)
        pairs = []
        for first in onset:
            for second in returning:
                if second['index'] <= first['index'] + 1:
                    continue
                a = np.asarray(first['local_median_change_NEU'])
                b = np.asarray(second['local_median_change_NEU'])
                denominator = np.linalg.norm(a) + np.linalg.norm(b)
                pairs.append({'start': first['index'], 'end': second['index'] - 1,
                    'vector_cancellation': float(np.linalg.norm(a + b) / denominator) if denominator else None})
        evidence.append({'candidate': candidate['id'], 'onset_search': onset,
            'return_search': returning, 'pair_evidence': pairs})
        extra_edges.update(fit['index'] for fit in [*onset, *returning])
    return evidence, sorted(extra_edges)


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
        evidence, edges = explore(values, candidates)
        views = parent.local.local_views(values, candidates)
        for day in edges:
            left, right = max(365, day - 21), min(730, day + 22)
            local_values = values[left:right].copy()
            finite = local_values[np.isfinite(local_values).all(axis=1)]
            if len(finite):
                local_values -= np.median(finite[:7], axis=0)
            views.append({'id': 'searched_' + str(day), 'candidate_index': day,
                'left': left, 'right': right, 'values': local_values})
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True)
        overview = inputs / 'overview.png'
        parent.data.render_sheet({'original': values}, overview, 'nrv_raw_only')
        images = [overview]
        for start in range(0, len(views), 6):
            path = inputs / f'edges-{start // 6}.png'
            parent.local.render_local(views[start:start + 6], path)
            images.append(path)
        write_json(inputs / 'evidence.json', {'candidates': candidates, 'searched_edges': edges,
            'boundary_search_evidence': evidence})
        for mode in MODES:
            metadata = {'monitoring_indices': [365, 729], 'fallible_proposals': candidates,
                'searched_edge_indices': edges}
            if mode != 'bmv_no_evidence':
                metadata['boundary_search_evidence'] = evidence
            content = [{'type': 'text', 'text': json.dumps(metadata, allow_nan=False)}]
            if mode == 'bmv_text':
                text_views = [{'id': 'overview', 'left': 180, 'values': values[180:730]}, *views]
                for view in text_views:
                    content.append({'type': 'text', 'text': json.dumps({'view': view['id'],
                        'first_index': view['left'], 'samples_NEU': finite_rows(view['values']),
                        'median3_NEU': finite_rows(parent.data.median_line(view['values']))})})
            else:
                content.extend(image_block(path) for path in images)
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(parent.data.MODEL, PROMPT, parent.LocalDecision, content))
            jobs.append({'work': str(work), 'station': station, 'mode': mode,
                'request_sha256': sha256(work / 'request.json')})
        input_hashes.extend({'path': str(path), 'sha256': sha256(path)} for path in [
            scan_root(phase) / 'predictions' / station / 'result.json',
            baseline_root(phase) / 'predictions/vector_nearest' / station / 'result.json'])
        print('Prepared exploratory review', phase, index, '/', len(records_for(phase)), flush=True)
    register(output, phase, 'review', jobs)
    write_json(output / 'input-hashes.json', input_hashes)


def configure():
    parent.ROOT = ROOT
    parent.DATASET = DATASET
    parent.PROTOCOL = PROTOCOL
    parent.DEPENDENCIES = DEPENDENCIES
    parent.configure()
    parent.guard = guard
    parent.data.guard = guard
    parent.register = register
    parent.data.register = register
    parent.records_for = records_for
    parent.data.records_for = records_for
    parent.proposals = proposals


def report(phase):
    output = ROOT / phase / 'review'
    registration = read_json(output / 'registration.json')
    parent.verify_inputs(output, registration)
    if not (output / 'complete.json').exists():
        raise ValueError('finish all requests first')
    methods = [*parent.data.numerical.METHODS, 'vector_nearest', 'bmv_scan_only', *MODES]
    rows = {method: [] for method in methods}
    protected = True
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        base, segmented, scan, candidates = proposals(phase, station)
        predictions = {method: read_json(baseline_root(phase) / 'predictions' / method / station / 'result.json')
            for method in (*parent.data.numerical.METHODS, 'vector_nearest')}
        predictions['bmv_scan_only'] = parent.visual_owned(observation, base, scan)
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json')
            predictions[mode] = parent.visual_owned(observation, base, receipt)
            for kind in ('spike', 'step', 'deformation'):
                protected = protected and [e for e in predictions[mode]['prediction']['events']
                    if e['kind'] == kind] == [e for e in base['prediction']['events'] if e['kind'] == kind]
            write_json(ROOT / phase / 'final' / mode / station / 'result.json', predictions[mode])
        truth = read_json(DATASET / record['truth'])
        for method, result in predictions.items():
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'],
                **score_record(observation, truth, result)})
    methods = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): parent.data.aggregate_types([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    receipts = request_receipts([job['work'] for job in registration['jobs']])
    if phase != 'pilot':
        receipts.extend(request_receipts([job['work'] for job in read_json(scan_root(phase) / 'registration.json')['jobs']]))
    summary = {'experiment_id': 'GNSS-VLM-EXP43-' + phase.upper(), 'stations': len(records_for(phase)),
        'methods': methods, 'costs': aggregate_cost(receipts), 'reused_scan_requests': 18 if phase == 'pilot' else 0}
    target = Path(f'results/exp43-{phase}-v1')
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'summary.json', summary)
    write_json(target / 'per-type.json', parent.data.typed_rows(summary))
    write_json(target / 'strata.json', strata)
    write_json(target / 'request-receipts.json', receipts)

    def aff(scores):
        return scores['range']['transient']['affiliation']['station_macro_f1']

    def count(scores, key):
        return scores['original']['types']['transient']['daily_micro'][key]

    main, text, plain = (methods[mode] for mode in MODES)
    seed_deltas = {seed: aff(strata['cohort_seed']['bmv_image'][seed]) - aff(strata['cohort_seed']['bmv_text'][seed])
        for seed in strata['cohort_seed']['bmv_image']}
    checks = {'aff_at_least_080': aff(main) >= .80,
        'visual_gain': aff(main) - aff(text) >= .02 or (aff(main) - aff(text) >= -.01
            and count(text, 'fp') > 0 and count(main, 'fp') <= .8 * count(text, 'fp')),
        'coverage_preserved': count(main, 'tp') >= .9 * count(plain, 'tp'),
        'other_types_preserved': protected,
        'negative_fp_halved': main['range']['transient']['negative_station_fp_days']
            <= .5 * methods['bmv_scan_only']['range']['transient']['negative_station_fp_days'],
        'two_seed_directions_nonnegative': phase == 'pilot' or sum(v >= 0 for v in seed_deltas.values()) >= 2}
    if phase == 'pilot':
        old = read_json(Path('results/exp41-pilot-v1/summary.json'))['methods']['bmv_image']
        checks['aff_gain_over_exp41'] = aff(main) - aff(old) >= .08
        checks['tp_gain_over_exp41'] = count(main, 'tp') > count(old, 'tp')
    write_json(target / 'admission.json', {'admitted': all(checks.values()), 'checks': checks,
        'seed_aff_deltas': seed_deltas})
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'costs': summary['costs'], 'checks': checks}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare-scan', 'infer-scan',
        'prepare-review', 'infer-review', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    configure()
    if args.stage == 'baseline':
        if args.phase == 'pilot':
            raise ValueError('pilot reuses registered EXP41 baseline')
        parent.data.baseline(args.phase)
    elif args.stage == 'prepare-scan':
        if args.phase == 'pilot':
            raise ValueError('pilot reuses recorded scan, no replay')
        parent.prepare_scan(args.phase)
    elif args.stage == 'prepare-review':
        prepare_review(args.phase)
    elif args.stage.startswith('infer-'):
        parent.infer(args.phase, args.stage.removeprefix('infer-'))
    else:
        report(args.phase)
