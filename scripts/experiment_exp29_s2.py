"""Visual localization, numerical evidence, then matched visual/text adjudication."""
from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from exp14_fingerprint import PairDecision
from exp29_data_baseline import DATASET, ROOT
from experiment_exp25 import settings
from experiment_exp26 import draw_panels, finite_rows
from experiment_exp29 import compose, image_block, payload, phase_records, transient_event
from pydantic import BaseModel, ConfigDict

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.provider import send_request

OUT = ROOT / 'validation-adjudication-v1'
JOINT = ROOT / 'validation-joint-v1'
PROTOCOL = Path('docs/experiment-exp29-s2.md')
MODES = {'vnv_image': ('image', 'image'), 'vnv_text_final': ('image', 'text'),
         'vnv_text_pipeline': ('text', 'text')}
PROMPT = """Adjudicate temporary GNSS observation offsets among proposed P IDs.
Return {"decision":"keep_numeric" or "replace_ranges","pair_ids":[],"evidence":"observations"}.
All candidates are fallible. No label identifies the correct source. Numerical information
criteria are evidence, NOT binding verdicts. Smooth physical motion, separate permanent
steps and isolated spikes can coexist with a real temporary offset. Preserve physical motion.
Choose a COMPLETE nonoverlapping set of genuine temporary platforms in this local review.
Use replace_ranges to revise the current reviewed temporary interpretation, including an
empty set if none is a genuine platform. Use keep_numeric if evidence is insufficient.

First inspect ORIGINAL N/E/U throughout the interval and surrounding context. A temporary
offset has a sustained shift followed by an opposing return across the same affected axes;
one axis can be weak. A smooth bend or isolated point plus unrelated step is not a platform.
An independent permanent step can lie inside, before or after the temporary range. Therefore
outside absolute levels need not agree. Do not automatically pair nearest/earliest edges.

Original context and hypothetical onset-offset subtraction may be provided as images or
equivalent point arrays. The onset estimate is the following 7-day median minus preceding
7-day median on original station-scaled observations; return never fitted. Such estimates
can be distorted by short intervals, nearby steps or motion. A correction is a diagnostic
hypothesis, not truth. Compare BOTH boundaries and persistent interior observations; do not
flatten physical motion or demand perfect numerical closure. Missing samples stay absent.
Candidate dates may be approximate or refined. Prefer boundaries supported by observations,
not the highest score, source tag or longest span. No universal amplitude/speed cutoff.
Only supplied pair IDs, no shared endpoints or overlapping temporary intervals. Other
quality events and motion are handled independently. Keep evidence concise and observational.
"""


class Adjudication(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    decision: Literal['keep_numeric', 'replace_ranges']
    pair_ids: list[str]
    evidence: str


def make_candidates(station, family, strong):
    first = read_json(ROOT / 'validation-visual-v1/predictions' / ('vf_vl_' + family) / station / 'result.json')
    proposals = first.get('range_decision', {}).get('intervals', [])
    tool = read_json(JOINT / ('joint_vl_' + family) / station / 'tool-evidence.json')
    bounds = [(v['start'], v['end']) for v in proposals]
    bounds.extend((r['best']['first'], r['best']['second'] - 1)
                  for r in tool['receipts'] if r['best'] is not None)
    reviewed = [e for e in strong['prediction']['events'] if e['kind'] == 'transient'
        and any(e['start'] <= v['end'] + 30 and e['end'] >= v['start'] - 30 for v in proposals)]
    bounds.extend((e['start'], e['end']) for e in reviewed)
    pairs = [{'id': f'P{i + 1}', 'start': a, 'end': b}
             for i, (a, b) in enumerate(sorted(set(bounds)))]
    return {'pairs': pairs, 'reviewed_existing': reviewed,
            'numerical_tool_receipts': tool['receipts'], 'first_stage_status': first['status']}


def prepare(phase):
    if phase == 'pilot':
        OUT.mkdir(parents=True, exist_ok=False)
    else:
        if not (OUT / 'pilot-complete.json').exists():
            raise ValueError('complete pilot first')
        check(read_json(OUT / 'pilot-registration.json'))
    registration_path = OUT / (phase + '-registration.json')
    if registration_path.exists():
        raise ValueError('phase already prepared')
    jobs, records = [], phase_records('validation', phase)
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        strong = read_json(ROOT / 'validation-visual-v1/final/vector_nearest' / station / 'result.json')
        evidence = numerical_evidence(observation)
        for family in ('image', 'text'):
            context = make_candidates(station, family, strong)
            inputs = OUT / 'inputs' / family / station
            inputs.mkdir(parents=True)
            write_json(inputs / 'candidates.json', context)
            content = []
            if context['pairs']:
                edges = sorted({p[key] + int(key == 'end') for p in context['pairs'] for key in ('start', 'end')})
                graph = {'jumps': [{'id': f'J{i + 1}', 'index': day} for i, day in enumerate(edges)],
                         'pairs': context['pairs']}
                left = max(observation['background_end'] + 1, min(edges) - 30)
                right = min(len(evidence.values), max(edges) + 31)
                local_vectors = {}
                raw = evidence.values / evidence.noise
                for pair in context['pairs']:
                    day = pair['start']
                    offset = np.nanmedian(raw[day:min(day + 7, len(raw))], axis=0) - np.nanmedian(raw[max(0, day - 7):day], axis=0)
                    local_vectors[pair['id']] = offset.tolist()
                    draw_panels(observation, graph, inputs / (pair['id'] + '.png'), pair, offset)
                draw_panels(observation, graph, inputs / 'original.png')
                metadata = {'pairs': context['pairs'], 'onset_offsets': local_vectors,
                    'numerical_tool_evidence': context['numerical_tool_receipts'],
                    'missing_indices': np.flatnonzero(~evidence.observed).tolist()}
                content = [{'type': 'text', 'text': json.dumps(metadata)}]
                text = {'first_index': left, 'end_index_exclusive': right,
                    'original_station_scaled': finite_rows(raw[left:right]),
                    'normal_history_residual': finite_rows((evidence.residual / evidence.noise)[left:right])}
            for mode, (first_family, final_view) in MODES.items():
                if first_family != family:
                    continue
                work = OUT / 'predictions' / mode / station
                work.mkdir(parents=True)
                write_json(work / 'candidates.json', context)
                if not context['pairs']:
                    write_json(work / 'result.json', {'status': 'not_requested', 'request_count': 0,
                        'adjudication': {'decision': 'keep_numeric', 'pair_ids': [], 'evidence': 'No first-stage ranges'}})
                    continue
                own = copy.deepcopy(content)
                if final_view == 'image':
                    own.append(image_block(inputs / 'original.png'))
                    own.extend(image_block(inputs / (p['id'] + '.png')) for p in context['pairs'])
                else:
                    own.append({'type': 'text', 'text': json.dumps(text, allow_nan=False)})
                write_json(work / 'request.json', payload('qwen3-vl-plus', PROMPT, Adjudication, own))
                jobs.append({'mode': mode, 'station': station, 'request_sha256': sha256(work / 'request.json')})
    write_json(registration_path, {'experiment_id': 'GNSS-VLM-EXP29-S2', 'phase': phase,
        'records': records, 'jobs': jobs, 'frozen_files': {str(p): sha256(p)
            for p in (Path(__file__), PROTOCOL, Path('scripts/exp29_joint_boundaries.py'))}})
    print('Prepared S2', phase, len(records), 'stations', len(jobs), 'requests', flush=True)


def check(registration):
    for file, digest in registration['frozen_files'].items():
        if sha256(Path(file)) != digest:
            raise ValueError('S2 source changed')


def validate_decision(decision, candidates):
    PairDecision(pair_ids=decision.pair_ids, evidence=decision.evidence)
    mapping = {p['id']: p for p in candidates['pairs']}
    if len(set(decision.pair_ids)) != len(decision.pair_ids) or not set(decision.pair_ids) <= mapping.keys():
        raise ValueError('unknown or duplicated candidates')
    selected = sorted((mapping[key] for key in decision.pair_ids), key=lambda p: p['start'])
    if any(a['end'] + 1 >= b['start'] for a, b in zip(selected, selected[1:])):
        raise ValueError('overlapping ranges or shared endpoints')


def infer(phase):
    registration = read_json(OUT / (phase + '-registration.json'))
    check(registration)
    write_json(OUT / (phase + '-started.json'), {'requests': len(registration['jobs']), 'retries': 0})
    def execute(job):
        work = OUT / 'predictions' / job['mode'] / job['station']
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed')
        receipt, content = send_request(read_json(work / 'request.json'), work, settings('qwen3-vl-plus'))
        if receipt['status'] == 'success':
            try:
                decision = Adjudication.model_validate_json(content)
                validate_decision(decision, read_json(work / 'candidates.json'))
                receipt['adjudication'] = decision.model_dump()
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']
    with ThreadPoolExecutor(max_workers=3) as executor:
        for number, result in enumerate(executor.map(execute, registration['jobs']), 1):
            print(number, '/', len(registration['jobs']), *result, flush=True)
    write_json(OUT / (phase + '-complete.json'), {'requests': len(registration['jobs'])})


def apply_decision(observation, numeric, strong, candidates, receipt):
    decision = receipt.get('adjudication')
    if receipt['status'] not in ('success', 'not_requested') or not decision or decision['decision'] == 'keep_numeric':
        return compose(observation, strong, [])
    reviewed = {(e['start'], e['end']) for e in candidates['reviewed_existing']}
    events = [copy.deepcopy(e) for e in strong['prediction']['events']
              if e['kind'] != 'transient' or (e['start'], e['end']) not in reviewed]
    restore = {day for first, last in reviewed for day in (first, last + 1)}
    known = {j['index'] for j in numeric['jumps']}
    present = {e['start'] for e in events if e['kind'] == 'step'}
    for day in sorted((restore & known) - present):
        events.append({**transient_event(day, day, 'Restored numerical step after reviewed pair removal'), 'kind': 'step'})
    mapping = {p['id']: p for p in candidates['pairs']}
    selected = [transient_event(mapping[key]['start'], mapping[key]['end'], decision['evidence'])
                for key in decision['pair_ids']]
    return compose(observation, {'prediction': {'events': events}}, selected)


def materialize(phase):
    registration = read_json(OUT / (phase + '-registration.json'))
    check(registration)
    if not (OUT / (phase + '-complete.json')).exists():
        raise ValueError('inference incomplete')
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        strong = read_json(ROOT / 'validation-visual-v1/final/vector_nearest' / station / 'result.json')
        numeric = read_json(ROOT / 'validation-baseline-v1/predictions/segment_numeric' / station / 'result.json')
        for mode in MODES:
            work = OUT / 'predictions' / mode / station
            receipt = read_json(work / 'result.json')
            result = apply_decision(observation, numeric, strong, read_json(work / 'candidates.json'), receipt)
            result['visual_dependency_status'] = receipt['status']
            write_json(OUT / 'final' / mode / station / 'result.json', result)
    write_json(OUT / (phase + '-materialized.json'), {'stations': len(registration['records'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'materialize'))
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'materialize': materialize}[args.stage](args.phase)
