"""Locate development bottlenecks without changing predictions or calling a model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from exp29_data_baseline import DATASET, ROOT

from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import Prediction, event_masks, observed_values


def transient_mask(prediction, observed):
    return event_masks(Prediction.model_validate(prediction), observed)['transient']


def audit_development():
    records = [r for r in read_json(DATASET / 'manifest.json')['records']
               if r['split'] == 'validation']
    totals = Counter()
    durations = Counter()
    evidence_texts = Counter()
    rows = []
    image_counts = []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = transient_mask({'events': truth['events']}, observed)
        visual = ROOT / 'validation-visual-v1'
        receipt = read_json(visual / 'predictions/vf_vl_image' / station / 'result.json')
        intervals = receipt.get('range_decision', {}).get('intervals', [])
        strong = read_json(visual / 'final/vector_nearest' / station / 'result.json')
        strong_mask = transient_mask(strong['prediction'], observed)
        tools = read_json(visual / 'predictions/vf_vl_image' / station / 'tool-evidence.json')
        details = []
        totals['stations'] += 1
        totals['positive_stations'] += int(target.any())
        totals['positive_without_proposal'] += int(target.any() and not intervals)
        for interval, tool in zip(intervals, tools['receipts'], strict=True):
            start, end = interval['start'], interval['end']
            duration = end - start + 1
            durations[duration] += 1
            proposed = np.zeros(len(observed), dtype=bool)
            proposed[start:end + 1] = True
            proposed &= observed
            added = proposed & ~strong_mask
            raw_tp = int((proposed & target).sum())
            new_tp = int((added & target).sum())
            new_fp = int((added & ~target).sum())
            blocked = bool((proposed & strong_mask).any())
            # compose() tests calendar overlap, including unobserved days.
            calendar_blocked = any(e['kind'] == 'transient' and e['start'] <= end
                and start <= e['end'] for e in strong['prediction']['events'])
            totals['proposals'] += 1
            totals['proposals_with_true_day'] += int(raw_tp > 0)
            totals['one_day_proposals'] += int(duration == 1)
            totals['three_or_fewer_day_proposals'] += int(duration <= 3)
            totals['proposals_blocked_by_existing_range'] += int(calendar_blocked)
            totals['blocked_proposals_with_uncovered_true_days'] += int(calendar_blocked and new_tp > 0)
            totals['unblocked_one_day_proposals'] += int(not calendar_blocked and duration == 1)
            totals['unblocked_one_day_true_days'] += new_tp if not calendar_blocked and duration == 1 else 0
            totals['unblocked_one_day_false_days'] += new_fp if not calendar_blocked and duration == 1 else 0
            edges = tool['refined_edges']
            if len(edges) == 2 and edges[0] >= edges[1]:
                totals['collapsed_or_reversed_refinements'] += 1
                totals['collapsed_proposals_with_true_day'] += int(raw_tp > 0)
            details.append({'interval': interval, 'duration': duration, 'raw_tp_days': raw_tp,
                'uncovered_true_days': new_tp, 'uncovered_false_days': new_fp,
                'calendar_overlap_blocks_addition': calendar_blocked,
                'observed_overlap': blocked, 'refined_edges': edges,
                'hard_verification_accepted': tool['accepted']})
        if receipt['status'] == 'success':
            evidence_texts[receipt['range_decision']['evidence']] += 1
        final_work = ROOT / 'validation-adjudication-v1/predictions/vnv_image' / station
        final = read_json(final_work / 'result.json')
        totals['positive_without_second_stage_request'] += int(target.any() and final['status'] == 'not_requested')
        if (final_work / 'request.json').exists():
            request = read_json(final_work / 'request.json')
            image_count = sum(block['type'] == 'image_url'
                for message in request['messages'] if isinstance(message['content'], list)
                for block in message['content'])
            image_counts.append(image_count)
        rows.append({'station': station, 'positive': bool(target.any()),
            'first_stage_status': receipt['status'], 'proposals': details,
            'second_stage_status': final['status']})
    return {'experiment_id': 'EXP29-optimization-audit-v1',
        'scope': 'Posthoc development-only diagnosis; no new predictions or causal attribution.',
        'new_model_requests': 0, 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'totals': dict(totals), 'proposal_duration_histogram': dict(sorted(durations.items())),
        'literal_short_observations_evidence_count': evidence_texts['short observations'],
        'successful_first_stage_requests': sum(evidence_texts.values()),
        'second_stage_images_per_request': {'requests': len(image_counts),
            'median': float(np.median(image_counts)), 'maximum': max(image_counts),
            'histogram': dict(sorted(Counter(image_counts).items()))}, 'stations': rows}


if __name__ == '__main__':
    output = Path('results/exp29-optimization-audit-v1')
    output.mkdir(parents=True, exist_ok=False)
    result = audit_development()
    write_json(output / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'stations'}, ensure_ascii=False, indent=2))
