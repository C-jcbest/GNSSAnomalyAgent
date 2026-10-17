"""Check frozen EXP45 evidence and event ownership without making model requests."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import experiment_exp45 as experiment

from gnss_research.files import read_json, sha256, write_json


def verify():
    checks = []

    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})

    all_ids, counts = [], {}
    for phase in ('pilot', 'confirmation'):
        output = experiment.ROOT / phase / 'review'
        registration = read_json(output / 'registration.json')
        experiment.verify_inputs(output, registration)
        check(phase + ':frozen_sources_observations_numeric_inputs', True)
        correction = read_json(output / 'evaluation-correction.json')
        check(phase + ':evaluator_correction_hash',
            correction['script_sha256'] == sha256(Path('scripts/report_exp45.py')))
        for record in registration['records']:
            check(phase + ':truth:' + record['station_id'],
                sha256(experiment.DATASET / record['truth']) == record['truth_sha256'])
        check(phase + ':validation_only', all(r['split'] == 'validation' for r in registration['records']))
        all_ids.append({r['station_id'] for r in registration['records']})
        statuses = []
        for job in registration['jobs']:
            work = Path(job['work'])
            check(phase + ':request:' + job['station'], sha256(work / 'request.json') == job['request_sha256'])
            receipt = read_json(work / 'result.json')
            statuses.append(receipt['status'])
            check(phase + ':no_retry:' + job['station'],
                receipt['request_count'] == 1 and receipt['retry_count'] == 0)
        target = Path(f'results/exp45-{phase}-v1')
        rows = json.loads(gzip.decompress((target / 'per-station.json.gz').read_bytes()))
        check(phase + ':all_stations_scored', all(len(values) == len(registration['records']) for values in rows.values()))
        numeric = {row['id']: row for row in rows['segment_numeric']}
        for mode in experiment.MODES:
            for row in rows[mode]:
                reference = numeric[row['id']]['events']['events']
                actual = row['events']['events']
                check(phase + ':' + mode + ':other_types:' + row['id'],
                    [e for e in actual if e['kind'] != 'deformation'] ==
                    [e for e in reference if e['kind'] != 'deformation'])
        audit = read_json(target / 'boundary-audit.json')
        for item in audit:
            for choice in item['boundary_choices']:
                check(phase + ':fit_preserved:' + item['station'],
                    choice['selected_gain'] >= choice['numeric_gain'])
                if item['visual_status'] != 'success':
                    check(phase + ':failed_preserves_numeric:' + item['station'], not choice['changed'])
        counts[phase] = {'stations': len(registration['records']),
            'requests': len(statuses), 'success': statuses.count('success'),
            'failed': statuses.count('failed'),
            'routed_stations': sum(r['visual_requested'] for r in registration['routes']),
            'admission': read_json(target / 'admission.json'),
            'modes': {mode: {
                'confirmed_visual_stations': sum(bool(item['visual_alternatives']) for item in audit if item['mode'] == mode),
                'changed_stations': [item['station'] for item in audit if item['mode'] == mode and
                    any(choice['changed'] for choice in item['boundary_choices'])]}
                for mode in experiment.MODES}}
    check('pilot_confirmation_disjoint', not (all_ids[0] & all_ids[1]))
    training = read_json(experiment.numerical.TRAINING / 'training.json')
    check('frozen_training_model', training['model_sha256'] == sha256(experiment.numerical.TRAINING / 'models.joblib'))
    result = {'passed': all(item['passed'] for item in checks), 'check_count': len(checks),
        'counts': counts, 'checks': checks, 'new_requests': 0}
    write_json(Path('docs/exp45-verification.json'), result)
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, indent=2))
    if not result['passed']:
        raise ValueError('EXP45 evidence verification failed')


if __name__ == '__main__':
    verify()
