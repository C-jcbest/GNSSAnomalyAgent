import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp29 import transient_event  # noqa: E402
from experiment_exp29_s2 import apply_decision  # noqa: E402


def test_revised_pair_restores_unpaired_step_and_keeps_physical_activity():
    observation = {'dates': list(range(300)), 'background_end': 89,
        'displacement_mm': np.random.default_rng(3).normal(size=(300, 3)).tolist()}
    old = transient_event(130, 159, 'Original numerical pairing')
    activity = {**transient_event(140, 240, 'Physical displacement'), 'kind': 'deformation'}
    strong = {'prediction': {'events': [old, activity]}}
    numeric = {'jumps': [{'index': day} for day in (130, 160, 180)]}
    candidates = {'reviewed_existing': [old], 'pairs': [{'id': 'P2', 'start': 160, 'end': 179}]}
    receipt = {'status': 'success', 'adjudication': {'decision': 'replace_ranges',
               'pair_ids': ['P2'], 'evidence': 'Observed platform returns at 180'}}
    result = apply_decision(observation, numeric, strong, candidates, receipt)
    signatures = {(e['kind'], e['start'], e['end']) for e in result['prediction']['events']}
    assert signatures == {('step', 130, 130), ('transient', 160, 179), ('deformation', 140, 240)}
    assert strong['prediction']['events'] == [old, activity]


def test_failed_adjudication_preserves_original_numeric_decision():
    observation = {'dates': list(range(300)), 'background_end': 89,
        'displacement_mm': np.random.default_rng(3).normal(size=(300, 3)).tolist()}
    old = transient_event(130, 159, 'Original pairing')
    result = apply_decision(observation, {'jumps': []}, {'prediction': {'events': [old]}},
        {'reviewed_existing': [old], 'pairs': []}, {'status': 'failed'})
    assert result['prediction']['events'] == [old]
