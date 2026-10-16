"""The confirmation cohort excludes every previously used development station."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_exp32 as experiment  # noqa: E402
from report_exp32 import admission  # noqa: E402


def test_confirmation_partitions_are_disjoint_and_do_not_use_old_or_test_stations(monkeypatch):
    records = [{'station_id': f'{prefix}{index:04d}', 'split': split}
        for prefix, split in (('v', 'validation'), ('e', 'test')) for index in range(360)]
    monkeypatch.setattr(experiment, 'read_json', lambda path: {'records': records})
    pilot = {record['station_id'] for record in experiment.records_for('pilot')}
    remaining = {record['station_id'] for record in experiment.records_for('remaining')}
    old = {f'v{index:04d}' for index in range(360) if index % 120 < 36}
    assert len(pilot) == 72
    assert len(remaining) == 180
    assert not pilot & remaining
    assert not (pilot | remaining) & old
    assert pilot | remaining == {record['station_id'] for record in experiment.records_for('full')}
    assert all(station.startswith('v') for station in pilot | remaining)


def test_failed_mechanism_gate_blocks_expansion(monkeypatch):
    monkeypatch.setattr(experiment, 'read_json', lambda path: {'mechanism_confirmed': False})
    with pytest.raises(ValueError, match='do not expand'):
        experiment.guard_phase('remaining')


def confirmation_inputs():
    table = {
        'jcv_image_pure': {'transient_aff_f1': .55, 'transient_daily_f1': .60},
        'jcv_text_final_pure': {'transient_aff_f1': .50, 'transient_daily_f1': .59},
        'jcv_image': {'transient_aff_f1': .91, 'transient_daily_f1': .89, 'four_type_daily_f1': .80},
        'vector_nearest': {'transient_aff_f1': .91, 'transient_daily_f1': .92, 'four_type_daily_f1': .83},
    }
    seeds = {method: {str(seed): {'range': {'transient': {'affiliation': {
        'station_macro_f1': row['transient_aff_f1']}}}} for seed in range(3)}
        for method, row in table.items()}
    paired = {'model_vs_text': {'tp_delta': 10}, 'fusion_vs_nearest': {'tp_delta': 0}}
    return table, {'cohort_seed': seeds}, paired


def test_model_mechanism_confirmation_does_not_admit_a_degrading_fusion_system():
    result = admission(*confirmation_inputs(), 'pilot')
    assert result['mechanism_confirmed'] is True
    assert result['stable_system_admitted'] is False


def test_affiliation_gain_without_real_tp_gain_does_not_pass_confirmation():
    table, strata, paired = confirmation_inputs()
    paired['model_vs_text']['tp_delta'] = 0
    result = admission(table, strata, paired, 'pilot')
    assert result['mechanism_confirmed'] is False
    assert result['mechanism_checks']['aff_gain_at_least_003'] is True
    assert result['mechanism_checks']['actual_tp_gain'] is False
