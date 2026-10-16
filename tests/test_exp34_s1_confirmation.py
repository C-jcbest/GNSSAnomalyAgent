"""A secondary result must be checked on newly reserved stations and endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_exp34_s1 as experiment  # noqa: E402
from report_exp34_s1 import admission  # noqa: E402


def test_image_confirmation_reserves_new_development_stations(monkeypatch):
    records = [{'station_id': f'{prefix}{index:04d}', 'split': split}
        for prefix, split in (('v', 'validation'), ('e', 'test')) for index in range(360)]
    monkeypatch.setattr(experiment, 'read_json', lambda path: {'records': records})
    pilot = {row['station_id'] for row in experiment.records_for('pilot')}
    remaining = {row['station_id'] for row in experiment.records_for('remaining')}
    assert len(pilot) == 36
    assert len(remaining) == 72
    assert not pilot & remaining
    assert all(station.startswith('v') and int(station[1:]) % 120 >= 84 for station in pilot | remaining)
    assert pilot | remaining == {row['station_id'] for row in experiment.records_for('full')}


def test_full_confirmation_requires_complete_no_image_control_as_well():
    table = {
        'jcv_image_pure': {'transient_aff_f1': .7, 'transient_daily_f1': .6},
        'jcv_text_final_pure': {'transient_aff_f1': .3, 'transient_daily_f1': .4},
        'jcv_text_pipeline_pure': {'transient_aff_f1': .8, 'transient_daily_f1': .7},
        'jcv_image': {'transient_aff_f1': .8, 'transient_daily_f1': .8, 'four_type_daily_f1': .7},
        'vector_nearest': {'transient_aff_f1': .8, 'transient_daily_f1': .9, 'four_type_daily_f1': .8},
    }
    seeds = {method: {str(seed): {'range': {'transient': {'affiliation': {
        'station_macro_f1': row['transient_aff_f1']}}}} for seed in range(3)} for method, row in table.items()}
    paired = {'image_final_vs_text_final': {'tp_delta': 20},
        'image_final_vs_full_text': {'tp_delta': -10}, 'fusion_vs_nearest': {'tp_delta': 0}}
    strata = {'cohort_seed': seeds}
    assert admission(table, strata, paired, 'pilot')['mechanism_confirmed'] is True
    result = admission(table, strata, paired, 'full')
    assert result['mechanism_confirmed'] is False
    assert result['mechanism_checks']['full_text_actual_tp_gain'] is False
