"""Preserve unseen development data for visual-selection confirmation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_exp34 as experiment  # noqa: E402
from report_exp34 import admission  # noqa: E402


def test_new_cohort_excludes_every_previously_evaluated_station(monkeypatch):
    records = [{'station_id': f'{prefix}{index:04d}', 'split': split}
        for prefix, split in (('v', 'validation'), ('e', 'test')) for index in range(360)]
    monkeypatch.setattr(experiment, 'read_json', lambda path: {'records': records})
    pilot = {record['station_id'] for record in experiment.records_for('pilot')}
    remaining = {record['station_id'] for record in experiment.records_for('remaining')}
    old = {f'v{index:04d}' for index in range(360) if index % 120 < 60}
    assert len(pilot) == 72
    assert len(remaining) == 108
    assert not pilot & remaining
    assert not (pilot | remaining) & old
    assert pilot | remaining == {record['station_id'] for record in experiment.records_for('full')}
    assert all(station.startswith('v') for station in pilot | remaining)


def visual_selection_inputs():
    table = {
        'jcv_text_final_pure': {'transient_aff_f1': .7, 'transient_daily_f1': .6},
        'jcv_text_pipeline_pure': {'transient_aff_f1': .3, 'transient_daily_f1': .4},
        'jcv_text_final': {'transient_aff_f1': .8, 'transient_daily_f1': .8, 'four_type_daily_f1': .7},
        'vector_nearest': {'transient_aff_f1': .8, 'transient_daily_f1': .9, 'four_type_daily_f1': .8},
    }
    seeds = {method: {str(seed): {'range': {'transient': {'affiliation': {
        'station_macro_f1': row['transient_aff_f1']}}}} for seed in range(3)}
        for method, row in table.items()}
    paired = {'visual_screen_vs_text_screen': {'tp_delta': 30}, 'fusion_vs_nearest': {'tp_delta': 0}}
    return table, {'cohort_seed': seeds}, paired


def test_visual_selection_gain_is_distinct_from_fusion_admission():
    result = admission(*visual_selection_inputs())
    assert result['mechanism_confirmed'] is True
    assert result['stable_system_admitted'] is False


def test_range_gain_with_no_real_coverage_gain_does_not_admit_expansion():
    table, strata, paired = visual_selection_inputs()
    paired['visual_screen_vs_text_screen']['tp_delta'] = 0
    assert admission(table, strata, paired)['mechanism_confirmed'] is False
