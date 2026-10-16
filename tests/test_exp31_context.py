"""Preserve event surroundings without mixing used development stations into a new run."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import exp31_data_baseline as data  # noqa: E402
from experiment_exp31 import merged_contexts  # noqa: E402


def test_two_views_of_the_same_transition_become_one_complete_context():
    observation = {'background_end': 364, 'dates': list(range(730))}
    regions = [{'start': 615, 'end': 625}, {'start': 565, 'end': 575}]
    assert merged_contexts(regions, observation) == [[530, 661]]


def test_context_union_preserves_real_gaps_and_clips_monitoring_boundaries():
    observation = {'background_end': 364, 'dates': list(range(730))}
    regions = [{'start': 400, 'end': 410}, {'start': 720, 'end': 729}]
    assert merged_contexts(regions, observation) == [[365, 446], [685, 730]]
    adjacent = [{'start': 400, 'end': 410}, {'start': 481, 'end': 491}]
    assert merged_contexts(adjacent, observation) == [[365, 527]]


def test_new_development_partition_excludes_all_previous_pilot_stations(monkeypatch):
    records = [{'station_id': f'{prefix}{index:04d}', 'split': split}
               for prefix, split in (('v', 'validation'), ('e', 'test'))
               for index in range(360)]
    monkeypatch.setattr(data, 'read_json', lambda path: {'records': records})
    pilot = {record['station_id'] for record in data.records_for('validation', 'pilot')}
    remaining = {record['station_id'] for record in data.records_for('validation', 'remaining')}
    full = {record['station_id'] for record in data.records_for('validation', 'full')}
    old = {f'v{index:04d}' for index in range(360) if index % 120 < 24}
    assert len(pilot) == 36
    assert len(remaining) == 252
    assert len(full) == 288
    assert pilot.isdisjoint(remaining)
    assert full == pilot | remaining
    assert full.isdisjoint(old)
    assert len(data.records_for('test', 'pilot')) == 36
    assert len(data.records_for('test', 'remaining')) == 324
