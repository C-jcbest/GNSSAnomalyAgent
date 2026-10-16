"""Preserve unseen development data for visual-selection confirmation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_exp34 as experiment  # noqa: E402


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
