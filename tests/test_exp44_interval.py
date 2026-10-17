"""Interval model competition must preserve an independent internal permanent step."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp44 import refine  # noqa: E402


def test_interval_model_recovers_offset_with_an_independent_step_inside():
    values = np.random.default_rng(44).normal(0, .2, (730, 3))
    values[430:470] += [8, -3, 2]
    values[450:] += [-5, 4, 1]
    result = refine(values, {'start': 433, 'end': 468}, [430, 450, 470])
    assert result['accepted']
    assert (result['start'], result['end']) == (430, 469)


def test_interval_model_rejects_permanent_step_with_no_opposing_return():
    values = np.random.default_rng(45).normal(0, .2, (730, 3))
    values[430:] += [8, -3, 2]
    result = refine(values, {'start': 430, 'end': 469}, [430])
    assert not result['accepted']
