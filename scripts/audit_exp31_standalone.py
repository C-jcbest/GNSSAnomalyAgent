"""Exploratory attribution of model-only outputs; never change the main admission gate."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from report_exp25 import bootstrap

from gnss_research.files import read_json, write_json


def audit(report):
    with gzip.open(report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    main = 'jcv_image_pure'
    controls = ('jcv_text_final_pure', 'jcv_text_pipeline_pure', 'jcv_global_only_pure',
                'jcv_local_only_pure', 'jcv_image_no_numeric_pure', 'air_image_pure')
    comparisons = {}
    for control in controls:
        changes = []
        for primary, secondary in zip(rows[main], rows[control], strict=True):
            a, b = primary['range']['transient'], secondary['range']['transient']
            aff_a, aff_b = a['affiliation']['f1'], b['affiliation']['f1']
            changes.append({'station': primary['id'], 'cohort_seed': primary['cohort_seed'],
                'scenario': primary['scenario'], 'tp_delta': a['daily']['tp'] - b['daily']['tp'],
                'fp_delta': a['daily']['fp'] - b['daily']['fp'],
                'aff_delta': aff_a - aff_b if aff_a is not None and aff_b is not None else None})
        comparisons[control] = {'tp_delta': sum(change['tp_delta'] for change in changes),
            'fp_delta': sum(change['fp_delta'] for change in changes),
            'tp_improved_stations': sum(change['tp_delta'] > 0 for change in changes),
            'tp_harmed_stations': sum(change['tp_delta'] < 0 for change in changes),
            'fp_increased_stations': sum(change['fp_delta'] > 0 for change in changes),
            'bootstrap': bootstrap(rows[main], rows[control]), 'changes': changes}
    numerical = read_json(report / 'summary.json')['methods']['vector_nearest']['range']['transient']['daily_micro']
    result = {'experiment_id': 'GNSS-VLM-EXP31', 'main_admission_unchanged': True,
        'scope': 'Exploratory comparison of preregistered model-only readouts after a failed main pilot; not a new success endpoint or independent confirmation.',
        'terminology': 'Pure means no numerical final predictions fused in; the model still receives numerical jump evidence except in no_numeric.',
        'strong_baseline_daily_counts': numerical, 'comparisons': comparisons, 'new_model_requests': 0}
    write_json(report / 'standalone-audit.json', result)
    print({control: {key: value[key] for key in ('tp_delta', 'fp_delta', 'tp_improved_stations', 'tp_harmed_stations')}
           for control, value in comparisons.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    audit(parser.parse_args().report)
