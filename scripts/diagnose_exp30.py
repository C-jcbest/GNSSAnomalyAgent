"""Attribute completed AIR results to scanning, confirmation and numerical fusion."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from exp30_data_baseline import ROOT
from experiment_exp30 import LOCAL_MODES

from gnss_research.files import read_json, write_json


def transient_intervals(prediction):
    return [(event['start'], event['end']) for event in prediction['events']
            if event['kind'] == 'transient']


def diagnose(report):
    summary = read_json(report / 'summary.json')
    split = summary['split']
    visual = ROOT / 'implementation-s1' / (split + '-visual-v1')
    with gzip.open(report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    by_station = {method: {row['id']: row for row in values}
                  for method, values in rows.items()}
    totals, details = {}, []
    for mode, (family, _, _) in LOCAL_MODES.items():
        counts = Counter()
        for station, main in by_station[mode].items():
            base = by_station['vector_nearest'][station]
            pure = by_station[mode + '_pure'][station]
            base_intervals = transient_intervals(base['events'])
            main_intervals = transient_intervals(main['events'])
            added_intervals = [interval for interval in main_intervals if interval not in base_intervals]
            scan = read_json(visual / 'predictions' / ('scan_' + family) / station / 'result.json')
            regions = scan.get('decision', {}).get('regions', [])
            confirmed, uncertain = [], []
            failed_local = 0
            for index in range(len(regions)):
                receipt = read_json(visual / 'predictions' / mode / station / f'region-{index}' / 'result.json')
                failed_local += int(receipt['status'] != 'success')
                for interval in receipt.get('decision', {}).get('intervals', []):
                    if interval['status'] == 'confirmed':
                        confirmed.append((interval['start'], interval['end']))
                    else:
                        uncertain.append((interval['start'], interval['end']))
            main_daily = main['range']['transient']['daily']
            base_daily = base['range']['transient']['daily']
            pure_daily = pure['range']['transient']['daily']
            tp_delta = main_daily['tp'] - base_daily['tp']
            fp_delta = main_daily['fp'] - base_daily['fp']
            aff_main = main['range']['transient']['affiliation']['f1']
            aff_base = base['range']['transient']['affiliation']['f1']
            aff_delta = aff_main - aff_base if aff_main is not None and aff_base is not None else None
            negative = base_daily['tp'] + base_daily['fn'] == 0
            overlaps_base = sum(any(left <= old_right and old_left <= right
                                   for old_left, old_right in base_intervals)
                                for left, right in confirmed)
            counts.update(stations=1, regions=len(regions), failed_local=failed_local,
                confirmed_proposals=len(confirmed), uncertain_proposals=len(uncertain),
                confirmed_overlapping_numerical=overlaps_base,
                exact_duplicate_confirmed=len(confirmed) - len(set(confirmed)),
                added_final_intervals=len(added_intervals),
                stations_with_confirmed=int(bool(confirmed)),
                stations_with_new_intervals=int(bool(added_intervals)),
                recovered_tp_days=tp_delta, added_fp_days=fp_delta,
                tp_improved_stations=int(tp_delta > 0), fp_increased_stations=int(fp_delta > 0),
                negative_stations=int(negative),
                negative_stations_with_added_fp=int(negative and fp_delta > 0),
                negative_added_fp_days=fp_delta if negative else 0,
                aff_only_improved_stations=int(aff_delta is not None and aff_delta > 0 and tp_delta <= 0),
                pure_tp_days=pure_daily['tp'], pure_fp_days=pure_daily['fp'])
            details.append({'method': mode, 'station': station, 'scenario': main['scenario'],
                'regions': regions, 'confirmed': confirmed, 'uncertain': uncertain,
                'numerical_transients': base_intervals, 'added_final_intervals': added_intervals,
                'confirmed_overlapping_numerical': overlaps_base,
                'tp_delta': tp_delta, 'fp_delta': fp_delta, 'aff_delta': aff_delta})
        totals[mode] = dict(counts)
    result = {'experiment_id': 'GNSS-VLM-EXP30-S1', 'split': split,
        'phase': summary['phase'], 'stations': summary['stations'], 'new_model_requests': 0,
        'scope': 'Posthoc attribution of frozen predictions; no changed methods, thresholds or scores.',
        'notes': ['Raw proposal overlap and exact duplication counts are diagnostic and can overlap.',
                  'Added final intervals are read from materialized predictions, not inferred from explanations.',
                  'Affiliation is evaluated on truth-positive stations; negative-station FP are reported separately.'],
        'methods': totals, 'details': details}
    write_json(report / 'contribution-audit.json', result)
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    diagnose(parser.parse_args().report)
