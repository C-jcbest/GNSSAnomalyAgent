"""Single contact-sheet localization without numerical or demonstration anchoring."""
from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiment_exp36 as parent
import matplotlib.pyplot as plt
import numpy as np
from exp28_data_baseline import METHODS
from experiment_exp25 import settings
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values, transient_event
from experiment_exp30 import LocalDecision
from experiment_exp30_s1 import complete_windows
from experiment_exp35 import aggregate_types
from matplotlib.ticker import MultipleLocator
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method, compare
from report_per_type import export

from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path('artifacts/exp37')
PROTOCOL = Path('docs/experiment-exp37.md')
MODES = {'owv_flash': ('qwen3.8-flash', 'image'),
    'owv_flash_text': ('qwen3.8-flash', 'text'),
    'owv_vl': ('qwen3-vl-plus', 'image'), 'owv_vl_text': ('qwen3-vl-plus', 'text')}
PROMPT = """Locate TEMPORARY OBSERVATION OFFSETS in daily GNSS N/E/U displacement.
The complete target is supplied as three overlapping windows. Every row is a different
component; every column a different original time window, not a different station.
Examine ALL windows, not only the last column or the largest jump. Read indices on EACH
panel. Overlapping samples are repeated views, not independent evidence. Text input gives
the exact same raw plotted samples. No numerical detection, anomaly count or labels supplied.

Find an abrupt displacement offset, a persistent shifted segment, then an opposing return
in the SAME component(s). One clear affected component can suffice; DO NOT require all
three axes to show a large jump. Smaller or noisy components can be inconclusive.
Smooth real movement may continue during a temporary offset. A separate permanent step
may lie inside or near it, so the outside absolute levels need NOT match. Compare local
onset and return directions rather than demanding a flat rectangle. A smooth gradual
bend followed by arrest is movement, not an offset; a permanent step alone has no return;
a spike is an isolated sample. Do not mistake the largest permanent step for the only event.

Missing points are unknown, never zeros. If the onset falls inside a gap but the observed
shifted interior AND an opposing return are clear, report only the OBSERVABLE interval:
start at the FIRST actually observed shifted sample, not an inferred date in the gap;
end at the LAST observed shifted sample BEFORE the return. A gap does not by itself prove
an offset. If the shifted interior or opposing return is absent/ambiguous, use uncertain.
Do not invent a return or assign labels to missing samples. No universal amplitude or speed
cutoff. Axes are normalized by station historical MAD, not physical millimeters.
Gray points/line are original data; blue 3-day medians are only readability aids.

Use original inclusive indices, never positions relative to the cropped panel.
Output temporary offsets ONLY, with confirmed/uncertain status and concise axis-specific
observations supporting both boundaries. Empty is valid. Do not output physical causes,
motion phases, point spikes, or permanent steps as temporary offsets.
"""


def render_sheet(observation, target):
    values = plot_values(observation)
    windows = complete_windows(observation)
    figure, panels = plt.subplots(3, 3, figsize=(25, 12.5), dpi=120, squeeze=False)
    for column, (left, right) in enumerate(windows):
        time = np.arange(left, right)
        for axis in range(3):
            panel = panels[axis, column]
            raw = values[left:right, axis]
            smooth = raw.copy()
            for position in range(1, len(raw) - 1):
                if np.isfinite(raw[position]):
                    smooth[position] = np.nanmedian(raw[position - 1:position + 2])
            panel.plot(time, raw, '.-', linewidth=.7, markersize=3, color='#777777')
            panel.plot(time, smooth, linewidth=1.1, color='#246e8c')
            panel.set_xlim(left, right - 1)
            panel.xaxis.set_major_locator(MultipleLocator(10))
            panel.xaxis.set_minor_locator(MultipleLocator(5))
            panel.grid(alpha=.25)
            panel.grid(which='minor', alpha=.10)
            panel.tick_params(axis='x', rotation=60, labelsize=9)
            panel.set_title(f"{'NEU'[axis]} | original indices {left} to {right - 1}")
            panel.set_ylabel('Own station MAD; centered')
            panel.set_xlabel('Original daily index; gaps stay unknown')
    figure.suptitle('ONE STATION: examine every time window. Rows N / E / U; columns overlap in time.')
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def prepare():
    records = parent.records_for('pilot')
    output = ROOT / 'diagnostic-v1'
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for record in records:
        station = record['station_id']
        observation = read_json(parent.DATASET / record['observation'])
        values = plot_values(observation)
        image = output / (station + '.png')
        render_sheet(observation, image)
        metadata = {'monitoring_start': observation['background_end'] + 1,
            'last_index': len(values) - 1, 'windows': complete_windows(observation),
            'window_end_convention': 'exclusive in metadata; plotted index labels inclusive'}
        for mode, (model, view) in MODES.items():
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            if view == 'image':
                content.append(image_block(image))
            else:
                for left, right in complete_windows(observation):
                    content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload(model, PROMPT, LocalDecision, content))
            jobs.append({'work': str(work), 'mode': mode, 'station': station, 'model': model,
                'request_sha256': sha256(work / 'request.json')})
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP37',
        'scope': 'posthoc development diagnostic on EXP36 stations; not independent confirmation',
        'records': records, 'jobs': jobs, 'source_hashes': source_hashes(),
        'frozen_files': {str(path): sha256(path) for path in (Path(__file__), PROTOCOL, *parent.DEPENDENCIES)},
        'parent_summary_sha256': sha256(Path('results/exp36-pilot-v1/summary.json')),
        'parent_per_station_sha256': sha256(Path('results/exp36-pilot-v1/per-station.json.gz'))})


def infer():
    output = ROOT / 'diagnostic-v1'
    registration = read_json(output / 'registration.json')
    if source_hashes() != registration['source_hashes'] or any(sha256(Path(path)) != digest
            for path, digest in registration['frozen_files'].items()):
        raise ValueError('registered source changed')
    write_json(output / 'started.json', {'requests': len(registration['jobs']), 'retries': 0})

    def call(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed')
        receipt, content = parent.send_request(read_json(work / 'request.json'), work, settings(job['model']))
        if receipt['status'] == 'success':
            try:
                decision = LocalDecision.model_validate_json(content).model_dump()
                if any(not 365 <= item['start'] <= item['end'] < 730 for item in decision['intervals']):
                    raise ValueError('interval outside target')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt['status']

    with ThreadPoolExecutor(max_workers=3) as executor:
        for number, result in enumerate(executor.map(call, registration['jobs']), 1):
            print(number, '/', len(registration['jobs']), *result, flush=True)
    write_json(output / 'complete.json', {'requests': len(registration['jobs'])})


def report():
    output = ROOT / 'diagnostic-v1'
    if not (output / 'complete.json').exists():
        raise ValueError('complete all requests first')
    registration = read_json(output / 'registration.json')
    with gzip.open('results/exp36-pilot-v1/per-station.json.gz', 'rt', encoding='utf-8') as stream:
        saved = json.load(stream)
    rows = {method: saved[method] for method in (*METHODS, 'vector_nearest', 'gnv_image')}
    for mode in MODES:
        rows[mode], rows[mode + '_pure'] = [], []
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(parent.DATASET / record['observation'])
        base = read_json(parent.ROOT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        predictions = {}
        for mode in MODES:
            receipt = read_json(output / 'predictions' / mode / station / 'result.json')
            events = [transient_event(item['start'], item['end'], item['observation'])
                for item in receipt.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']
            predictions[mode] = parent.fuse(observation, base, events)
            predictions[mode + '_pure'] = parent.result_from_events(observation, events)
        truth = read_json(parent.DATASET / record['truth'])
        for method, result in predictions.items():
            write_json(output / 'final' / method / station / 'result.json', result)
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    paired = {control: compare(rows, 'owv_flash', control)
        for control in ('vector_nearest', 'gnv_image', 'owv_flash_text', 'owv_vl')}
    paired['pure_flash_image_vs_text'] = compare(rows, 'owv_flash_pure', 'owv_flash_text_pure')
    paired['pure_vl_image_vs_text'] = compare(rows, 'owv_vl_pure', 'owv_vl_text_pure')
    main = paired['vector_nearest']
    checks = {'pure_visual_aff_gain_at_least_003': paired['pure_flash_image_vs_text']['aff_delta'] >= .03,
        'system_aff_gain_vs_nearest_at_least_002': main['aff_delta'] >= .02,
        'actual_station_tp_gain': main['tp_improved_stations'] > 0}
    target = Path('results/exp37-development-diagnostic-v1')
    target.mkdir(parents=True, exist_ok=False)
    summary = {'experiment_id': 'GNSS-VLM-EXP37', 'stations': len(registration['records']),
        'scope': registration['scope'], 'methods': summaries,
        'costs': aggregate_cost(request_receipts([job['work'] for job in registration['jobs']]))}
    write_json(target / 'summary.json', summary)
    write_json(target / 'paired-changes.json', paired)
    write_json(target / 'admission.json', {'fresh_confirmation_admitted': all(checks.values()), 'checks': checks})
    strata = {key: {method: {str(value): aggregate_types([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()
        if values[0]['original'].get('types') is not None} for key in ('cohort_seed', 'scenario')}
    write_json(target / 'strata.json', strata)
    with (target / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    export(target / 'summary.json', target / 'per-type')
    print(json.dumps({'checks': checks, 'costs': summary['costs'],
        'paired': {key: {k: v for k, v in value.items() if k != 'changes'} for key, value in paired.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'report'))
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'report': report}[args.stage]()
