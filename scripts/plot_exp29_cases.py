"""Show actual recovered days, false alarms and detections lost in verification."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp29_data_baseline import DATASET, ROOT
from experiment_exp29 import plot_values

from gnss_research.files import read_json, write_json


def main():
    report = Path('results/exp29-validation-full-v1')
    diagnosis = read_json(report / 'visual-role-diagnosis.json')
    rows = diagnosis['stations']
    cases = {}
    criteria = [
        ('verified_added_tp', lambda r: r['modes']['vf_vl_image']['complement']['added_tp_days']),
        ('verified_added_fp', lambda r: r['modes']['vf_vl_image']['complement']['added_fp_days']),
        ('raw_added_tp', lambda r: r['modes']['vf_vl_image']['raw_complement']['added_tp_days']),
        ('raw_added_fp', lambda r: r['modes']['vf_vl_image']['raw_complement']['added_fp_days'])]
    for title, score in criteria:
        selected = max(rows, key=score)
        if score(selected) > 0:
            cases.setdefault(selected['station'], {'selection': [], 'diagnosis': selected})['selection'].append(title)
    visual = ROOT / 'validation-visual-v1'
    output = report / 'cases'
    output.mkdir(exist_ok=False)
    for station, case in cases.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        methods = {'Truth': {'events': truth['events']}}
        for title, method in [('Visual raw', 'vf_vl_image_pure'),
            ('Visual + numerical complement', 'vf_vl_image_complement'), ('Numerical nearest', 'vector_nearest')]:
            methods[title] = read_json(visual / 'final' / method / station / 'result.json')['prediction']
        values = plot_values(observation)
        first = observation['background_end'] + 1
        time = np.arange(first, len(values))
        figure, panels = plt.subplots(4, 1, figsize=(14, 9), dpi=140, sharex=True)
        for axis, panel in enumerate(panels[:3]):
            panel.plot(time, values[first:, axis], '.-', color='#346c85', markersize=2.5, linewidth=.8)
            for event in truth['events']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'] + 1, color='#d5a537', alpha=.2)
            panel.set_ylabel('NEU'[axis] + ' / station MAD')
            panel.grid(alpha=.2)
        colors = ['#c29525', '#9b6eab', '#387f91', '#727272']
        for index, (title, prediction) in enumerate(methods.items()):
            spans = [(e['start'], e['end'] - e['start'] + 1)
                for e in prediction['events'] if e['kind'] == 'transient']
            panels[3].broken_barh(spans, (index - .3, .6), facecolors=colors[index])
        panels[3].set_yticks(range(len(methods)), list(methods))
        panels[3].set_ylim(-.7, len(methods) - .3)
        panels[3].set_xlim(first, len(values) - 1)
        panels[3].set_xlabel('Original daily index; missing observations are not filled')
        panels[3].grid(axis='x', alpha=.2)
        figure.suptitle(station + ': ' + ', '.join(case['selection']) + '; gold shading = transient truth')
        figure.tight_layout()
        figure.savefig(output / (station + '.png'))
        plt.close(figure)
        case['truth_events'] = truth['events']
        case['method_predictions'] = methods
    write_json(output / 'cases.json', cases)
    print('Wrote', len(cases), 'distinct positive/negative cases', flush=True)


if __name__ == '__main__':
    main()
