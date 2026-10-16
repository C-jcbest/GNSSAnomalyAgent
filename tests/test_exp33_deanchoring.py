"""Remove the proposed source of bias while preserving the paired evidence."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp33 import remove_narratives  # noqa: E402


def test_deanchoring_preserves_images_numeric_evidence_bounds_and_original_request():
    metadata = {'screening_summary': 'suspected interpretation',
        'screened_regions': [{'start': 400, 'end': 430, 'observation': 'fallible description'}],
        'allowed_contexts': [[365, 465]], 'optional_numerical_jumps': [{'index': 410, 'jump_vector': [1, 2, 3]}]}
    request = {'model': 'qwen3-vl-plus', 'messages': [{'role': 'system', 'content': 'unchanged prompt'},
        {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(metadata)},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,unchanged'}},
            {'type': 'text', 'text': 'unchanged original numeric rows'}]}]}
    original = copy.deepcopy(request)
    transformed = remove_narratives(request)
    expected_metadata = {key: value for key, value in metadata.items() if key != 'screening_summary'}
    expected_metadata['screened_regions'] = [{'start': 400, 'end': 430}]
    assert json.loads(transformed['messages'][1]['content'][0]['text']) == expected_metadata
    transformed['messages'][1]['content'][0] = original['messages'][1]['content'][0]
    assert transformed == original
    assert request == original
