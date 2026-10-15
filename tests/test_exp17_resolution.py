import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp17_resolution import local_samples  # noqa: E402

from gnss_research.features import numerical_evidence


def test_local_pair_alignment_preserves_original_days_gaps_and_opposite_return():
    time = np.arange(150)
    values = np.column_stack([np.sin(time), np.cos(time), np.sin(.7 * time)])
    values[80:115] += [12, -15, 8]
    observation = {"dates": time.tolist(), "background_end": 60,
                   "displacement_mm": values.tolist()}
    observation["displacement_mm"][117] = None
    graph = {"jumps": [{"id": "J1", "index": 80}, {"id": "J2", "index": 115}],
             "pairs": [{"id": "P1", "first_id": "J1", "second_id": "J2"}]}
    before = [row.copy() if row is not None else None
              for row in observation["displacement_mm"]]
    pair = local_samples(observation, graph)[0]
    onset, returning = pair["edges"]
    assert onset["absolute_index"] == 80
    assert returning["absolute_index"] == 115
    assert returning["relative_days"][16] == 2
    assert returning["values_NEU"][16] == [None, None, None]
    noise = numerical_evidence(observation).noise
    onset_values = np.asarray(onset["values_NEU"], dtype=float)
    return_values = np.asarray(returning["values_NEU"], dtype=float)
    # A physical +N/-E/+U shift and its return must have the same displayed orientation.
    for displayed in (onset_values, return_values):
        jump = np.nanmedian(displayed[14:21], axis=0) - np.nanmedian(displayed[7:14], axis=0)
        assert np.sign(jump).tolist() == [1, -1, 1]
        assert np.allclose(jump * noise, [12, -15, 8], atol=1.5)
    assert observation["displacement_mm"] == before
