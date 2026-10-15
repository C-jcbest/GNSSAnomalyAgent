import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp18_distribution import distributions, training_reference  # noqa: E402

from gnss_research.files import write_json


def test_distribution_reference_uses_training_only_and_observed_sample_support(tmp_path):
    time = np.arange(150)
    values = np.column_stack([np.sin(time), np.cos(time), np.sin(.7 * time)])
    values[80:115] += [12, -15, 8]
    values[95:] += [-3, 4, 5]
    observation = {"dates": time.tolist(), "background_end": 60,
                   "displacement_mm": values.tolist()}
    observation["displacement_mm"][82] = None
    truth = {"events": [{"kind": "step", "start": 95, "end": 95},
                        {"kind": "transient", "start": 80, "end": 114}]}
    write_json(tmp_path / "observation.json", observation)
    write_json(tmp_path / "truth.json", truth)
    records = [{"station_id": name, "split": split,
                "observation": "observation.json", "truth": "truth.json"}
               for name, split in (("a_validation", "validation"), ("b_training", "train"))]
    reference = training_reference(tmp_path, records)
    assert reference["station_id"] == "b_training"
    assert reference["split"] == "train"
    positive, negative = reference["samples"]
    assert positive["edges"][1]["absolute_index"] == 115
    assert negative["edges"][1]["absolute_index"] == 95
    onset = positive["edges"][0]
    assert onset["after_indices"][2] == 82
    assert onset["after"][2] == [None, None, None]
    # Closely spaced numerical proposals truncate actual samples rather than borrow the next jump.
    graph = {"jumps": [{"id": "J1", "index": 80}, {"id": "J2", "index": 83}],
             "pairs": [{"id": "P1", "first_id": "J1", "second_id": "J2"}]}
    near = distributions(observation, graph)[0]["edges"][0]
    assert near["after_indices"] == [80, 81, 82]
    assert sum(all(value is not None for value in row) for row in near["after"]) == 2
