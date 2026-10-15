import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp14_fingerprint import apply, free_graph  # noqa: E402
from exp15_vector import nearest, vectors  # noqa: E402
from exp16_overlap import contrasts  # noqa: E402

from gnss_research.schema import Prediction, event_masks, observed_values


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "evidence": "observed", "phases": []}


def test_only_free_edges_can_pair_and_interior_events_and_missing_stay_unchanged():
    numeric = {"jumps": [{"index": i} for i in (100, 105, 120, 130, 140)],
        "prediction": {"events": [event("step", i, i) for i in (100, 105, 120)]
            + [event("transient", 130, 139), event("spike", 104, 104), event("deformation", 95, 145)]}}
    obs = {"dates": list(range(150)), "displacement_mm": np.zeros((150, 3)).tolist()}
    obs["displacement_mm"][110] = None
    graph = free_graph(numeric)
    assert {j["index"] for j in graph["jumps"]} == {100, 105, 120}
    pair = next(p for p in graph["pairs"] if p["start"] == 100 and p["end"] == 119)
    response = {"status": "success", "pair_decision": {"pair_ids": [pair["id"]], "evidence": "opposing edges"}}
    result = apply(obs, numeric, graph, response)
    found = {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]}
    assert found == {("transient", 100, 119), ("step", 105, 105), ("transient", 130, 139),
                     ("spike", 104, 104), ("deformation", 95, 145)}
    masks = event_masks(Prediction.model_validate(result["prediction"]), np.isfinite(observed_values(obs)).all(axis=1))
    assert not any(mask[110] for mask in masks.values())
    assert apply(obs, numeric, graph, {"status": "failed"})["prediction"] == numeric["prediction"]


def test_local_vectors_exclude_following_edge_and_do_not_fill_missing_samples():
    t = np.arange(100)
    values = np.column_stack([np.sin(t), np.cos(t), np.sin(t * .5)])
    values[60:64] += [10, -20, 5]
    values[64:] += [100, 100, 100]
    observation = {"dates": t.tolist(), "background_end": 50,
                   "displacement_mm": values.tolist()}
    observation["displacement_mm"][62] = None
    graph = {"jumps": [{"id": "J1", "index": 60}, {"id": "J2", "index": 64}], "pairs": []}
    result = vectors(observation, graph)
    assert result[0]["after_observed"] == 3
    assert np.sign(result[0]["delta"]).tolist() == [1, -1, 1]
    assert observation["displacement_mm"][62] is None


def test_untuned_nearest_direction_selects_opposite_nonadjacent_edges_without_sharing():
    graph = {"pairs": [{"id": "P1", "first_id": "J1", "second_id": "J2"},
                        {"id": "P2", "first_id": "J1", "second_id": "J3"},
                        {"id": "P3", "first_id": "J2", "second_id": "J3"}]}
    local = [{"id": "J1", "delta": [2, -3, 1]}, {"id": "J2", "delta": [-5, 1, 1]},
             {"id": "J3", "delta": [-2, 3, -1]}]
    assert nearest(graph, local)["pair_decision"]["pair_ids"] == ["P2"]


def test_overlap_compares_onset_with_negated_return_without_changing_observations():
    graph = {"pairs": [{"id": "P2", "first_id": "J1", "second_id": "J3"}]}
    local = [{"id": "J1", "delta": [2, -3, 1]}, {"id": "J3", "delta": [-2, 3, -1]}]
    pair = contrasts(graph, local)[0]
    assert pair["onset"] == pair["negated_return"] == [2, -3, 1]
    assert local[1]["delta"] == [-2, 3, -1]
