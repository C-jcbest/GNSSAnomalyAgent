"""Read-only lifecycle audit of frozen data, inputs, predictions and call records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gnss_sim.evaluation import evaluate_pilot, load_results_jsonl
from gnss_sim.evaluation_views import evaluate_views
from gnss_sim.numerical_runner import _predict
from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseInput, CaseTruth, PointResult, RangeResult
from gnss_sim.visual import parse_result

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_source(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def main():
    pilot = ROOT / "data/pilots/pilot-v1"
    manifest = verify_pilot(pilot)
    truths, inputs = {}, {}
    for entry in manifest["cases"]:
        key = entry["case_id"]
        directory = pilot / "cases" / key
        case = CaseInput.model_validate_json((directory / "input.json").read_bytes())
        truth = CaseTruth.model_validate_json((directory / "truth.json").read_bytes())
        displacement = np.asarray(case.displacement_mm)
        np.testing.assert_allclose(displacement, np.asarray(case.observed_coordinate_mm)
                                   - case.reference_coordinate_mm, rtol=0, atol=1e-13)
        np.testing.assert_allclose(case.horizontal_offset_mm,
                                   np.linalg.norm(displacement[:, :2], axis=1), rtol=0, atol=1e-13)
        np.testing.assert_allclose(case.spatial_offset_mm,
                                   np.linalg.norm(displacement, axis=1), rtol=0, atol=1e-13)
        assert all((right - left).days == 1 for left, right in zip(case.dates, case.dates[1:]))
        truths[key], inputs[key] = truth, case
    component_unique = {name: len({getattr(truth.component_seeds, name)
                                  for truth in truths.values()})
                        for name in ("annual_phase", "semiannual_phase", "white_noise")}
    assert all(value == 300 for value in component_unique.values())
    assert len({entry["case_seed"] for entry in manifest["cases"]}) == 300
    background_unique = len({hashlib.sha256(np.asarray(t.normal_background_mm).tobytes()).hexdigest()
                             for t in truths.values()})
    assert background_unique == 300
    registry = json.loads((ROOT / "configs/comparison-views-v1.json").read_bytes())
    saved = json.loads((ROOT / "runs/comparison-views-v1/summary.json").read_bytes())
    frozen = {name: json.loads((ROOT / f"configs/{name}-frozen.json").read_bytes())
              for name in ("p5", "p6", "p7a")}
    for record in frozen.values():
        for name, expected in record["source_sha256"].items():
            assert normalized_source(ROOT / "src/gnss_sim" / name) == expected, name
    for name in ("p6", "p7a"):
        config = "configs/p6-visual.json" if name == "p6" else "configs/p7a-visual-semantics.json"
        assert digest(ROOT / config) == frozen[name]["config_sha256"]
    methods = {}
    call_audit = {}
    sample_ids = [next(e["case_id"] for e in manifest["cases"] if e["case_type"] == kind)
                  for kind in ("normal", "spike", "step", "slow_trend",
                               "acceleration", "transient_shift")]
    for spec in registry["methods"]:
        method, task = spec["method_id"], spec["task"]
        key = f"{method}/{task}"
        path = ROOT / spec["predictions"]
        rows, errors = load_results_jsonl(path, task)
        assert not errors and len(rows) == 300 and {row.case_id for row in rows} == truths.keys()
        assert all(row.method == method for row in rows)
        assert digest(path) == saved["methods"][key]["provenance"]["predictions_sha256"]
        actual = evaluate_pilot(pilot, rows, method, task)
        assert actual == json.loads(path.with_name("report.json").read_bytes())
        methods[key] = {row.case_id: row for row in rows}
        if method in frozen["p5"]["methods"]:
            parameters = frozen["p5"]["methods"][method]
            run = json.loads(path.with_name("run.json").read_bytes())
            assert parameters == run["parameters"]
            for case_id in sample_ids:
                assert _predict(method, inputs[case_id], parameters["calibration"]) == methods[key][case_id]
            continue
        record = frozen["p6" if method == "visual-v1" else "p7a"]
        assert digest(path) == record["predictions_sha256"][task]
        raw_path = path.with_name("raw.jsonl")
        raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        assert len(raw) == 300 and {row["case_id"] for row in raw} == truths.keys()
        for row in raw:
            result = methods[key][row["case_id"]]
            assert result.status == row["status"]
            if row["status"] == "success":
                assert parse_result(row["response"], row["case_id"], task, method) == result
            else:
                assert not any(getattr(result.predictions, axis) for axis in ("N", "E", "U"))
            image_path = ROOT / "runs/p6/qwen3.8-flash/images" / (row["case_id"] + ".png")
            assert digest(image_path) == row["image_sha256"]
        call_audit[key] = {"calls": len(raw), "success": sum(r["status"] == "success" for r in raw),
                           "errors": dict(Counter(r["error_type"] for r in raw if r["status"] != "success"))}
    union = {}
    for task, numerical, model in (("point", "sr", PointResult), ("range", "theilsen", RangeResult)):
        n, v = methods[f"{numerical}/{task}"], methods[f"visual-semantics-v2/{task}"]
        combined = {key: model(case_id=key, method="audit_union", status="success", predictions={
            axis: sorted(set(getattr(n[key].predictions, axis) +
                             (getattr(v[key].predictions, axis) if v[key].status == "success" else [])))
            for axis in ("N", "E", "U")}) for key in truths}
        union[task] = evaluate_views(truths, combined, task)
    hits = {}
    for method in ("sr", "pelt"):
        for kind in ("spike", "step"):
            events = [(key, event) for key, truth in truths.items()
                      for event in truth.events if event.type == kind]
            hits[f"{method}/{kind}"] = {
                "events": len(events), "exact_hits": sum(event.start_index in getattr(
                    methods[f"{method}/point"][key].predictions, event.axis) for key, event in events)}
    result = {"scope": "posthoc_development_audit_no_model_calls", "verified_cases": len(truths),
              "unique_component_seeds": component_unique, "unique_backgrounds": background_unique,
              "reproduced_reports": len(methods), "numerical_replayed_cases_per_method": sample_ids,
              "call_audit": call_audit, "point_hits_by_truth_type": hits,
              "union_replay_numerical_fallback_on_visual_failure": union,
              "pilot_manifest_sha256": digest(pilot / "manifest.json"),
              "source_sha256": digest(Path(__file__))}
    out = ROOT / "runs/audit-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
