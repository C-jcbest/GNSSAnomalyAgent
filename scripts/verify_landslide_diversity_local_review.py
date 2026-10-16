"""Replay all bounded-review decisions and request bindings without new API calls."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_diversity_local_review import ARMS, NUMERIC, assemble, verify_sources
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_visual import ACTIVITY_TASK, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity
from gnss_sim.visual import _unique_object


def verify(run, out):
    protocol = verify_sources(run)
    verify_manifest(run, "prediction-freeze.json")
    verify_manifest(run / "analysis-v1", "freeze.json")
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Require completed execution and scoring")
    out.mkdir(parents=True, exist_ok=False)
    policy = ReviewPolicy(**protocol["policy"])
    stage = XGBClassifier()
    stage.load_model(Path(protocol["stage_source"]) / "stage-xgboost.json")
    numeric = read_json(run / "numeric-initial.json")
    parent = read_json(Path(protocol["parent"]) / "predictions.json")
    if numeric != parent[NUMERIC]:
        raise ValueError("Strong numerical prior changed")
    codes, masks = read_json(run / "stage-codes.json"), read_json(run / "support-masks.json")
    receipts, predictions, selections, used, windows_saved = {}, {}, {}, {}, {}
    parse_failures = set()

    def response(request_id, prompt, images):
        started = read_json(run / "requests" / (request_id + ".started.json"))
        receipt = read_json(run / "requests" / (request_id + ".response.json"))
        content = [{"type": "text", "text": prompt}]
        for path in images:
            encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded_image}})
        payload = dict(model=protocol["model"], messages=[dict(role="user", content=content)],
                       temperature=.1, top_p=.3, max_tokens=2048, enable_thinking=False,
                       response_format={"type": "json_object"})
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        images_sha = {p.name: sha(p) for p in images}
        for record in (started, receipt):
            if record["request_sha256"] != expected or record["image_sha256"] != images_sha or record["prompt"] != prompt:
                raise AssertionError("Image, prompt or complete request hash differs: " + request_id)
        receipts[request_id] = receipt
        if receipt["status"] != "returned_json":
            raise RuntimeError("Transport failure retained")
        return json.loads(receipt["output"], object_pairs_hook=_unique_object)

    data = Path(protocol["data"])
    for rep in range(protocol["repetitions"]):
        for cid in protocol["ids"]:
            case, features = load_case_features(data, cid)
            with np.load(Path(protocol["encoded"]) / "validation" / (cid + ".npz")) as bank:
                np.testing.assert_array_equal(stage.predict(bank["stage_matrix"]).astype(int) + 1, codes[cid])
                np.testing.assert_array_equal(bank["supported"], masks[cid]["activity"])
                np.testing.assert_array_equal(bank["stage_supported"], masks[cid]["stage"])
            evidence = raw_motion_evidence(case, features)
            windows = regular_windows(len(case.dates), 180, 90)
            key = f"{cid}-r{rep}"
            global_id = key + "-global"
            global_image = run / "images" / (cid + "-global.png")
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200]."
            )
            try:
                initial = parse_activity(response(global_id, prompt, [global_image]), features.observed)
                status = "success"
            except (ValueError, RuntimeError):
                parse_failures.add(global_id)
                initial, status = np.full(len(case.dates), -1), "failed"
            prior = np.array(numeric[cid]["activity"])
            visual_priority, visual_targets = review_priorities(initial, features, evidence, windows, policy)
            numeric_priority, numeric_targets = review_priorities(prior, features, evidence, windows, policy)
            centers = np.linspace(0, len(case.dates), 4)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index]) / 2 - center))
                            for center in centers})
            selection = dict(fixed=fixed, visual_selected=[row["index"] for row in visual_targets],
                             numeric_selected=[row["index"] for row in numeric_targets],
                             visual_priorities=visual_priority, numeric_priorities=numeric_priority)
            selections[key] = selection
            reviews = {}
            for index in sorted(set(fixed + selection["visual_selected"] + selection["numeric_selected"])):
                start, stop = windows[index]
                local_id = key + f"-window-{index:02d}"
                try:
                    payload = response(local_id, review_prompt(features, evidence, start, stop, False),
                                       [global_image, run / "images" / f"{cid}-window-{index:02d}.png"])
                    activity = parse_window_activity(payload, features.observed, start, stop)
                    review_status = "success"
                except (ValueError, RuntimeError):
                    parse_failures.add(local_id)
                    activity, review_status = np.full(len(case.dates), -1), "failed"
                reviews[index] = dict(start=start, stop=stop, activity=activity.tolist(),
                                      status=review_status, request_id=local_id)
            windows_saved[key] = {str(index): value for index, value in reviews.items()}
            outputs = assemble(initial, status, prior, np.array(codes[cid]), features.observed,
                               np.array(masks[cid]["activity"]), np.array(masks[cid]["stage"]), reviews, selection)
            for arm in ARMS:
                method = arm + f"_r{rep}"
                predictions.setdefault(method, {})[cid] = outputs[arm]
                if arm == "direct_visual":
                    indices = []
                elif arm.startswith("fixed"):
                    indices = selection["fixed"]
                elif arm == "adaptive_visual":
                    indices = selection["visual_selected"]
                else:
                    indices = selection["numeric_selected"]
                request_ids = [] if arm.endswith("numeric") else [global_id]
                request_ids += [reviews[index]["request_id"] for index in indices]
                used.setdefault(method, {})[cid] = request_ids
    if len(receipts) != len(list((run / "requests").glob("*.started.json"))) or len(receipts) > protocol["maximum_requests"]:
        raise ValueError("Extra request, incomplete response or budget violation")
    for name, rebuilt in (("predictions", predictions), ("selections", selections),
                          ("window-outputs", windows_saved), ("used", used)):
        if rebuilt != read_json(run / (name + ".json")):
            raise ValueError("Exact replay differs: " + name)
    if parse_failures != set(read_json(run / "failures.json")):
        raise ValueError("Recorded transport/parser failures differ")
    labels = {cid: [] for cid in protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    parent.update(predictions)
    results, summary, _ = score_target(labels, observed, weak, parent, used, receipts)
    if results != read_json(run / "analysis-v1/results.json") or summary != read_json(run / "analysis-v1/summary.json"):
        raise ValueError("Unified scores differ")
    write_json(out / "verification.json", dict(
        status="VERIFIED", requests=len(receipts), recorded_failures=len(parse_failures),
        exact_payload_images_prior_selection_stage_outputs_scores=True,
        models=len(parent), outputs=len(parent) * len(labels), new_api_calls=0,
        prepared_manifest_sha256=sha(run / "prepared-freeze.json"),
        prediction_manifest_sha256=sha(run / "prediction-freeze.json"),
        analysis_manifest_sha256=sha(run / "analysis-v1/freeze.json"), reserved_content_loaded=False,
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
