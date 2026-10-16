"""Replay all expanded RCRA answers, permissions, request bindings and scores."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_daily_authority_expansion import verify_prepared
from run_landslide_diversity_local_review import assemble
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from verify_landslide_daily_revision_authority import reconstruct
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
    protocol = verify_prepared(run)
    verify_manifest(run, "prediction-freeze.json")
    verify_manifest(run / "analysis-v1", "freeze.json")
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Completed inference and scoring required")
    data = Path(protocol["data"])
    cached = Path(protocol["cached_source"])
    stage = XGBClassifier()
    stage.load_model(Path(protocol["stage_source"]) / "stage-xgboost.json")
    codes, masks = read_json(run / "stage-codes.json"), read_json(run / "support-masks.json")
    predictions, used, selections, reviews_saved, audits, scopes = {}, {}, {}, {}, [], {}
    receipts, failures, new_ids = {}, {}, set()
    policy = ReviewPolicy(**protocol["policy"])

    def response(cid, request_id, prompt, images):
        folder = cached if cid in protocol["reused_ids"] else run
        if folder == run:
            new_ids.add(request_id)
        started = read_json(folder / "requests" / (request_id + ".started.json"))
        receipt = read_json(folder / "requests" / (request_id + ".response.json"))
        content = [dict(type="text", text=prompt)]
        for path in images:
            content.append(dict(type="image_url", image_url=dict(
                url="data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            )))
        payload = dict(model=protocol["model"], messages=[dict(role="user", content=content)],
                       temperature=.1, top_p=.3, max_tokens=2048, enable_thinking=False,
                       response_format=dict(type="json_object"))
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        for record in (started, receipt):
            if record["prompt"] != prompt or record["request_sha256"] != expected:
                raise AssertionError("Complete request payload differs: " + request_id)
            if record["image_sha256"] != {p.name: sha(p) for p in images}:
                raise AssertionError("Image binding differs: " + request_id)
        receipts[request_id] = receipt
        if receipt["status"] != "returned_json":
            raise RuntimeError("Retained transport failure")
        return json.loads(receipt["output"], object_pairs_hook=_unique_object)

    for rep in range(2):
        for cid in protocol["ids"]:
            case, features = load_case_features(data, cid)
            with np.load(Path(protocol["encoded"]) / "validation" / (cid + ".npz")) as bank:
                np.testing.assert_array_equal(stage.predict(bank["stage_matrix"]).astype(int) + 1, codes[cid])
                np.testing.assert_array_equal(bank["supported"], masks[cid]["activity"])
                np.testing.assert_array_equal(bank["stage_supported"], masks[cid]["stage"])
            evidence = raw_motion_evidence(case, features)
            windows = regular_windows(len(case.dates), 180, 90)
            key, global_id = f"{cid}-r{rep}", f"{cid}-r{rep}-global"
            image_folder = cached if cid in protocol["reused_ids"] else run
            global_image = image_folder / "images" / (cid + "-global.png")
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200]."
            )
            try:
                initial = parse_activity(response(cid, global_id, prompt, [global_image]), features.observed)
                status = "success"
            except (ValueError, RuntimeError):
                failures[global_id] = True
                initial, status = np.full(len(case.dates), -1), "failed"
            priorities, targets = review_priorities(initial, features, evidence, windows, policy)
            centers = np.linspace(0, len(case.dates), 4)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index]) / 2 - center))
                            for center in centers})
            selection = dict(fixed=fixed, visual_selected=[row["index"] for row in targets],
                             visual_priorities=priorities, numeric_selected=[])
            selections[key] = selection
            reviews = {}
            for index in sorted(set(fixed + selection["visual_selected"])):
                start, stop = windows[index]
                request_id = key + f"-window-{index:02d}"
                try:
                    payload = response(cid, request_id, review_prompt(features, evidence, start, stop, False),
                                       [global_image, image_folder / "images" / f"{cid}-window-{index:02d}.png"])
                    activity = parse_window_activity(payload, features.observed, start, stop)
                    review_status = "success"
                except (ValueError, RuntimeError):
                    failures[request_id] = True
                    activity, review_status = np.full(len(case.dates), -1), "failed"
                reviews[index] = dict(start=start, stop=stop, activity=activity.tolist(),
                                      status=review_status, request_id=request_id)
            reviews_saved[key] = {str(index): value for index, value in reviews.items()}
            decoded = assemble(initial, status, initial, np.array(codes[cid]), features.observed,
                               np.array(masks[cid]["activity"]), np.array(masks[cid]["stage"]), reviews, selection)
            for arm in ("direct_visual", "fixed_visual", "adaptive_visual"):
                method = f"{arm}_r{rep}"
                predictions.setdefault(method, {})[cid] = decoded[arm]
                if arm == "direct_visual":
                    indices = []
                elif arm == "fixed_visual":
                    indices = fixed
                else:
                    indices = selection["visual_selected"]
                used.setdefault(method, {})[cid] = [global_id] + [reviews[index]["request_id"] for index in indices]
            outputs, diagnostics, details = reconstruct(
                initial, [reviews[index] for index in selection["visual_selected"]],
                features.observed, features, evidence, policy, targets, np.array(codes[cid]),
                {name: np.array(value) for name, value in masks[cid].items()},
            )
            scopes[key] = details
            for variant, prediction in outputs.items():
                method = f"visual_{variant}_r{rep}"
                predictions.setdefault(method, {})[cid] = prediction
                used.setdefault(method, {})[cid] = [global_id] if variant == "no_local_visual" else used[f"adaptive_visual_r{rep}"][cid]
                audits.append(dict(method=method, case_id=cid, **diagnostics[variant]))
    for name, value in (("predictions", predictions), ("used", used), ("selections", selections),
                        ("window-outputs", reviews_saved), ("authority-audit", audits), ("scopes", scopes)):
        if value != read_json(run / (name + ".json")):
            raise ValueError("Exact expanded reconstruction differs: " + name)
    if set(failures) != set(read_json(run / "failures.json")):
        raise ValueError("Failure identity differs")
    if new_ids != {p.name.removesuffix(".started.json") for p in (run / "requests").glob("*.started.json")}:
        raise ValueError("Extra or incomplete new requests")
    if len(new_ids) > protocol["maximum_requests"]:
        raise ValueError("Request budget exceeded")
    labels = {cid: [] for cid in protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    parent = read_json(Path(protocol["parent"]) / "predictions.json")
    parent.update(predictions)
    results, summary, _ = score_target(labels, observed, weak, parent, used, receipts)
    if results != read_json(run / "analysis-v1/results.json") or summary != read_json(run / "analysis-v1/summary.json"):
        raise ValueError("Full-data unified score differs")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "verification.json", dict(
        status="VERIFIED", new_api_calls=0, requests_verified=len(receipts), new_requests_verified=len(new_ids),
        models=len(parent), outputs=len(parent) * len(labels), failures=len(failures),
        exact_requests_images_prior_selection_conflicts_authority_stage_outputs_scores=True,
        prediction_manifest_sha256=sha(run / "prediction-freeze.json"),
        analysis_manifest_sha256=sha(run / "analysis-v1/freeze.json"),
        reserved_content_loaded=False, independent_effectiveness_established=False,
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
