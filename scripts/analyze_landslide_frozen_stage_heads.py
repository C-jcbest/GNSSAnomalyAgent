"""Replay immutable stage predictions and present support and paired disagreement."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_landslide_raw_adjudication import verify as verify_activity_run  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_baselines import observation_features, rule_stages  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_evaluation import ActivityPrediction  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    METHODS,
    MOTION_FEATURES,
    apply_stage_support,
    compile_stage_answer,
    numeric_stage_rows,
    render_stage_heads,
    stage_pair_counts,
    stage_prompt,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_trace import request_evidence  # noqa: E402
from gnss_sim.visual import _unique_object  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def verify_hashes(run, manifest):
    freeze = read_json(run / "inference-freeze.json")
    if sha(run / "inference-freeze.json") != manifest["inference_freeze_sha256"]:
        raise ValueError("Inference freeze changed")
    for name, digest in freeze["evidence_sha256"].items():
        if sha(run / name) != digest:
            raise ValueError(f"Inference evidence changed: {name}")
    for name, digest in manifest["source_sha256"].items():
        if sha(ROOT / name) != digest or sha(run / "source" / name) != digest:
            raise ValueError(f"Execution source changed: {name}")
    if sha(run / "models/stage-xgboost.json") != manifest["model_sha256"]:
        raise ValueError("Saved model changed")
    return freeze


def replay_visual(public, packet, case, review, rows, motion):
    if packet is None:
        return unclassified_rows(rows), unclassified_rows(rows), {}, None
    request_id = packet["request_id"]
    started_url = f"requests/{request_id}.started.json"
    response_url = f"requests/{request_id}.response.json"
    started = read_json(public / started_url)
    response = read_json(public / response_url)
    if started["prompt"] != packet["prompt"] or response["request_sha256"] != started["request_sha256"]:
        raise ValueError("Receipt differs from prepared request")
    image_hashes = {Path(name).name: sha(public / name) for name in packet["images"]}
    if list(started["image_sha256"].items()) != list(image_hashes.items()):
        raise ValueError("Image content/order differs from prepared request")
    images = request_evidence(started, public / "images", public)
    if len(images) != 7:
        raise ValueError("Exactly seven frozen input images required")
    error = None
    if response["status"] == "returned_json":
        try:
            payload = json.loads(response["output"], object_pairs_hook=_unique_object)
            stage_review, native, final, suppressed = compile_stage_answer(
                payload, case, review, rows, packet["activity_sha256"], motion)
        except (ValueError, TypeError, KeyError) as failure:
            native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
            error = {"type": type(failure).__name__, "message": str(failure)[:400]}
        else:
            if stage_review != read_json(public / f"stage-reviews/{case.case_id}.json"):
                raise ValueError("Saved stage draft differs from response replay")
    else:
        native, final, suppressed = unclassified_rows(rows), unclassified_rows(rows), {}
        error = {"type": "RuntimeError", "message": f"Visual request failed: {request_id}; see receipt"}
    request = {"started": started, "response": response, "started_url": started_url,
               "response_url": response_url, "images": packet["images"], "replay_error": error,
               "receipt_summary": {key: response.get(key) for key in (
                   "request_id", "model", "returned_model", "temperature", "top_p", "max_tokens",
                   "no_retries", "status", "finish_reason", "http_status", "usage", "seconds", "request_sha256")}}
    return native, final, suppressed, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "artifacts/landslide-frozen-stage-heads-2026-10-08/run-v1")
    parser.add_argument("--findings", type=Path, help="Separate post-inference author observations, never stage truth")
    args = parser.parse_args()
    run, public = args.run, args.run / "public"
    manifest = read_json(run / "manifest.json")
    freeze = verify_hashes(run, manifest)
    config = manifest["config"]
    verify_activity_run(ROOT / config["activity_run"])
    ledger = read_json(run / "run-ledger-start.json")
    if any(sha(public / name) != expected for name, expected in ledger["prepared_inputs_sha256"].items()):
        raise ValueError("Prepared inputs changed")
    source = read_json(public / "source-manifest.json")
    author = ROOT / config["activity_run"]
    author_entries = {item["case_id"]: item for item in read_json(author / "public/manifest.json")["cases"]}
    packets = {p["case_id"]: p for p in read_json(public / "packets.json")}
    summary = read_json(public / "results.json")
    results = {r["case_id"]: r for r in summary["results"]}
    model = XGBClassifier()
    model.load_model(run / "models/stage-xgboost.json")
    contract = read_json(public / "feature-contract.json")
    plot = read_json(ROOT / config["observation_packet"] / "protocol.json")["plot"]
    activity_freeze = read_json(public / "activity-freeze.json")
    heads = {method: {"native_activity_features": Counter(), "final_activity_features": Counter(),
                      "suppressed_support": Counter(), "statuses": Counter()} for method in METHODS}
    pair_totals = {f"{a}_vs_{b}": Counter() for a, b in combinations(METHODS, 2)}
    all_csv, cases, plot_audits = [], [], {}
    for record in source["cases"]:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        entry = author_entries[case_id]
        review = read_json(public / "activity-source" / entry["review"])
        rows = read_json(public / "activity-source" / entry["daily"])
        frozen_hash = activity_freeze["reviews"][case_id]["activity_sha256"]
        activity = validate_activity(case, review, rows, frozen_hash)
        diagnostics = {w: LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"][str(w)]["data"]).read_bytes()) for w in (31, 61, 91)}
        if any(d.case_id != case_id or d.input_sha256 != record["input_sha256"] or d.window_days != w
               for w, d in diagnostics.items()):
            raise ValueError("Auxiliary provenance changed")
        features = observation_features(case, diagnostics)
        if features.names != contract["names"]:
            raise ValueError("Feature contract changed")
        numeric = {"rule": rule_stages(features, activity, config["rule_relative_change_60d"])}
        stage = model.predict(features.matrix).astype(int) + 1
        stage[(activity != 1) | ~features.supported] = -1
        stage[activity == 0] = 0
        numeric["xgboost"] = ActivityPrediction(activity.copy(), stage)
        native, final, suppressed = {}, {}, {}
        for method in ("rule", "xgboost"):
            native[method] = numeric_stage_rows(rows, numeric[method])
            final[method], suppressed[method] = apply_stage_support(native[method], rows, case, diagnostics[61])
        packet = packets.get(case_id)
        if packet is not None and packet["prompt"] != stage_prompt(
                case_id, review, frozen_hash, packet["images"], len(case.dates)):
            raise ValueError("Stage prompt changed")
        native["visual"], final["visual"], suppressed["visual"], request = replay_visual(
            public, packet, case, review, rows, diagnostics[61])
        result = results[case_id]
        if request and request["replay_error"] != result["visual_error"]:
            raise ValueError("Failure replay differs")
        for method in METHODS:
            if native[method] != read_json(public / f"native/{case_id}-{method}.json"):
                raise ValueError("Native stage replay differs")
            if final[method] != read_json(public / f"daily/{case_id}-{method}.json"):
                raise ValueError("Final stage replay differs")
            counts = {"native_activity_features": dict(Counter(r["feature"] for r in native[method]
                                                               if r["activity_label"] == 1)),
                      "final_activity_features": dict(Counter(r["feature"] for r in final[method]
                                                              if r["activity_label"] == 1)),
                      "suppressed_support": suppressed[method]}
            if any(counts[k] != result["heads"][method][k] for k in counts):
                raise ValueError("Head counts differ")
            for key in counts:
                heads[method][key].update(counts[key])
            heads[method]["statuses"].update([result["heads"][method]["status"]])
            for day, row in enumerate(final[method]):
                if row["feature"] in MOTION_FEATURES and (row["activity_label"] != 1
                                                         or case.displacement_mm[day] is None):
                    raise ValueError("Definite stage outside observed activity")
                if case.displacement_mm[day] is None and row["feature"] != "unknown":
                    raise ValueError("Missing day stage must remain unknown")
                all_csv.append({"method": method, **row})
        for a, b in combinations(METHODS, 2):
            name = f"{a}_vs_{b}"
            pair = stage_pair_counts(final[a], final[b])
            if pair != result["pairs"][name]:
                raise ValueError("Pair counts differ")
            pair_totals[name].update(pair["transitions"])
        plots = {mode: [] for mode in ("native", "final")}
        for mode, labels in (("native", native), ("final", final)):
            for index, view in enumerate(record["views"]):
                relative = f"presentation/{case_id}-{mode}-{index}.png"
                target = public / relative
                target.parent.mkdir(exist_ok=True)
                audit = render_stage_heads(case, view, labels, target, plot)
                if audit["xlim"] != view["audit"]["panels"][0]["xlim"]:
                    raise ValueError("Rendered raw x-axis changed")
                if audit["ylim_NEU"] != [p["ylim"] for p in view["audit"]["panels"]]:
                    raise ValueError("Rendered raw y-axes changed")
                codes = {"unknown": 0, "none": 1, "acceleration": 2, "steady_motion": 3, "deceleration": 4}
                for number, method in enumerate(METHODS, 1):
                    expected = [5 if case.displacement_mm[d] is None else codes[r["feature"]]
                                for d, r in enumerate(labels[method])]
                    if audit["label_colors"][number] != expected:
                        raise ValueError("Rendered strip differs from daily export")
                plot_audits[relative] = {**audit, "sha256": sha(target)}
                plots[mode].append(relative)
        sources = [{"label": "冻结活动复核", "url": "activity-source/" + entry["review"]},
                   {"label": "冻结活动逐日", "url": "activity-source/" + entry["daily"]}]
        for method in METHODS:
            sources += [{"label": method + " 原生", "url": f"native/{case_id}-{method}.json"},
                        {"label": method + " 最终", "url": f"daily/{case_id}-{method}.json"}]
        cases.append({"case_id": case_id, "result": result, "activity": review,
                      "plots": plots, "sources": sources, "request": request})
    with (public / "daily-stage-heads.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_csv = list(csv.DictReader(stream))
    expected_csv = [{key: str(value) for key, value in row.items()} for row in all_csv]
    if saved_csv != expected_csv or len(saved_csv) != 39420:
        raise ValueError("Combined stage CSV differs from prediction replay")
    starts = list((public / "requests").glob("*.started.json"))
    receipts = list((public / "requests").glob("*.response.json"))
    if len(starts) != len(receipts) or len(starts) != len(packets) or len(packets) != 9:
        raise ValueError("Fixed nine-call request budget differs")
    pairs = {}
    for name, transitions in pair_totals.items():
        common = sum(n for k, n in transitions.items() if all(f in MOTION_FEATURES for f in k.split("->")))
        same = sum(n for k, n in transitions.items() if k.split("->")[0] == k.split("->")[1]
                   and k.split("->")[0] in MOTION_FEATURES)
        pairs[name] = {"transitions": dict(transitions), "common_classified_days": common,
                       "common_same_class_days": same, "common_class_disagreement_days": common - same,
                       "meaning": "Agreement, NOT accuracy"}
    analysis = {"protocol_id": config["protocol_id"], "active_days": sum(r["active_days"] for r in results.values()),
                "heads": heads, "pairs": pairs, "csv_rows_replayed": len(saved_csv),
                "request_bodies_replayed": 9, "image_associations_replayed": 63,
                "inference_files_verified": len(freeze["evidence_sha256"]), "plots_verified": len(plot_audits),
                "definite_stages_outside_observed_activity": 0, "independent_stage_reference": False,
                "performance_scores": None, "sealed_test_read": False}
    findings = read_json(args.findings) if args.findings else []
    for finding in findings:
        if finding["case_id"] not in results or not finding["limit"]:
            raise ValueError("Author finding requires a known case and explicit limit")
        for image in finding["images"]:
            if "public/" + image not in freeze["evidence_sha256"]:
                raise ValueError("Author finding image is not frozen inference evidence")
    write_json(public / "analysis.json", analysis)
    write_json(public / "presentation/plot-audits.json", plot_audits)
    page_data = {"summary": summary, "analysis": analysis, "cases": cases, "findings": findings}
    template = (ROOT / "scripts/landslide_stage_heads_page.html").read_text(encoding="utf-8")
    encoded = json.dumps(page_data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    (public / "index.html").write_text(template.replace("__PAGE_DATA__", encoded), encoding="utf-8")
    verify_hashes(run, manifest)
    verify_activity_run(author)
    write_json(run / "presentation-audit.json", {
        "inference_unchanged": True, "activity_unchanged": True,
        "analysis_sha256": sha(public / "analysis.json"), "page_sha256": sha(public / "index.html"),
        "author_findings_sha256": sha(args.findings) if args.findings else None,
        "plot_audits_sha256": sha(public / "presentation/plot-audits.json")})
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
