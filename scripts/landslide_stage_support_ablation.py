"""Freeze support variants before loading the disclosed development reference."""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_landslide_frozen_stage_heads import verify_hashes
from analyze_landslide_stage_agreement import load_inputs
from audit_landslide_window_support import utc_now
from landslide_stage_evidence import load_case
from landslide_stage_grounding import verify_prepared
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_stage_agreement import stage_agreement, support_filter_effect
from gnss_sim.landslide_stage_support_ablation import (
    CONDITIONS,
    baseline_changes,
    build_support_variants,
    coverage_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/landslide-stage-support-ablation-2026-10-09/run-v2"
CONFIG = ROOT / "configs/landslide-stage-support-ablation-v1.json"
SOURCE_FILES = (
    "configs/landslide-stage-support-ablation-v1.json", "docs/landslide-stage-support-ablation-protocol.md",
    "src/gnss_sim/landslide_stage_support_ablation.py", "tests/test_landslide_stage_support_ablation.py",
    "scripts/landslide_stage_support_ablation.py", "scripts/landslide_stage_support_ablation_page.html",
    "scripts/verify_landslide_stage_support_ablation.py", "scripts/check_landslide_stage_support_ablation_ui.cjs",
    "src/gnss_sim/landslide_frozen_stage_heads.py", "src/gnss_sim/landslide_stage_agreement.py",
    "src/gnss_sim/landslide_evaluation.py", "scripts/analyze_landslide_stage_agreement.py",
    "scripts/verify_landslide_stage_agreement.py",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_digest(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Frozen bytes changed: {path}")


def render_comparison(case, review, reference, native, outputs, method, path):
    figure = Figure(figsize=(15, 8), dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[2,1.5]})
    values = np.array([value if value is not None else [np.nan]*3 for value in case.displacement_mm])
    days = np.arange(len(values))
    for axis,color in enumerate(("#147d72","#cc6d4b","#a38a32")):
        panels[0].plot(days,values[:,axis],color=color,linewidth=.7,label=("N","E","U")[axis])
    for span in review["spans"]:
        if span["state"] == "activity":
            panels[0].axvspan(span["start"]-.5,span["stop"]-.5,color="#147d72",alpha=.08)
    panels[0].margins(y=.15)
    panels[0].legend(ncols=3,loc="upper left")
    panels[0].set_ylabel("Observed displacement (mm)")
    panels[0].set_title(f"{case.case_id} / {method} / fixed native output; support masks only / NOT new model input")
    bands = [reference, native, *[outputs[condition] for condition in CONDITIONS]]
    codes = {"none":0,"unknown":1,"acceleration":2,"steady_motion":3,"deceleration":4}
    encoded = []
    for band in bands:
        encoded.append([5 if case.displacement_mm[day] is None else codes[row["feature"]]
                        for day,row in enumerate(band)])
    colors = ListedColormap(["#f0f2ed","#bac3c8","#d7774d","#d3b859","#237f73","#26343d"])
    panels[1].imshow(encoded,aspect="auto",interpolation="nearest",origin="upper",
        extent=(-.5,len(days)-.5,len(bands)-.5,-.5),cmap=colors,
        norm=BoundaryNorm(np.arange(-.5,6),colors.N))
    panels[1].set_yticks(range(7),["AI dev reference","native",*CONDITIONS],fontsize=9)
    panels[1].set_xlabel("Day index | grey unknown; orange A; gold S; green D; black missing")
    panels[0].set_xlim(-.5,len(days)-.5)
    panels[0].grid(alpha=.15)
    figure.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path)


def prepare_predictions(config):
    """No reference loading in this phase; baseline equality precedes all scoring."""
    _, _, records, entries = verify_prepared()
    audit = ROOT / config["support_audit_run"]
    native_run = ROOT / config["native_prediction_run"]
    check_digest(audit / "audit-freeze.json", config["support_audit_freeze_sha256"])
    check_digest(native_run / "inference-freeze.json",config["native_inference_freeze_sha256"])
    audit_ledger = read_json(audit / "run-ledger-start.json")
    audit_freeze = read_json(audit / "audit-freeze.json")
    for name,digest in audit_freeze["files_sha256"].items():
        check_digest(audit / "public" / name,digest)
    for name,digest in audit_ledger["source_sha256"].items():
        check_digest(ROOT / name,digest)
        check_digest(audit / "source-code" / name,digest)
    verify_hashes(native_run,read_json(native_run / "manifest.json"))
    for item in read_json(native_run / "delivery-audit.json")["files"]:
        check_digest(native_run / "public" / item["path"],item["sha256"])
    registered = read_json(audit / "run-ledger-end.json")["additional_document_sha256"]
    for name in ("configs/landslide-stage-support-ablation-v1.json","docs/landslide-stage-support-ablation-protocol.md"):
        check_digest(ROOT / name,registered[name])
    RUN.mkdir(parents=True,exist_ok=False)
    public = RUN / "public"
    public.mkdir()
    source_hashes = {}
    for name in SOURCE_FILES:
        source_hashes[name] = sha(ROOT / name)
        target = RUN / "source-code" / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT / name,target)
    inputs = {}
    for name,digest in audit_ledger["input_sha256"].items():
        source = audit / "public" / name
        check_digest(source,digest)
        target = public / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        inputs[source.relative_to(ROOT).as_posix()] = digest
    native_manifest = read_json(native_run / "public/source-manifest.json")
    if [record["case_id"] for record in records] != [record["case_id"] for record in native_manifest["cases"]]:
        raise ValueError("Native source record set changed")
    contexts, all_predictions, csv_rows = [], [], []
    native_activity = read_json(native_run / "public/activity-freeze.json")
    native_records = {record["case_id"]:record for record in native_manifest["cases"]}
    for record in records:
        case,review,activity_rows,digest,motions = load_case(public,record,entries)
        old_record = native_records[case.case_id]
        if old_record["input_sha256"] != record["input_sha256"]:
            raise ValueError("Native and audit observation identities differ")
        if native_activity["reviews"][case.case_id]["activity_sha256"] != digest:
            raise ValueError("Native activity differs from the audited activity")
        for name in (record["input"], *[record["diagnostics"][str(window)]["data"] for window in (31,61,91)]):
            original = native_run / "public" / name
            if sha(original) != sha(public / "source" / name):
                raise ValueError("Native and audited calendar/cache bytes differ")
            inputs[original.relative_to(ROOT).as_posix()] = sha(original)
        entry = entries[case.case_id]
        for key in ("daily","review"):
            original = native_run / "public/activity-source" / entry[key]
            if sha(original) != sha(public / "source/activity-source" / entry[key]):
                raise ValueError("Native and audited raw activity bytes differ")
            inputs[original.relative_to(ROOT).as_posix()] = sha(original)
        context = {"case":case,"review":review,"rows":activity_rows,"motions":motions,
                   "record":record,"activity_sha256":digest,"methods":{}}
        for method in config["methods"]:
            native_path = native_run / f"public/native/{case.case_id}-{method}.json"
            final_path = native_run / f"public/daily/{case.case_id}-{method}.json"
            native = read_json(native_path)
            outputs,traces = build_support_variants(case,review,activity_rows,native,motions)
            if outputs["fixed_61"] != read_json(final_path):
                raise ValueError("61-day baseline did not exactly reproduce the frozen final")
            for source in (native_path,final_path):
                inputs[source.relative_to(ROOT).as_posix()] = sha(source)
            write_json(public / f"native/{case.case_id}-{method}.json",native)
            context["methods"][method] = {"native":native,"outputs":outputs,"traces":traces}
            for condition in CONDITIONS:
                output = outputs[condition]
                trace = traces[condition]
                name = f"daily/{case.case_id}-{method}-{condition}.json"
                write_json(public / name,output)
                write_json(public / f"support-trace/{case.case_id}-{method}-{condition}.json",trace)
                inventory = coverage_inventory(native,output,trace)
                all_predictions.append({"case_id":case.case_id,"method":method,"condition":condition,**inventory})
                for row,evidence in zip(output,trace,strict=True):
                    csv_rows.append({"method":method,"condition":condition,**row,**evidence})
        contexts.append(context)
        print(f"{len(contexts)}/12 {case.case_id}: all 15 support outputs prepared; baseline equal",flush=True)
    write_json(RUN / "run-ledger-start.json", {
        "prepared_utc":utc_now(),"source_sha256":source_hashes,"input_sha256":inputs,
        "audit_freeze_sha256":sha(audit / "audit-freeze.json"),
        "native_freeze_sha256":sha(native_run / "inference-freeze.json"),
        "phase":"all_predictions_prepared_before_reference_loading","model_calls":0,
        "stage_reference_loaded_by_execution":False,"prior_material_exposure":True,
    })
    with (public / "daily-support-predictions.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(public / "prediction-inventory.json",all_predictions)
    names = [path for directory in ("native","daily","support-trace") for path in (public / directory).glob("*.json")]
    names.extend([public / "daily-support-predictions.csv",public / "prediction-inventory.json"])
    write_json(RUN / "prediction-freeze.json",{
        "frozen_utc":utc_now(),"source_sha256":source_hashes,"input_sha256":inputs,
        "files_sha256":{path.relative_to(public).as_posix():sha(path) for path in names},
        "baseline_exact_cases_methods":36,"prediction_outputs":180,"prediction_rows":len(csv_rows),
        "reference_loaded_by_execution":False,"model_calls":0,
    })
    return contexts


def analyze(config, contexts):
    analysis_started = time.monotonic()
    prediction_freeze = read_json(RUN / "prediction-freeze.json")
    public = RUN / "public"
    for name,digest in prediction_freeze["files_sha256"].items():
        check_digest(public / name,digest)
    # All 180 outputs have now been frozen. This is the first full reference load in execution.
    packet,reference,_,reference_inputs = load_inputs({
        "prediction_run":config["native_prediction_run"],"reference_run":config["reference_run"]})
    write_json(RUN / "reference-load-ledger.json",{
        "loaded_utc":utc_now(),"prediction_freeze_sha256":sha(RUN / "prediction-freeze.json"),
        "input_sha256":reference_inputs,"reference_use":"development_only",
        "execution_sequence":"prediction freeze precedes reference loading; not independent blinding",
    })
    write_json(public / "daily-reference.json",reference)
    write_json(public / "reference-review.json",packet)
    reference_by_case = {context["case"].case_id:[row for row in reference if row["case_id"] == context["case"].case_id]
                         for context in contexts}
    reference_cases = {case["case_id"]:case for case in packet["cases"]}
    totals = {method:{condition:[] for condition in CONDITIONS} for method in config["methods"]}
    natives = {method:[] for method in config["methods"]}
    page_cases,csv_rows = [],[]
    native_run = ROOT / config["native_prediction_run"]
    native_packets = {packet["case_id"]:packet for packet in read_json(native_run / "public/packets.json")}
    origin_images = set()
    for context in contexts:
        if time.monotonic()-analysis_started > 180:
            raise TimeoutError("Support analysis exceeded 180 seconds")
        case,review,record = context["case"],context["review"],context["record"]
        actual = reference_by_case[case.case_id]
        authored = reference_cases[case.case_id]
        if authored["input_sha256"] != record["input_sha256"] or authored["activity_sha256"] != context["activity_sha256"]:
            raise ValueError("Reference and support experiment identities differ")
        page = {"case_id":case.case_id,"summary":{},"interiors":[],"plots":{},
                "raw_images":[view["image"] for view in record["views"]],
                "aux_images":[record["diagnostics"][str(window)]["image"] for window in (31,61,91)],
                "native_scores":{},"visual_request":None}
        for method in config["methods"]:
            entry = context["methods"][method]
            native = entry["native"]
            natives[method].extend(native)
            page["native_scores"][method] = stage_agreement(actual,native)
            page["summary"][method] = {}
            for condition in CONDITIONS:
                output = entry["outputs"][condition]
                totals[method][condition].extend(output)
                score = stage_agreement(actual,output)
                page["summary"][method][condition] = {
                    "scores":score,"coverage":coverage_inventory(native,output,entry["traces"][condition]),
                    "support_effect":support_filter_effect(actual,native,output),
                    "baseline_changes":baseline_changes(actual,entry["outputs"]["fixed_61"],output),
                }
                for row,evidence,ref in zip(output,entry["traces"][condition],actual,strict=True):
                    csv_rows.append({"method":method,"condition":condition,**row,**evidence,
                                     "reference_feature":ref["feature"],"reference_evaluable":ref["stage_evaluable"]})
            plot = f"comparison-images/{case.case_id}-{method}.png"
            render_comparison(case,review,actual,native,entry["outputs"],method,public / plot)
            page["plots"][method] = plot
        for interior in authored["interiors"]:
            start,stop = interior["start"],interior["stop"]
            view = {"start":start,"stop":stop,"methods":{}}
            for method,entry in context["methods"].items():
                view["methods"][method] = {
                    condition:{"scores":stage_agreement(actual[start:stop],entry["outputs"][condition][start:stop]),
                        "coverage":coverage_inventory(entry["native"][start:stop],entry["outputs"][condition][start:stop],entry["traces"][condition][start:stop]),
                        "baseline_changes":baseline_changes(actual[start:stop],entry["outputs"]["fixed_61"][start:stop],entry["outputs"][condition][start:stop])}
                    for condition in CONDITIONS}
            page["interiors"].append(view)
        if case.case_id in native_packets:
            request = native_packets[case.case_id]
            request_started = read_json(native_run / f"public/requests/{request['request_id']}.started.json")
            response = read_json(native_run / f"public/requests/{request['request_id']}.response.json")
            if request_started["prompt"] != request["prompt"]:
                raise ValueError("Native request prompt provenance mismatch")
            page["visual_request"] = {"started":request_started,"response":response,
                "images":["native-origin/"+name for name in request["images"]],"historical_not_new":True}
            for name in request["images"]:
                origin_images.add(name)
        page_cases.append(page)
    for name in origin_images:
        destination = public / "native-origin" / name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(native_run / "public" / name,destination)
    summary = {"protocol_id":config["protocol_id"],"conditions":list(CONDITIONS),"methods":{},
        "native_scores":{method:stage_agreement(reference,native) for method,native in natives.items()},
        "reference_reviewer":packet["reviewer"],"reference_use":"development_only",
        "formal_reference_verified":False,"performance_scores":None,"model_calls":0,
        "prediction_rows":len(csv_rows),"cases":12,"interiors":21,
        "reference_classes":dict(Counter(row["feature"] for row in reference if row["stage_evaluable"])),
        "prediction_freeze_sha256":sha(RUN / "prediction-freeze.json")}
    for method in config["methods"]:
        summary["methods"][method] = {condition:{
            "scores":stage_agreement(reference,output),
            "support_effect":support_filter_effect(reference,natives[method],output),
            "baseline_changes":baseline_changes(reference,totals[method]["fixed_61"],output),
            "coverage":{key:sum(case["summary"][method][condition]["coverage"][key] for case in page_cases)
                        for key in ("observed_activity_days","native_definite_days","native_unknown_days",
                                    "final_definite_days","suppressed_native_days","selected_cross_boundary_days")},
        } for condition,output in totals[method].items()}
    write_json(public / "analysis.json",{"summary":summary,"cases":page_cases})
    with (public / "daily-stage-support.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    payload = json.dumps({"summary":summary,"cases":page_cases},ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    template = (ROOT / "scripts/landslide_stage_support_ablation_page.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(template.replace("__PAGE_DATA__",payload),encoding="utf-8")
    for name,digest in prediction_freeze["files_sha256"].items():
        check_digest(public / name,digest)
    write_json(RUN / "analysis-freeze.json",{
        "completed_utc":utc_now(),"prediction_freeze_sha256":sha(RUN / "prediction-freeze.json"),
        "reference_load_ledger_sha256":sha(RUN / "reference-load-ledger.json"),
        "public_files_sha256":{path.relative_to(public).as_posix():sha(path) for path in sorted(public.rglob("*")) if path.is_file()},
        "actual_new_calls":0,"main_configuration_replaced":False,
    })
    print(json.dumps({method:{condition:{key:value["scores"][key] for key in
        ("exact_agreement_days","different_class_days","abstained_evaluable_days","development_agreement_macro_f1")}
        for condition,value in configurations.items()} for method,configurations in summary["methods"].items()},indent=2))


def run():
    started = time.monotonic()
    config = read_json(CONFIG)
    if tuple(condition["id"] for condition in config["conditions"]) != CONDITIONS:
        raise ValueError("Registered support conditions changed")
    contexts = prepare_predictions(config)
    if time.monotonic()-started > 180:
        raise TimeoutError("Support preparation exceeded 180 seconds")
    analyze(config,contexts)
    print(f"Completed offline support ablation in {time.monotonic()-started:.2f}s",flush=True)


if __name__ == "__main__":
    run()
