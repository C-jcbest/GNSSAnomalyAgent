"""Score saved responses with the unchanged author's parser and vector conversion.

This is a disclosed post-registration analysis, never a replacement of the
frozen strict-parser run. No model calls, prompt edits or threshold search occur.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

import numpy as np

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate, evaluate_record, paired_bootstrap
from gnss_research.schema import observed_values
from gnss_research.upstream import ROOT, setup_sources


def load_author_functions() -> tuple[dict, dict]:
    source = setup_sources()["visualtime"]
    namespace = {"np": np}
    hashes = {}
    for filename, names in (
        ("result_agg.py", {"parse_output"}),
        ("utils.py", {"point_to_vector", "interval_to_vector"}),
    ):
        path = ROOT / "vendor/upstream/visualtime/src" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError("locked author functions missing")
        # Execute the exact AST, avoiding unrelated dataset/affiliation imports.
        module = ast.Module(body=functions, type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        hashes[filename] = sha256(path)
    return namespace, {"commit": source["commit"], "files": hashes,
                       "loading": "unchanged function AST from locked author source"}


def replay(run: Path, out: Path, dataset_override: Path | None = None) -> None:
    config = read_json(run / "registration.json")
    if not (run / "execution-complete.json").exists():
        raise ValueError("wait for the frozen run to finish")
    if "visualtime_gnss" not in config["methods"]:
        raise ValueError("run has no author responses")
    dataset = dataset_override or Path(config["dataset"])
    manifest = verify_dataset(dataset)
    if sha256(dataset / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("registered dataset changed")
    functions, source = load_author_functions()
    if source["commit"] != config["upstream"]["visualtime"]["commit"]:
        raise ValueError("author commit differs from registration")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "analysis.py")
    reports = []
    failures = 0
    subrequest_failures = 0
    for record in manifest["records"]:
        if record["station_id"] not in config["station_ids"]:
            continue
        observation = read_json(dataset / record["observation"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        work = run / "predictions/visualtime_gnss" / record["station_id"]
        mask = np.zeros(len(observed), dtype=bool)
        receipts = []
        for axis in "NEU":
            for task in ("point", "range"):
                request = work / f"{axis}-{task}"
                receipt = {"axis": axis, "task": task, "status": "failed"}
                try:
                    original = read_json(request / "receipt.json")
                    if original.get("http_status") != 200:
                        raise ValueError("provider response unsuccessful")
                    response = read_json(request / "response.json")
                    content = response["choices"][0]["message"]["content"]
                    parsed = functions["parse_output"](
                        content, "global" if task == "point" else "trend")
                    if task == "point":
                        vector = functions["point_to_vector"](parsed, len_vector=len(observed))
                    else:
                        vector = functions["interval_to_vector"](
                            parsed, start=0, end=len(observed), pred=True)
                    mask |= vector.astype(bool)
                    receipt.update(status="success", parsed=parsed,
                                   response_sha256=sha256(request / "response.json"))
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    receipt["error_type"] = type(error).__name__
                receipts.append(receipt)
        failed = any(receipt["status"] != "success" for receipt in receipts)
        failures += failed
        subrequest_failures += sum(receipt["status"] != "success" for receipt in receipts)
        # Retain the frozen all-six-success policy; don't reward partial failures.
        if failed:
            mask[:] = False
        mask &= observed
        result = {"status": "failed" if failed else "success", "prediction": None,
                  "generic_mask": mask.tolist(), "generic_score": None,
                  "request_count": 0, "subrequests": receipts}
        write_json(out / "predictions" / f"{record['station_id']}.json", result)
        reports.append(evaluate_record(read_json(dataset / record["truth"]), result,
                                       observed, observation["background_end"]))
    summary = aggregate(reports)
    summary.update(failed_records=failures, not_run_records=0)
    write_json(out / "per-station.json", reports)
    analysis = {"method": "visualtime_gnss_author_parser", "source": source,
                "source_run": str(run.resolve()),
                "registration_sha256": sha256(run / "registration.json"),
                "analysis_script_sha256": sha256(Path(__file__)),
                "new_requests": 0, "failed_subrequests": subrequest_failures,
                "methods": {"visualtime_gnss_author_parser": summary},
                "disclosure": ["post-registration offline analysis, same saved responses",
                               "original parser bracket extraction/truncation repair unchanged",
                               "original vector clipping/skipping behavior unchanged",
                               "NEU point/range union and missing mask remain GNSS adaptations",
                               "all six calls must parse successfully; otherwise zero detections",
                               "strict-parser results retained separately; not paper reproduction"]}
    original_reports = read_json(run / "evaluation/per-station.json")
    if "fusion" in original_reports:
        analysis["paired_comparisons"] = {"fusion_minus_visualtime_gnss_author_parser":
            paired_bootstrap(original_reports["fusion"], reports, config["seed"])}
    write_json(out / "summary.json", analysis)
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    replay(args.run, args.out, args.dataset)
