"""Immutable author code and explicitly named GNSS wrappers."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, write_json
from gnss_research.plots import render_observation
from gnss_research.provider import send_request
from gnss_research.schema import observed_values

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_METHODS = ("visualtime_gnss",)


def setup_sources() -> dict:
    lock = read_json(ROOT / "configs/upstream.json")
    receipts = {}
    for name, source in lock.items():
        checkout = ROOT / "vendor/upstream" / name
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-c", "http.sslBackend=schannel", "clone", "--no-checkout",
                            source["repository"], str(checkout)], check=True, timeout=120)
            subprocess.run(["git", "checkout", "--detach", source["commit"]], cwd=checkout,
                           check=True, timeout=60)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout,
                                         text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                         cwd=checkout, text=True).splitlines()
        # Importing author modules can generate bytecode; it does not change source files.
        changed = [line for line in status if not (
            line.startswith("?? ") and "/__pycache__/" in line and line.endswith(".pyc"))]
        if commit != source["commit"] or changed:
            raise ValueError(f"upstream checkout changed: {name}; never silently reset")
        receipts[name] = {**source, "source_files": {
            str(path.relative_to(checkout)): sha256(path)
            for path in sorted(checkout.rglob("*.py")) if ".git" not in path.parts}}
    return receipts


def prepare_visualtime(observation: dict, work: Path, model: str) -> None:
    receipts = setup_sources()
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / "source.json", receipts["visualtime"])
    prompt_module = ROOT / "vendor/upstream/visualtime/src/prompt.py"
    # Run the ORIGINAL create_openai_request in an isolated process: its relative data paths
    # do not require changing this process's cwd, and all original defaults remain intact.
    script = (
        "import importlib.util,json,sys; sys.dont_write_bytecode=True; "
        "spec=importlib.util.spec_from_file_location('original_prompt',sys.argv[1]); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "r=m.create_openai_request(vision=True,"
        "data_tuple=('synthetic','univariate',None,1,0,sys.argv[2],0)); "
        "print(json.dumps(r))"
    )
    for axis in range(3):
        for task, data_name in (("point", "global"), ("range", "trend")):
            request_root = work / f"{'NEU'[axis]}-{task}"
            figure = request_root / "data/synthetic/univariate" / data_name / "eval/fig/000.png"
            render_observation(observation, figure, axes=(axis,), show_background=False)
            output = subprocess.check_output([sys.executable, "-c", script, str(prompt_module),
                                              data_name], cwd=request_root, text=True, timeout=30)
            payload = json.loads(output)
            payload.update({"model": model, "enable_thinking": False})
            write_json(request_root / "request.json", payload)
    write_json(work / "adaptation.json", {
        "name": "VisualTimeAnomaly original zero-shot source / GNSS adapted inputs",
        "original_functions": ["prompt.create_openai_request", "prompt.create_vision_messages"],
        "original_defaults": {"temperature": 0.4, "stop": "unchanged source default"},
        "changes": ["GNSS plots with retained original daily indices and missing gaps",
                    "N/E/U run separately, point and range tasks; union for generic scoring",
                    "existing qwen3.8-flash provider replaces original paper model",
                    "enable_thinking=False provider option", "strict output validation",
                    "no upstream retries; invalid output remains failed"],
        "threshold_adjustments": [], "paper_score_reproduction": False,
        "typed_deformation_score": "not available; source returns generic point/range anomalies"})


def parse_strict_list(content: str, task: str, length: int) -> np.ndarray:
    """Our strict GNSS contract; the author parser is a separate offline analysis."""
    value = json.loads(content)
    if not isinstance(value, list):
        raise ValueError("original prompt requires a JSON list")
    mask = np.zeros(length, dtype=bool)
    for entry in value:
        if task == "point":
            start = end = entry
        else:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("original range prompt requires inclusive index pairs")
            start, end = entry
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < length:
            raise ValueError("invalid original temporal indices; no silent clipping")
        mask[start:end + 1] = True
    return mask


def call_visualtime(observation: dict, work: Path, settings: tuple[str, str, str]) -> dict:
    length = len(observation["dates"])
    generic = np.zeros(length, dtype=bool)
    receipt = {"status": "success", "request_count": 0, "method": "visualtime_gnss",
               "prediction": None, "generic_score": None, "subrequests": []}
    for axis in "NEU":
        for task in ("point", "range"):
            request_root = work / f"{axis}-{task}"
            subreceipt, content = send_request(read_json(request_root / "request.json"),
                                                request_root, settings)
            if subreceipt["status"] == "success":
                try:
                    generic |= parse_strict_list(content, task, length)
                except (ValueError, TypeError) as error:
                    subreceipt.update({"status": "failed", "error_type": type(error).__name__})
            if subreceipt["status"] != "success":
                receipt["status"] = "failed"
            write_json(request_root / "receipt.json", subreceipt)
            receipt["request_count"] += subreceipt["request_count"]
            receipt["subrequests"].append(subreceipt)
    generic &= np.isfinite(observed_values(observation)).all(axis=1)
    receipt["generic_mask"] = generic.tolist()
    write_json(work / "result.json", receipt)
    return receipt
