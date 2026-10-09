"""Apply a recorded, minimal portability patch and run the real TAMA CLI on development plots."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generated/landslide-v1-20261007-075217-7b9f8f2c"


def patch_upstream(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {
            "BigModel.Azure", "BigModel.GLM", "BigModel.Claude", "BigModel.Groq",
            "BigModel.QWen", "BigModel.Gemini",
        }:
            replacement = "from local_backend import LocalBackend\n" if node.module == "BigModel.Azure" else ""
            replacements.append((node.lineno - 1, node.end_lineno, replacement))
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"ChatBotMAP", "JSON_whitelist"}
            for target in node.targets
        ):
            name = node.targets[0].id
            replacement = "ChatBotMAP = {'local': LocalBackend}\n"
            if name == "JSON_whitelist":
                replacement = "JSON_whitelist = ['local']\n"
            replacements.append((node.lineno - 1, node.end_lineno, replacement))
        if isinstance(node, ast.ClassDef) and node.name == "NormalReferenceHelper":
            for function in node.body:
                if isinstance(function, ast.FunctionDef) and function.name == "find_normal_reference":
                    fallback = function.body[0].orelse
                    replacements.append((fallback[0].lineno - 1, fallback[-1].end_lineno,
                                         "            raise RuntimeError('Training references required')\n"))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    adapted = "".join(lines)
    substitutions = {
        "'/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log'": "os.environ['TAMA_LOG_DIR']",
        "'/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output'": "os.environ['TAMA_OUTPUT_DIR']",
        "max_retry:int=6": "max_retry:int=0",
        "The vertical axis represents the time series index.":
            "The horizontal axis represents the time series index.",
        "The horizontal axis represents the value of the time series.":
            "The vertical axis represents the value of the time series.",
        "print(f'Earned: {money:.2f} USD')": "print('No billing estimate; see token receipts.')",
        "ax.set_xticks(range(0, len(data_for_check)+1, xticks))":
            "ax.set_xlim(0, len(data_for_check)-1)\n                        ax.set_xticks(range(0, len(data_for_check), xticks))",
        "ax.set_xticklabels(range(xticks_start, xticks_end+1, xticks))":
            "ax.set_xticklabels(range(xticks_start, xticks_end, xticks))",
    }
    for before, after in substitutions.items():
        if before not in adapted:
            raise ValueError(f"Upstream patch anchor missing: {before}")
        adapted = adapted.replace(before, after)
    boundary_instruction = (
        "\nThe complete record has 1095 samples, indexed from 0 through 1094 inclusive. "
        "Local plots retain these original day indices. Never output indices outside that range. "
        "abnormal_index and corrected_abnormal_index must be JSON strings, including the string [] "
        "when empty; never output a JSON array for these fields.\n"
    )
    insertion = (
        f"normal_reference_prompt += {boundary_instruction!r}\n"
        f"anormaly_detection_prompt += {boundary_instruction!r}\n"
        f"double_check_prompt += {boundary_instruction!r}\n"
        "random.seed(20261007)\nnp.random.seed(20261007)\n\n"
    )
    adapted = adapted.replace("if __name__ == '__main__':", insertion + "if __name__ == '__main__':")
    ast.parse(adapted)
    return adapted


def plot_series(values: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.5), dpi=110)
    axis.plot(np.arange(len(values)), values, linewidth=0.7)
    axis.set_xlabel("Day index")
    axis.set_ylabel("Displacement (mm)")
    axis.set_xlim(0, len(values) - 1)
    axis.set_xticks(sorted({*range(0, len(values), 100), len(values) - 1}))
    finite = values[np.isfinite(values)]
    padding = max(float(np.ptp(finite)) * 0.15, 5)
    axis.set_ylim(float(finite.min()) - padding, float(finite.max()) + padding)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def prepare(work: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    original = work / "upstream/tama"
    adapted_root = output / "adapted"
    shutil.copytree(original, adapted_root)
    source_path = original / "main_cli.py"
    original_source = source_path.read_text(encoding="utf-8")
    adapted = patch_upstream(original_source)
    (adapted_root / "main_cli.py").write_text(adapted, encoding="utf-8")
    shutil.copyfile(ROOT / "scripts/tama_backend.py", adapted_root / "local_backend.py")
    shutil.copyfile(ROOT / "scripts/reproduction_contracts.py", adapted_root / "reproduction_contracts.py")
    loader_path = adapted_root / "Datasets/Dataset.py"
    loader_original = loader_path.read_text(encoding="utf-8")
    before = "return remove_padding(data[num_stride, :, ch])"
    if loader_original.count(before) != 1:
        raise ValueError("Unexpected upstream observation loader")
    loader_adapted = loader_original.replace(before, "return data[num_stride, :, ch]")
    loader_path.write_text(loader_adapted, encoding="utf-8")
    patch = "".join(difflib.unified_diff(original_source.splitlines(True), adapted.splitlines(True),
                                         fromfile="upstream/main_cli.py", tofile="adapted/main_cli.py"))
    patch += "".join(difflib.unified_diff(loader_original.splitlines(True), loader_adapted.splitlines(True),
                                         fromfile="upstream/Datasets/Dataset.py", tofile="adapted/Datasets/Dataset.py"))
    (output / "portability.patch").write_text(patch, encoding="utf-8")

    dataset_root = output / "data/GNSS"
    dataset_root.mkdir(parents=True)
    (dataset_root / "background.txt").write_text("Synthetic development GNSS observations.", encoding="utf-8")
    rng = np.random.default_rng(202610071)
    length = 1095
    references = [rng.normal(0, 1.2, length) for _ in range(3)]
    fixtures = {"stationary_control": rng.normal(0, 1.2, length)}
    sources = []
    for case_id in ("case_0003", "case_0009"):
        path = DATA / "cases" / case_id / "input.json"
        raw = path.read_bytes()
        case = json.loads(raw)
        fixtures[case_id] = np.array([row[0] if row is not None else np.nan
                                      for row in case["displacement_mm"]])
        sources.append({"case_id": case_id, "axis": "N", "sha256": hashlib.sha256(raw).hexdigest()})
    structure = {}
    for case_id, values in fixtures.items():
        test = dataset_root / case_id / "test"
        test.mkdir(parents=True)
        np.save(test / "data.npy", values.reshape(1, length, 1))
        # The original CLI expects a label file even at inference. It gets unknown sentinels only.
        np.save(test / "labels.npy", np.full((1, length, 1), -1, dtype=int))
        plot_series(values, test / "image/0-0.png")
        for index, reference in enumerate(references):
            plot_series(reference, dataset_root / case_id / "train/image" / f"{index}-0.png")
        structure[case_id] = {"normal": [], "abnormal": [f"{case_id}-0-0"]}
    (dataset_root / "test_structure.yaml").write_text(yaml.safe_dump(structure), encoding="utf-8")
    (output / "logs").mkdir()
    manifest = {"scope": "TAMA workflow portability pilot, not paper-score or stage reproduction",
                "fixtures": list(fixtures), "input_sources": sources, "axis": "N only",
                "normal_reference_source": "three independently generated stationary training fixtures",
                "test_labels": "all -1; no evaluation labels or label-selected references",
                "double_check": True, "max_requests": 9, "model": "read from local configuration",
                "patches": ["local provider and output paths", "disable automatic retries and formatter LLM",
                            "forbid test-selected normal references", "correct swapped axis descriptions",
                            "preserve internal missing-day indices", "omit upstream fixed-price estimate",
                            "explicit valid index limits and strict interval-string validation",
                            "fixed reference-selection seeds"],
                "upstream_main_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def execute(output: Path, env_file: Path) -> None:
    marker = output / "execution-started.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"no_automatic_resume": True}, stream)
    environment = os.environ.copy()
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"QWEN_BASE_URL", "QWEN_API_KEY", "QWEN_MODEL"}:
            environment[key.strip()] = value.strip().strip("\"'")
    for key in ("QWEN_BASE_URL", "QWEN_API_KEY", "QWEN_MODEL"):
        if not environment.get(key):
            raise ValueError(f"Missing local setting {key}")
    environment.update({"TAMA_LOG_DIR": str(output / "logs"),
                        "TAMA_OUTPUT_DIR": str(output / "data"),
                        "TAMA_REQUEST_DIR": str(output / "requests"),
                        "TAMA_MAX_REQUESTS": "9", "TAMA_SERIES_LENGTH": "1095", "MPLBACKEND": "Agg",
                        "PYTHONIOENCODING": "utf-8"})
    command = [sys.executable, "-u", str(output / "adapted/main_cli.py"), "--dataset", "GNSS",
               "--LLM", "local", "--normal_reference", "3", "--double_check"]
    start = time.monotonic()
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, env=environment, cwd=output / "adapted", stdout=log,
                                stderr=subprocess.STDOUT, timeout=1500, check=False)
    summary = {"exit_code": result.returncode, "seconds": time.monotonic() - start,
               "model": environment["QWEN_MODEL"], "status": "failed" if result.returncode else "completed"}
    (output / "execution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    if result.returncode:
        raise RuntimeError("TAMA pilot failed; inspect execution.log before a new run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/reproduction-2026-10-07")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.work.resolve(), args.out.resolve())
    else:
        if not (args.out / "manifest.json").exists():
            raise ValueError("Prepare and inspect the TAMA pilot before execution")
        execute(args.out.resolve(), args.env_file)
