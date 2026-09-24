import itertools
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .contracts import Prediction, Window
from .datasets import load_dataset
from .detectors import NAMES, NUMERICAL_NAMES, detect
from .diagnostics import runtime_diagnostics
from .evaluation import METRIC_VERSION, aggregate, counts
from .fusion import EVIDENCE_VERSION, FUSION_VERSION, merge_branches
from .models import PROMPT_VERSION, ModelClient, resolve_model
from .plotting import PLOT_VERSION
from .policies import METHODS, Executor, load_capabilities
from .storage import child, digest, file_hash, read_json, write_json


def source_identity():
    root = Path(__file__).parent
    files = {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "source_sha256": digest(files),
        "git_commit": result.stdout.strip() or None,
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pydantic", "httpx")
        },
    }


def validate_config(config: dict):
    if config.get("execution_order") not in (None, "paired-random-v1"):
        raise ValueError("unknown execution order")
    if not isinstance(config.get("workers", 1), int) or not 1 <= config.get("workers", 1) <= 4:
        raise ValueError("workers must be between 1 and 4")
    d = config["detectors"]
    if "reference_trend" in d:
        from .reference_trend import validate_reference_config

        validate_reference_config(d["reference_trend"])
        if d.get("sampling_hours") != 24 or d.get("max_interpolation_samples") != 0:
            raise ValueError("reference_trend requires daily data without interpolation")
    if d.get("implementation") not in (None, "daily-observed-v2"):
        raise ValueError("unknown detector implementation")
    if d.get("implementation") == "daily-observed-v2" and (
        d.get("sampling_hours") != 24
        or d.get("max_interpolation_samples") != 0
        or not 3 <= d.get("min_window_samples", 0) <= d["window"]
        or not isinstance(d.get("reset_gap_days"), int)
        or d["reset_gap_days"] < 0
    ):
        raise ValueError("invalid observed-only daily settings")
    if config["model"].get("attempts_per_call", 1) not in (1, 2):
        raise ValueError("attempts_per_call must be 1 or 2")
    if config["fixed_detector"] not in NAMES or config["repeats"] < 1:
        raise ValueError("invalid fixed detector or repeats")
    if d["window"] < 3 or d["window"] % 2 != 1 or d["min_samples"] < 3:
        raise ValueError("window must be odd >= 3; min_samples must be >= 3")
    if any(
        d[k] <= 0 for k in ("hampel_threshold", "cusum_threshold", "cusum_drift", "iforest_trees")
    ):
        raise ValueError("detector parameters must be positive")
    if not 0 < d["iforest_threshold"] < 1:
        raise ValueError("iforest threshold must be in (0, 1)")
    if not 1 <= config["model"]["max_requests"] <= 5:
        raise ValueError("model max_requests must be between 1 and 5")


def numerical_identity(config: dict):
    return digest({k: config[k] for k in ("detectors", "seed", "fixed_detector")})


def daily_config_identity(config: dict):
    return digest({key: value for key, value in config.items() if key != "calibration"})


def execution_schedule(case_id: str, methods: list[str], config: dict):
    if config.get("execution_order") == "paired-random-v1":
        rng = random.Random(f"{config['seed']}:{case_id}")
        schedule = []
        for repeat in range(config["repeats"]):
            block = [m for m in methods if repeat == 0 or m not in NUMERICAL_NAMES]
            rng.shuffle(block)
            schedule.extend((m, repeat) for m in block)
        return schedule
    return [
        (m, repeat)
        for m in methods
        for repeat in range(1 if m in NUMERICAL_NAMES else config["repeats"])
    ]


def run_experiment(
    dataset: Path,
    out: Path,
    config: dict,
    split: str,
    methods: list[str],
    resume=False,
    limit: int | None = None,
):
    validate_config(config)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("methods must be unique supported names")
    if "reference_trend" in methods and (
        split != "development" or "reference_trend" not in config["detectors"]
    ):
        raise ValueError("reference_trend is a configured development-only candidate")
    manifest = load_dataset(dataset)
    if manifest["kind"] == "daily_injection_unverified_v1":
        paired_methods = ["hampel", "cusum", "iforest", "visual", "daily_union"]
        if (
            manifest.get("sampling_hours") != 24
            or config["detectors"].get("implementation") != "daily-observed-v2"
            or config.get("visual_protocol") != "global-mask-v2"
        ):
            raise ValueError("daily benchmark requires observed-only daily and global visual input")
        if split == "test" and methods != paired_methods:
            raise ValueError("daily test requires all five paired methods")
        if "daily_union" in methods and (
            methods != paired_methods
            or config["repeats"] != 1
            or config.get("execution_order") is not None
            or config["fixed_detector"] != "hampel"
        ):
            raise ValueError("daily_union requires ordered, single-repeat paired baselines")
    elif "daily_union" in methods:
        raise ValueError("daily_union requires the frozen daily injection benchmark")
    if {"split_fusion", "split_fusion_evidence"} & set(methods) and (
        split != "development"
        or manifest.get("sampling_hours") != 24
        or config.get("visual_protocol") != "global-mask-v2"
        or config["fixed_detector"] != "hampel"
    ):
        raise ValueError("split_fusion requires daily global development protocol with Hampel")
    if manifest["kind"] == "platform_pilot_provisional" and (
        split != "development"
        or not set(methods)
        <= {*NUMERICAL_NAMES, "visual", "fixed", "split_fusion", "split_fusion_evidence"}
    ):
        raise ValueError("provisional pilot supports development baseline checks only")
    dataset_hash = file_hash(dataset / "manifest.json")
    if split == "test" and config.get("calibration", {}).get("dataset_sha256") != dataset_hash:
        raise ValueError(
            "test requires a configuration calibrated on this dataset's validation split"
        )
    if split == "test" and config["calibration"].get("source_split") != "validation":
        raise ValueError("calibration must come from validation")
    if split == "test" and config["calibration"].get("numerical_sha256") != numerical_identity(
        config
    ):
        raise ValueError("numerical configuration changed after validation calibration")
    if split == "test" and manifest["kind"] == "daily_injection_unverified_v1":
        if (
            config["calibration"].get("mode") != "fixed-without-label-selection"
            or config["calibration"].get("config_sha256") != daily_config_identity(config)
            or limit is not None
        ):
            raise ValueError(
                "test requires the full unchanged validation-frozen daily configuration"
            )
    records = [r for r in manifest["records"] if r["split"] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError("empty split")
    card = load_capabilities(config.get("capability_file"))
    if any(m == "rule" or m.startswith("lma") for m in methods):
        if not card or card.get("dataset_sha256") != dataset_hash:
            raise ValueError("this dataset's measured capability card is required")
        if card.get("detector_config") != config["detectors"] or card.get("seed") != config["seed"]:
            raise ValueError("detector configuration changed since capability measurement")
        if card.get("model_config") != config["model"] or card.get("model") != resolve_model(
            config["model"]
        ):
            raise ValueError("model configuration changed since capability measurement")
    identity = {
        "dataset_sha256": dataset_hash,
        "dataset_kind": manifest["kind"],
        "sampling_hours": manifest.get("sampling_hours", 1),
        "expected_case_ids": [r["case_id"] for r in records],
        "config": config,
        "capabilities": card,
        "split": split,
        "methods": methods,
        "limit": limit,
        "source": source_identity(),
        "model": resolve_model(config["model"]),
        "provider_hash": digest(os.getenv("QWEN_BASE_URL", "")),
        "prompt_version": PROMPT_VERSION,
        "plot_version": PLOT_VERSION,
        "metric_version": METRIC_VERSION,
    }
    if config.get("execution_order") == "paired-random-v1":
        identity["execution_schedule"] = {
            r["case_id"]: execution_schedule(r["case_id"], methods, config) for r in records
        }
    if {"split_fusion", "split_fusion_evidence"} & set(methods):
        identity["fusion_version"] = FUSION_VERSION
    if "split_fusion_evidence" in methods:
        identity["evidence_version"] = EVIDENCE_VERSION
    run_id = digest(identity)
    if out.exists():
        if not resume or read_json(out / "run.json")["run_id"] != run_id:
            raise ValueError("run exists or identity changed; use new output directory")
    else:
        out.mkdir(parents=True)
        write_json(
            out / "run.json",
            {
                "run_id": run_id,
                **identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(dataset.resolve()),
            },
        )
    labels = read_json(dataset / "labels.json")

    def process(record):
        model = (
            ModelClient(config["model"]) if any(m not in NUMERICAL_NAMES for m in methods) else None
        )
        executor = Executor(config, model, card)
        try:
            window = Window.model_validate(read_json(child(dataset, record["file"])))
            truth = Prediction(events=labels[record["case_id"]]["events"])
            paired = {}
            for schedule_index, (method, repeat) in enumerate(
                execution_schedule(record["case_id"], methods, config)
            ):
                result_path = out / "results" / f"{record['case_id']}_{method}_{repeat}.json"
                if result_path.exists():
                    if method in ("hampel", "visual"):
                        paired[method] = Prediction.model_validate(
                            read_json(result_path)["prediction"]
                        )
                    continue  # Atomic completed/error records are never silently re-sampled.
                if model:
                    model.reset()
                executor.tools = []
                started_at = datetime.now(timezone.utc).isoformat()
                start = time.perf_counter()
                error = None
                try:
                    if method == "daily_union":
                        prediction = merge_branches(
                            paired["hampel"], paired["visual"], len(window.values)
                        )
                        executor.tools = [
                            {
                                "tool": name,
                                "status": branch.status,
                                "result": branch.model_dump(),
                                "paired_repeat": repeat,
                            }
                            for name, branch in paired.items()
                        ]
                    else:
                        prediction = executor.run(window, method)
                    prediction.mask(len(window.values))
                except (ValueError, RuntimeError, KeyError, TypeError, ArithmeticError) as exc:
                    error = type(exc).__name__
                    prediction = Prediction(status="error", reason=str(exc)[:500])
                if method in ("hampel", "visual"):
                    paired[method] = prediction
                result = {
                    "run_id": run_id,
                    **record,
                    "method": method,
                    "repeat": repeat,
                    "prediction": prediction.model_dump(),
                    "error_type": error,
                    "metrics": counts(window, truth, prediction),
                    "seconds": time.perf_counter() - start,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "schedule_index": schedule_index,
                    "tools": executor.tools,
                    "model_requests": len(model.calls) if model else 0,
                    "cost": model.cost if model else 0.0,
                    "model_calls": model.calls if model else [],
                    "diagnostics": runtime_diagnostics(
                        prediction, executor.tools, model.calls if model else []
                    ),
                }
                write_json(result_path, result)
        finally:
            if model:
                model.close()

    workers = config.get("workers", 1)
    if workers == 1:
        for record in records:
            process(record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(process, records))
    return summarize(out)


def result_rows(root: Path) -> tuple[dict, list[dict]]:
    meta = read_json(root / "run.json")
    rows = [read_json(p) for p in sorted((root / "results").glob("*.json"))]
    if not rows or any(r["run_id"] != meta["run_id"] for r in rows):
        raise ValueError("empty or mismatched run outputs")
    return meta, rows


def summarize_rows(meta: dict, rows: list[dict]):
    """Pure aggregation also used by the read-only dashboard; never trust cached summary files."""
    if any(r["run_id"] != meta["run_id"] for r in rows):
        raise ValueError("mismatched run outputs")
    groups = defaultdict(list)
    for row in rows:
        for dimension, value in (
            ("all", "all"),
            ("kind", row["kind"]),
            ("state", row["state"]),
            ("semantic", row["semantic"]),
            ("kind_state", f"{row['kind']}:{row['state']}"),
        ):
            groups[(row["method"], row["repeat"], dimension, value)].append(row)
    detailed = [
        {"method": m, "repeat": r, "dimension": d, "value": v, **aggregate(group)}
        for (m, r, d, v), group in sorted(groups.items())
    ]
    combined = defaultdict(list)
    for row in detailed:
        combined[(row["method"], row["dimension"], row["value"])].append(row)
    averages = []
    for (method, dimension, value), trials in sorted(combined.items()):
        row = {
            "method": method,
            "dimension": dimension,
            "value": value,
            "repeats": len(trials),
            "cases_per_repeat": trials[0]["cases"],
        }
        for field in (
            "precision",
            "recall",
            "f1",
            "completion_rate",
            "mean_seconds",
            "mean_model_requests",
            "normal_false_alarm_rate_completed",
            "first_attempt_completion_rate",
            "numerical_coverage",
            "mean_logical_model_calls",
        ):
            vals = [t[field] for t in trials if t[field] is not None]
            row[field + "_mean"] = float(np.mean(vals)) if vals else None
            row[field + "_std"] = float(np.std(vals)) if vals else None
        averages.append(row)
    expected = {
        (case, method, repeat)
        for case in meta["expected_case_ids"]
        for method in meta["methods"]
        for repeat in range(1 if method in NUMERICAL_NAMES else meta["config"]["repeats"])
    }
    actual = {(r["case_id"], r["method"], r["repeat"]) for r in rows}
    if not actual <= expected or len(actual) != len(rows):
        raise ValueError("unexpected or duplicate result rows")
    result = {
        "metric_version": METRIC_VERSION,
        "records_expected": len(expected),
        "records_present": len(actual),
        "all_records_present": actual == expected,
        "run_id": meta["run_id"],
        "dataset_kind": meta["dataset_kind"],
        "note": "synthetic demo is engineering validation, not paper evidence"
        if meta["dataset_kind"] == "synthetic_demo_only"
        else "provisional development pilot; injection-only truth; not paper evidence or field accuracy"
        if meta["dataset_kind"] == "platform_pilot_provisional"
        else "controlled injection only; native background unverified, not field accuracy"
        if meta["dataset_kind"] == "daily_injection_unverified_v1"
        else "controlled injection results",
        "by_repeat": detailed,
        "summary": averages,
    }
    return result


def summarize(root: Path):
    meta, rows = result_rows(root)
    result = summarize_rows(meta, rows)
    write_json(root / "summary.json", result)
    lines = [
        "# 实验汇总",
        "",
        result["note"],
        f"输出进度：{result['records_present']}/{result['records_expected']} 条记录；技术完成率另见下表。",
        "",
        "| 方法 | F1 均值 | F1 标准差 | 有效完成率 | 每例模型请求 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["summary"]:
        if row["dimension"] == "all":

            def fmt(value):
                return f"{value:.4f}" if value is not None else "N/A"

            lines.append(
                "| "
                + " | ".join(
                    [
                        row["method"],
                        fmt(row["f1_mean"]),
                        fmt(row["f1_std"]),
                        fmt(row["completion_rate_mean"]),
                        fmt(row["mean_model_requests_mean"]),
                    ]
                )
                + " |"
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def freeze_daily_validation(dataset: Path, validation_run: Path, out: Path):
    """Attest an unchanged validation run without selecting thresholds from weak labels."""
    if out.exists():
        raise FileExistsError(out)
    manifest = load_dataset(dataset)
    meta, rows = result_rows(validation_run)
    if manifest["kind"] != "daily_injection_unverified_v1" or meta["split"] != "validation":
        raise ValueError("requires a daily benchmark validation run")
    if meta["dataset_sha256"] != file_hash(dataset / "manifest.json"):
        raise ValueError("validation run uses another dataset")
    required_methods = ["hampel", "cusum", "iforest", "visual", "daily_union"]
    expected_cases = [r["case_id"] for r in manifest["records"] if r["split"] == "validation"]
    expected = {(case_id, method, 0) for case_id in expected_cases for method in required_methods}
    actual = [(row["case_id"], row["method"], row["repeat"]) for row in rows]
    if (
        meta["methods"] != required_methods
        or meta["expected_case_ids"] != expected_cases
        or len(actual) != len(expected)
        or set(actual) != expected
        or any(r["prediction"]["status"] != "ok" for r in rows)
    ):
        raise ValueError("validation must complete all paired methods before freeze")
    config = dict(meta["config"])
    if "calibration" in config:
        raise ValueError("validation configuration must not already be calibrated")
    config["calibration"] = {
        "mode": "fixed-without-label-selection",
        "source_split": "validation",
        "dataset_sha256": meta["dataset_sha256"],
        "numerical_sha256": numerical_identity(config),
        "config_sha256": daily_config_identity(config),
        "validation_run_id": meta["run_id"],
    }
    write_json(out, config)
    return config


def calibrate(dataset: Path, config: dict, out: Path):
    validate_config(config)
    if out.exists():
        raise FileExistsError(out)
    manifest = load_dataset(dataset)
    if manifest["kind"] in ("platform_pilot_provisional", "daily_injection_unverified_v1"):
        raise ValueError("provisional pilot cannot calibrate formal thresholds")
    labels = read_json(dataset / "labels.json")
    records = [r for r in manifest["records"] if r["split"] == "validation"]
    if not records:
        raise ValueError("no validation cases")
    candidates = {
        "hampel": ("hampel_threshold", [3.0, 4.0, 5.0]),
        "cusum": ("cusum_threshold", [5.0, 8.0, 12.0]),
        "iforest": ("iforest_threshold", [0.55, 0.62, 0.7]),
    }
    report = []
    for name, (key, values) in candidates.items():
        scores = []
        for value in values:
            trial = dict(config["detectors"], **{key: value})
            totals = np.zeros(3, dtype=int)
            for record in records:
                window = Window.model_validate(read_json(child(dataset, record["file"])))
                prediction = detect(window, name, trial, config["seed"])
                metric = counts(
                    window, Prediction(events=labels[record["case_id"]]["events"]), prediction
                )
                totals += [metric["tp"], metric["fp"], metric["fn"]]
            tp, fp, fn = totals
            f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
            scores.append((f1, value))
        best_score, best_value = max(scores, key=lambda pair: pair[0])
        config["detectors"][key] = best_value
        report.append(
            {
                "method": name,
                "f1": best_score,
                "parameter": key,
                "selected": best_value,
                "trials": scores,
            }
        )
    config["fixed_detector"] = max(report, key=lambda r: r["f1"])["method"]
    config["calibration"] = {
        "source_split": "validation",
        "dataset_sha256": file_hash(dataset / "manifest.json"),
        "numerical_sha256": numerical_identity(config),
        "report": report,
    }
    write_json(out, config)
    return config


def capabilities(run: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    meta, rows = result_rows(run)
    if meta["dataset_kind"] == "platform_pilot_provisional":
        raise ValueError("provisional pilot cannot generate a capability card")
    if meta["split"] not in ("development", "validation"):
        raise ValueError("cannot build capability card from test results")
    required = {*NAMES, "visual"}
    if meta.get("limit") is not None:
        raise ValueError("capability cards require a full split, not a limited smoke run")
    baseline = [r for r in rows if r["method"] in required]
    expected = {
        (case, method, repeat)
        for case in meta["expected_case_ids"]
        for method in required
        for repeat in range(1 if method in NAMES else meta["config"]["repeats"])
    }
    actual = {(r["case_id"], r["method"], r["repeat"]) for r in baseline}
    if actual != expected or len(actual) != len(baseline):
        raise ValueError("missing/duplicate cases or repeats in capability measurements")
    case_sets = {m: {r["case_id"] for r in baseline if r["method"] == m} for m in required}
    if not all(case_sets.values()) or len({tuple(sorted(s)) for s in case_sets.values()}) != 1:
        raise ValueError("capabilities need all 3 numerical methods and visual on identical cases")
    table = []
    for method, bucket in itertools.product(sorted(required), ("complete", "missing")):
        selected = [
            r
            for r in baseline
            if r["method"] == method and (r["state"] == "complete") == (bucket == "complete")
        ]
        if not selected or not all(r["metrics"]["completed"] for r in selected):
            raise ValueError("incomplete capability measurements; inspect failures first")
        stats = aggregate(selected)
        table.append(
            {
                "method": method,
                "data_state": bucket,
                "f1": stats["f1"],
                "mean_seconds": stats["mean_seconds"],
            }
        )
    best = {
        b: max([r for r in table if r["data_state"] == b], key=lambda r: r["f1"] or 0.0)["method"]
        for b in ("complete", "missing")
    }
    strata = defaultdict(list)
    for row in baseline:
        strata[(row["method"], row["kind"], row["state"])].append(row)
    by_type = []
    for (method, kind, state), selected in sorted(strata.items()):
        stats = aggregate(selected)
        by_type.append(
            {
                "method": method,
                "kind": kind,
                "state": state,
                "f1": stats["f1"],
                "recall": stats["recall"],
                "normal_false_alarm_rate": stats["normal_false_alarm_rate_completed"],
            }
        )
    card = {
        "source_split": meta["split"],
        "dataset_sha256": meta["dataset_sha256"],
        "source_run_id": meta["run_id"],
        "best_by_missing": best,
        "measurements": table,
        "by_type_and_state": by_type,
        "detector_config": meta["config"]["detectors"],
        "seed": meta["config"]["seed"],
        "model_config": meta["config"]["model"],
        "model": meta["model"],
        "note": "Observed development results, not ground-truth anomaly types at inference.",
    }
    write_json(out, card)
    return card
