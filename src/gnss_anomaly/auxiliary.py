"""Run an anonymous unlabelled window without fabricating evaluation truth."""

import time
from pathlib import Path

from .contracts import Prediction, Window
from .detectors import NAMES
from .diagnostics import runtime_diagnostics
from .experiments import source_identity, validate_config
from .figures import figure_bytes
from .models import ModelClient
from .policies import Executor
from .storage import file_hash, read_json, write_json


def observe(source: Path, out: Path, config: dict):
    validate_config(config)
    if out.exists():
        raise FileExistsError(out)
    window = Window.model_validate(read_json(source))
    write_json(out / "input.json", window.model_dump(mode="json"))
    write_json(
        out / "manifest.json",
        {
            "source_sha256": file_hash(source),
            "input_sha256": file_hash(out / "input.json"),
            "source": source_identity(),
            "config": config,
            "ground_truth": None,
            "purpose": "unlabelled development auxiliary; no precision/recall/F1",
        },
    )
    for method in [*NAMES, "visual", "fixed"]:
        model = ModelClient(config["model"]) if method not in NAMES else None
        executor = Executor(config, model)
        start = time.perf_counter()
        try:
            prediction = executor.run(window, method)
            prediction.mask(len(window.values))
        except (ValueError, RuntimeError, KeyError, TypeError, ArithmeticError) as exc:
            prediction = Prediction(status="error", reason=str(exc)[:500])
        finally:
            if model:
                model.close()
        write_json(
            out / f"{method}.json",
            {
                "method": method,
                "prediction": prediction.model_dump(),
                "tools": executor.tools,
                "seconds": time.perf_counter() - start,
                "model_calls": model.calls if model else [],
                "cost": model.cost if model else 0.0,
                "diagnostics": runtime_diagnostics(
                    prediction, executor.tools, model.calls if model else []
                ),
            },
        )
        for fmt in ("png", "svg", "pdf"):
            (out / f"{method}.{fmt}").write_bytes(
                figure_bytes(None, "snapshot", fmt, case=(window, None, prediction))
            )
        print(f"{method}: {prediction.status}, {len(prediction.events)} intervals", flush=True)
